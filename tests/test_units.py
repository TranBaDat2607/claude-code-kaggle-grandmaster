"""Per-unit scores (#136): comparisons for metrics that are not one array of row predictions.

A record carries a unit table: one row per thing its metric is computed over (an image, a study, a tomogram, a query,
a problem, a match), with the values the metric is built from. Means of per-unit scores, ratios of summed counts
(dataset-level IoU / Dice, micro and per-class F-beta) and any custom aggregate are compared paired, by resampling
units (or whole groups of them), like the OOF bootstrap does with rows."""

import json

import numpy as np
import pandas as pd
import pytest

from kgkit import cli
from kgkit import units as U
from kgkit.experiment import Ledger, report_comparison


# ---------------------------------------------------------------------------- aggregates
def test_mean_is_the_weighted_mean_of_the_unit_scores():
    u = U.make(["a", "b", "c"], score=[1.0, 0.0, 0.5], weight=[1, 1, 2])
    assert u.agg == "mean" and u.score() == pytest.approx((1 + 0 + 1.0) / 4)
    assert U.make([1, 2], score=[0.2, 0.4]).score() == pytest.approx(0.3)


def test_summed_counts_give_dataset_level_iou_dice_and_fbeta():
    rng = np.random.default_rng(0)
    y = rng.random((6, 20, 20)) < 0.3
    p = rng.random((6, 20, 20)) < 0.3
    inter = (y & p).sum(axis=(1, 2))
    union = (y | p).sum(axis=(1, 2))
    iou = U.make(range(6), counts={"inter": inter, "union": union})
    assert iou.agg == "iou" and iou.score() == pytest.approx((y & p).sum() / (y | p).sum())
    dice = U.make(range(6), counts={"inter": inter, "pred": p.sum(axis=(1, 2)), "true": y.sum(axis=(1, 2))})
    assert dice.agg == "dice" and dice.score() == pytest.approx(2 * (y & p).sum() / (y.sum() + p.sum()))
    tp, fp, fn = np.array([8, 3, 0]), np.array([2, 1, 4]), np.array([1, 5, 2])
    f4 = U.make(["t1", "t2", "t3"], counts={"tp": tp, "fp": fp, "fn": fn}, params={"beta": 4})
    T, P, N = tp.sum(), fp.sum(), fn.sum()
    assert f4.agg == "fbeta" and f4.score() == pytest.approx(17 * T / (17 * T + 16 * N + P))
    assert U.make(["t1"], counts={"tp": [3], "fp": [1]}, agg="precision").score() == pytest.approx(0.75)
    assert U.make(["t1"], counts={"tp": [3], "fn": [3]}, agg="recall").score() == pytest.approx(0.5)
    assert U.make(["t1", "t2"], counts={"num": [1, 2], "den": [4, 4]}).score() == pytest.approx(3 / 8)


def test_per_class_fbeta_is_the_weighted_mean_of_each_class_fbeta():
    """CZII-style: F-beta per particle type from counts summed over tomograms, then a weighted mean over types."""
    counts = {"tp_apo": [5, 5], "fp_apo": [1, 1], "fn_apo": [0, 2], "tp_rib": [1, 0], "fp_rib": [3, 3], "fn_rib": [4, 4]}
    u = U.make(["t1", "t2"], counts=counts, params={"beta": 4, "class_weights": {"apo": 1, "rib": 3}})

    def fb(t, p, n, b=4):
        return (1 + b * b) * t / ((1 + b * b) * t + b * b * n + p)

    want = (1 * fb(10, 2, 2) + 3 * fb(1, 6, 8)) / 4
    assert u.agg == "fbeta" and u.score() == pytest.approx(want)


def test_a_custom_aggregate_is_called_with_the_columns_and_the_draw_weights(tmp_path, monkeypatch):
    (tmp_path / "mymetric.py").write_text(
        "import numpy as np\n"
        "def score(cols, w):\n"
        "    return float(np.sum(w * cols['hits']) / np.sum(w * cols['n']))\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    u = U.make(["a", "b"], counts={"hits": [1, 3], "n": [2, 4]}, agg="custom",
               params={"fn": "mymetric:score"}, greater_is_better=True)
    assert u.score() == pytest.approx(4 / 6)
    with pytest.raises(ValueError, match="greater_is_better"):
        U.make(["a"], counts={"hits": [1], "n": [2]}, agg="custom", params={"fn": "mymetric:score"})


def test_missing_columns_and_unknown_aggregates_are_refused():
    with pytest.raises(ValueError, match="fn"):
        U.make(["a"], counts={"tp": [1], "fp": [0]}, agg="fbeta")
    with pytest.raises(ValueError, match="unknown"):
        U.make(["a"], counts={"x": [1]}, agg="median")
    with pytest.raises(ValueError, match="duplicate"):
        U.make(["a", "a"], score=[1, 2])


def test_fold_scores_aggregate_each_fold_separately():
    u = U.make(list("abcd"), counts={"tp": [1, 1, 2, 0], "fp": [1, 0, 0, 2], "fn": [0, 1, 0, 0]}, fold=[0, 0, 1, 1])
    f0 = U.make(list("ab"), counts={"tp": [1, 1], "fp": [1, 0], "fn": [0, 1]}).score()
    f1 = U.make(list("cd"), counts={"tp": [2, 0], "fp": [0, 2], "fn": [0, 0]}).score()
    assert u.fold_scores() == pytest.approx([f0, f1])
    assert U.make(["a"], score=[1.0]).fold_scores() is None


# ---------------------------------------------------------------------------- paired noise
def _tomograms(rng, n, recall_a, recall_b):
    """Detection counts per tomogram for two pickers sharing each tomogram's difficulty."""
    d = rng.uniform(0.5, 1.0, n)
    true = rng.poisson(30, n) + 1
    out = []
    for r in (recall_a, recall_b):
        tp = rng.binomial(true, np.clip(d * r, 0, 1))
        fp = rng.poisson(6 * (1.1 - d))
        out.append(U.make(np.arange(n), counts={"tp": tp, "fp": fp, "fn": true - tp}, params={"beta": 4}))
    return out


def test_the_paired_unit_bootstrap_se_matches_the_spread_of_the_gain():
    """Over many datasets from one process, the bootstrap SE of the gain is about the gain's true spread, and under
    no difference about 5% of the comparisons land beyond 1.96 SE."""
    rng = np.random.default_rng(1)
    gains = [(lambda a, b: a.score() - b.score())(*_tomograms(rng, 80, 0.80, 0.78)) for _ in range(400)]
    sd = float(np.std(gains, ddof=1))
    ses = [U.compare(*_tomograms(rng, 80, 0.80, 0.78), n_boot=400, seed=i)["se"] for i in range(40)]
    assert 0.8 < float(np.mean(ses)) / sd < 1.25
    zs = [U.compare(*_tomograms(rng, 80, 0.8, 0.8), n_boot=300, seed=i)["z"] for i in range(200)]
    assert np.mean(np.abs(zs) > 1.96) < 0.11


def test_a_real_gain_is_found_and_the_result_has_the_paired_fields():
    rng = np.random.default_rng(2)
    a, b = _tomograms(rng, 120, 0.85, 0.70)
    res = U.compare(a, b, n_boot=500)
    assert res["question"] == "paired" and res["gain"] == pytest.approx(a.score() - b.score())
    assert res["z"] > 4 and res["p_better"] > 0.99 and res["unit"] == "units" and res["n_units"] == 120
    assert "bootstrap" in res["method"]
    lower = U.make(np.arange(3), score=[0.3, 0.2, 0.1], greater_is_better=False)
    higher = U.make(np.arange(3), score=[0.4, 0.3, 0.2], greater_is_better=False)
    assert U.compare(lower, higher, n_boot=50)["gain"] == pytest.approx(0.1)  # positive = better, for a loss


def test_groups_are_resampled_whole():
    """Several images per patient whose differences share the patient: resampling images alone understates the SE."""
    rng = np.random.default_rng(3)
    n_pat, per = 40, 8
    group = np.repeat(np.arange(n_pat), per)
    shift = rng.normal(0, 0.05, n_pat)[group]
    base = rng.uniform(0.6, 0.9, len(group))
    ids = np.arange(len(group))
    a = U.make(ids, score=base + 0.01 + shift + rng.normal(0, 0.01, len(group)), group=group)
    b = U.make(ids, score=base)
    a_flat = U.make(ids, score=a.cols["score"])
    grouped, flat = U.compare(a, b, n_boot=500), U.compare(a_flat, b, n_boot=500)
    assert grouped["unit"] == "groups" and grouped["n_units"] == n_pat
    assert grouped["se"] > 1.8 * flat["se"]


def test_tables_are_paired_by_id_and_must_hold_the_same_units():
    a = U.make(["x", "y", "z"], score=[0.9, 0.5, 0.1])
    b = U.make(["z", "x", "y"], score=[0.1, 0.8, 0.5])
    assert U.compare(a, b, n_boot=50)["gain"] == pytest.approx(0.1 / 3)
    with pytest.raises(ValueError, match="same units"):
        U.compare(a, U.make(["x", "y", "w"], score=[1, 1, 1]))
    with pytest.raises(ValueError, match="aggregate"):
        U.compare(a, U.make(["x", "y", "z"], counts={"num": [1, 1, 1], "den": [2, 2, 2]}))


def test_absolute_noise_of_one_score():
    rng = np.random.default_rng(4)
    a, _ = _tomograms(rng, 100, 0.8, 0.8)
    res = U.compare(a, None, n_boot=300)
    assert res["question"] == "absolute" and res["value"] == pytest.approx(a.score()) and res["se"] > 0


def test_save_and_load_keep_string_ids_groups_and_folds(tmp_path):
    u = U.make(["s1", "s2", "s3"], counts={"tp": [1, 2, 3], "fp": [0, 1, 0], "fn": [1, 0, 0]}, params={"beta": 2},
               group=["p1", "p1", "p2"], fold=[0, 1, 1])
    path = tmp_path / "units.npz"
    spec = u.save(path)
    assert spec["agg"] == "fbeta" and spec["params"] == {"beta": 2} and spec["n"] == 3 and spec["n_groups"] == 2
    back = U.load(path, spec)
    assert list(back.ids) == ["s1", "s2", "s3"] and list(back.group) == ["p1", "p1", "p2"]
    assert back.score() == pytest.approx(u.score()) and back.fold_scores() == pytest.approx(u.fold_scores())
    assert spec["hash"] == U.make(["s3", "s1", "s2"], score=[0, 0, 0]).digest()  # the unit set, not its order
    with np.load(path, allow_pickle=False) as z:  # no pickled objects in the file
        assert set(z.files) >= {"id", "tp", "fp", "fn", "group", "fold"}


def test_a_table_read_from_a_file(tmp_path):
    df = pd.DataFrame({"id": ["a", "b"], "tp": [3, 1], "fp": [1, 1], "fn": [0, 2], "group": ["g", "g"]})
    df.to_csv(tmp_path / "u.csv", index=False)
    u = U.read(tmp_path / "u.csv", params={"beta": 1})
    assert u.agg == "fbeta" and list(u.group) == ["g", "g"] and u.score() == pytest.approx(2 * 4 / (2 * 4 + 2 + 2))
    with pytest.raises(ValueError, match="id"):
        pd.DataFrame({"x": [1]}).to_csv(tmp_path / "bad.csv", index=False)
        U.read(tmp_path / "bad.csv")


# ---------------------------------------------------------------------------- the ledger and the CLI
@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli.main(["init", "czii", "--metric", "fbeta2", "--target", "target", "--id-col", "id"])
    return tmp_path


def _write(ws, name, u: U.Units) -> str:
    df = pd.DataFrame({"id": u.ids, **u.cols, **({"fold": u.fold} if u.fold is not None else {})})
    df.to_csv(ws / f"{name}.csv", index=False)
    return f"{name}.csv"


def test_ledger_log_units_records_the_score_and_the_table(ws):
    rng = np.random.default_rng(5)
    a, _ = _tomograms(rng, 30, 0.8, 0.8)
    a = U.make(a.ids, counts=a.cols, params={"beta": 4}, fold=np.arange(30) % 3)
    led = Ledger(ws)
    rec = led.log_units("unet", a, metric="czii_fbeta4", notes="custom pipeline")
    assert rec["cv"] == pytest.approx(a.score()) and rec["fold_scores"] == pytest.approx(a.fold_scores())
    assert rec["units"]["agg"] == "fbeta" and rec["units"]["n"] == 30 and rec["metric"] == "czii_fbeta4"
    assert (ws / rec["units"]["path"]).is_file()
    assert led.load_units(rec["id"]).score() == pytest.approx(a.score())


def test_cli_add_compare_and_noise_on_unit_tables(ws, capsys):
    rng = np.random.default_rng(6)
    a, b = _tomograms(rng, 150, 0.70, 0.86)
    folds = np.arange(150) % 5
    a = U.make(a.ids, counts=a.cols, params={"beta": 4}, fold=folds)
    b = U.make(b.ids, counts=b.cols, params={"beta": 4}, fold=folds)
    cli.main(["ledger", "add", "base", "auto", "--units", _write(ws, "base", a), "--units-param", "beta=4"])
    out = capsys.readouterr().out
    assert "fbeta" in out and "150 units" in out
    rec = Ledger(ws).get("0001")
    assert rec["cv"] == pytest.approx(a.score()) and rec.get("oof_path") is None
    cli.main(["ledger", "decide", "0001", "baseline"])
    cli.main(["ledger", "add", "better", "auto", "--units", _write(ws, "better", b), "--units-param", "beta=4",
              "--parent", "0001"])
    out = capsys.readouterr().out
    assert "0002_better vs 0001_base: KEEP" in out and "150 units" in out  # compared when it was added
    cli.main(["ledger", "compare", "0002", "0001"])
    out = capsys.readouterr().out
    assert "KEEP" in out and "folds: gain" in out and "unit bootstrap" in out
    comp = [json.loads(line) for line in (ws / ".kaggle-gm" / "comparisons.jsonl").read_text().splitlines()][-1]
    assert comp["unit"] == "units" and comp["n_units"] == 150 and comp["decision"] == "KEEP"
    cli.main(["ledger", "noise", "0002"])
    assert "150 units" in capsys.readouterr().out
    cli.main(["exp", "finish", "0002", "keep", "better picker"])
    assert Ledger(ws).get("0002")["decision"] == "keep"


def test_cli_refuses_tables_of_other_units_and_a_typed_score_that_disagrees(ws, capsys):
    a = U.make(["t1", "t2", "t3"], counts={"tp": [3, 2, 1], "fp": [0, 1, 1], "fn": [1, 1, 1]})
    cli.main(["ledger", "add", "base", "0.5", "--units", _write(ws, "base", a)])
    assert "recomputed from the units" in capsys.readouterr().out
    assert Ledger(ws).get("0001")["cv"] == pytest.approx(a.score())
    other = U.make(["t1", "t2", "t9"], counts={"tp": [3, 2, 1], "fp": [0, 1, 1], "fn": [1, 1, 1]})
    cli.main(["ledger", "add", "other", "auto", "--units", _write(ws, "other", other)])
    with pytest.raises(SystemExit, match="same units"):
        cli.main(["ledger", "compare", "0002", "0001"])


def test_a_custom_pipeline_gets_the_verdict_when_it_logs(ws, capsys):
    rng = np.random.default_rng(7)
    a, b = _tomograms(rng, 100, 0.7, 0.88)
    led = Ledger(ws)
    led.log_units("base", a, decision="baseline")
    rec = led.log_units("better", b, parent="0001")
    report_comparison(rec)
    out = capsys.readouterr().out
    assert "0002_better vs 0001_base: KEEP" in out and "kgkit exp finish 0002_better keep" in out


def test_units_and_best_of_n_work_together(ws, capsys):
    rng = np.random.default_rng(8)
    a, b = _tomograms(rng, 100, 0.80, 0.83)
    led = Ledger(ws)
    led.log_units("base", a, decision="baseline")
    led.log_units("small", b, parent="0001")
    cli.main(["ledger", "compare", "0002", "0001", "--tries", "500"])
    out = capsys.readouterr().out
    assert "best of 500" in out
