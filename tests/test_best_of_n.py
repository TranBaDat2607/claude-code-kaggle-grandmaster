"""Best of N (#134): a run promoted from many tries against one baseline is the luckiest of N noisy draws, so the paired
test that promotes it asks for more than the usual 2 SE: the z that pure noise reaches with probability alpha as the
best of N (Sidak), where alpha is what z = 2 means for one comparison."""

import json

import numpy as np
import pytest

from kgkit import cli
from kgkit.compare import Z_KEEP, best_of_n_z, expected_max, rejudge, verdict
from kgkit.experiment import Ledger


# ---------------------------------------------------------------------------- the noise of a maximum
@pytest.mark.parametrize("n, want", [(1, 0.0), (2, 0.5642), (10, 1.5388), (50, 2.2491), (100, 2.5076)])
def test_expected_max_of_n_standard_normals(n, want):
    assert expected_max(n) == pytest.approx(want, abs=2e-3)


def test_best_of_n_z_is_the_usual_threshold_for_one_try_and_grows_with_n():
    assert best_of_n_z(1) == pytest.approx(Z_KEEP)
    zs = [best_of_n_z(n) for n in (1, 2, 5, 10, 20, 50, 100, 850)]
    assert zs == sorted(zs) and len(set(zs)) == len(zs)
    assert best_of_n_z(10) == pytest.approx(2.83, abs=0.02)
    assert best_of_n_z(100) == pytest.approx(3.50, abs=0.02)


def _boot(z: float, se: float = 0.001) -> dict:
    return {"gain": z * se, "se": se, "z": z, "p_better": 0.5, "n_rows": 1000, "method": "bootstrap"}


def test_the_best_of_n_null_runs_passes_no_more_often_than_one_null_run():
    """N runs with no real gain: without the correction their best passes about 1 - (1 - alpha)^N of the time; with
    it, about alpha, the rate of a single pre-planned comparison."""
    rng = np.random.default_rng(0)
    n, sims = 20, 4000
    best = rng.normal(size=(sims, n)).max(axis=1)
    naive = np.mean([verdict(None, _boot(z))["decision"] == "KEEP" for z in best])
    gated = np.mean([verdict(None, _boot(z), tries=n)["decision"] == "KEEP" for z in best])
    alpha = 1 - 0.5 * (1 + __import__("math").erf(Z_KEEP / 2 ** 0.5))
    assert naive > 0.25
    assert gated < 1.6 * alpha


@pytest.mark.parametrize("tries", [1, 20])
def test_a_real_gain_of_a_few_se_still_passes(tries):
    assert verdict(None, _boot(4.5), tries=tries)["decision"] == "KEEP"


def test_the_verdict_says_how_many_tries_it_corrected_for():
    one = verdict(None, _boot(2.5))
    assert one["decision"] == "KEEP" and one["tries"] == 1
    assert not any("best of" in r for r in one["reasons"])
    many = verdict(None, _boot(2.5), tries=12)
    assert many["decision"] == "INCONCLUSIVE" and many["tries"] == 12
    assert many["z_needed"] == pytest.approx(best_of_n_z(12))
    line = next(r for r in many["reasons"] if r.startswith("best of 12"))
    assert f"{expected_max(12):.2f}" in line and f"{best_of_n_z(12):.2f}" in line
    assert many["strength"] == pytest.approx(2.5)


def test_fold_t_counts_as_the_strength_when_it_is_the_stronger_test():
    folds = {"k": 5, "gain": 0.004, "per_fold": [0.004] * 5, "folds_improved": 5, "diff_sd": 0.001, "se": 0.0004,
             "t": 10.0, "base_fold_std": 0.01}
    v = verdict(folds, _boot(2.2), tries=30)
    assert v["decision"] == "KEEP" and v["strength"] == pytest.approx(10.0)


def test_rejudge_a_logged_comparison_with_more_tries():
    logged = {"decision": "KEEP", "gain": 0.003, "se": 0.001, "strength": 3.0, "tries": 1}
    assert rejudge(logged, 1)["decision"] == "KEEP"
    later = rejudge(logged, 40)
    assert later["decision"] == "INCONCLUSIVE" and later["tries"] == 40
    assert any(r.startswith("best of 40") for r in later["reasons"])
    assert rejudge({**logged, "decision": "DISCARD"}, 40)["decision"] == "DISCARD"
    # an entry logged before the strength was stored: gain / se stands in
    assert rejudge({"decision": "KEEP", "gain": 0.003, "se": 0.001}, 40)["decision"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------- counting the tries
def test_tries_counts_the_children_of_one_baseline_on_one_split(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    led = Ledger(tmp_path)
    folds, other = [0, 1, 2, 3, 4] * 4, [4, 3, 2, 1, 0] * 4
    led.log("base", 0.80, folds=folds, decision="baseline")
    for i in range(4):
        led.log(f"idea{i}", 0.80 + i / 1000, folds=folds, parent="0001", variant=f"idea{i}")
    led.log("idea0_seed2", 0.801, folds=folds, parent="0001", variant="idea0")  # a re-run of a try: no new try
    monkeypatch.setenv("KG_RECHECK", "7")
    led.log("idea1_recheck", 0.80, folds=folds, parent="0001")  # on re-drawn folds
    monkeypatch.delenv("KG_RECHECK")
    led.log("idea_other_split", 0.81, folds=other, parent="0001")
    led.log("lb_blend", None, parent="0001")  # no score of its own
    led.log("grandchild", 0.82, folds=folds, parent="0002")
    assert led.tries("0003", "0001") == 4
    assert led.tries("0006", "0001") == 4  # the replica counts as its own pipeline's try
    assert led.tries("0010", "0002") == 1
    # a record with no parent is counted with the children of the baseline it is compared with
    led.log("orphan", 0.80, folds=folds)
    assert led.tries("0011", "0001") == 5


def test_a_sweep_counts_with_the_ideas_tried_before_it_against_the_same_baseline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    led = Ledger(tmp_path)
    folds = [0, 1, 2, 3, 4] * 4
    led.log("base", 0.80, folds=folds, decision="baseline")
    led.log("before", 0.80, folds=folds, parent="0001", decision="discard")
    for lr in (0.01, 0.03, 0.1):
        led.log(f"lr{lr}", 0.80, folds=folds, parent="0001", params={"lr": lr})
    assert led.tries("0004", "0001") == 4
    # a new accepted baseline starts its own count
    led.log("kept", 0.81, folds=folds, parent="0001", decision="keep")
    led.log("next", 0.81, folds=folds, parent="0006")
    assert led.tries("0007", "0006") == 1


# ---------------------------------------------------------------------------- the CLI
@pytest.fixture
def ws(tmp_path, monkeypatch, binary_df):
    monkeypatch.chdir(tmp_path)
    cli.main(["init", "demo", "--metric", "auc", "--target", "target", "--id-col", "id"])
    binary_df.to_csv(tmp_path / "data" / "train.csv", index=False)
    cli.main(["folds", "data/train.csv", "--target", "target", "--id-col", "id", "--out", "data/folds.csv"])
    return tmp_path


SPEC = ["--truth", "data/train.csv:target", "--folds", "data/folds.csv:fold"]


def _comparisons(ws) -> list[dict]:
    p = ws / ".kaggle-gm" / "comparisons.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()] if p.is_file() else []


def _pair(ws, binary_df, capsys) -> None:
    """0001_base (the accepted baseline) and 0002_new, a child with a moderate real gain (paired z about 3)."""
    rng = np.random.default_rng(3)
    signal = 1.5 * binary_df["x1"] - binary_df["x2"]
    np.save(ws / "base.npy", signal + rng.normal(0, 1.0, len(signal)))
    np.save(ws / "new.npy", signal + rng.normal(0, 0.85, len(signal)))
    cli.main(["ledger", "import", "base", "base.npy", *SPEC])
    cli.main(["ledger", "decide", "0001", "baseline"])
    cli.main(["ledger", "import", "new", "new.npy", *SPEC])
    Ledger(ws).update("0002", parent="0001_base")
    capsys.readouterr()


def test_ledger_compare_corrects_for_the_siblings_of_the_new_run(ws, binary_df, capsys):
    _pair(ws, binary_df, capsys)
    cli.main(["ledger", "compare", "0002", "0001"])
    out = capsys.readouterr().out
    assert "0002_new vs 0001_base: KEEP" in out and "best of" not in out
    assert _comparisons(ws)[-1]["tries"] == 1 and _comparisons(ws)[-1]["strength"] >= Z_KEEP
    cli.main(["ledger", "compare", "0002", "0001", "--tries", "1000"])
    out = capsys.readouterr().out
    assert "0002_new vs 0001_base: INCONCLUSIVE" in out and "best of 1000" in out
    assert _comparisons(ws)[-1]["tries"] == 1000


def test_exp_finish_rejudges_the_logged_comparison_once_more_runs_were_tried(ws, binary_df, capsys):
    """The template compared the run when it logged, before the rest of the sweep ran: finishing it counts the tries
    as they are now."""
    _pair(ws, binary_df, capsys)
    cli.main(["ledger", "compare", "0002", "0001"])
    assert "KEEP" in capsys.readouterr().out
    n = len(_comparisons(ws))
    cli.main(["exp", "finish", "0002", "--tries", "1000"])
    out = capsys.readouterr().out
    assert "compared when it was logged" in out and "best of 1000" in out and "INCONCLUSIVE" in out
    assert 'kgkit exp finish 0002_new inconclusive "<why>"' in out
    assert len(_comparisons(ws)) == n  # no bootstrap re-run
    cli.main(["exp", "finish", "0002"])  # one try: the logged KEEP stands
    out = capsys.readouterr().out
    assert "KEEP" in out and "best of" not in out
