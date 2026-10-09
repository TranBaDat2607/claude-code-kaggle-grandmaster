"""Paired comparisons, accepted baseline / lineage, backlog, and their CLI commands."""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from kgkit import state as S
from kgkit.backlog import Backlog, score
from kgkit.compare import compare_folds, paired_bootstrap, verdict
from kgkit.experiment import Ledger


def run(args, cwd, plugin_root, check=True):
    env = dict(os.environ, PYTHONPATH=str(plugin_root))
    r = subprocess.run([sys.executable, "-m", "kgkit", *args], cwd=cwd, env=env, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(r.stdout + r.stderr)
    return r


# ----------------------------------------------------------------------------- compare
def test_paired_folds_detect_gain_hidden_by_fold_std():
    # folds differ a lot in difficulty (fold std 0.02) but the new model wins every fold by ~0.003
    base = [0.80, 0.84, 0.78, 0.83, 0.81]
    new = [b + 0.003 + e for b, e in zip(base, [0.0004, -0.0003, 0.0002, -0.0001, 0.0])]
    f = compare_folds(new, base, greater_is_better=True)
    assert f["folds_improved"] == 5
    assert f["gain"] < f["base_fold_std"]  # the old "beat one fold std" rule would discard it
    assert f["t"] > 10
    assert verdict(f)["decision"] == "KEEP"


def test_compare_lower_is_better_and_discard():
    f = compare_folds([0.51, 0.52, 0.50], [0.50, 0.50, 0.49], greater_is_better=False)  # rmse went up
    assert f["gain"] < 0 and verdict(f)["decision"] == "DISCARD"


def test_inconclusive_when_within_noise():
    f = compare_folds([0.801, 0.795, 0.812, 0.790, 0.806], [0.800, 0.799, 0.805, 0.795, 0.803])
    assert f["gain"] > 0
    assert verdict(f)["decision"] == "INCONCLUSIVE"


def test_large_jump_is_flagged_for_leak_audit():
    f = compare_folds([0.95, 0.96, 0.95, 0.94, 0.96], [0.80, 0.81, 0.80, 0.79, 0.81])
    v = verdict(f)
    assert v["decision"] == "KEEP" and any("leak" in x for x in v["flags"])


def test_paired_bootstrap_sees_small_real_gain_and_groups():
    rng = np.random.default_rng(0)
    n = 6000
    y = rng.integers(0, 2, n)
    s = y + rng.normal(0, 1.2, n)
    base = s + rng.normal(0, 0.8, n)
    new = s + rng.normal(0, 0.3, n)  # strictly less noise
    b = paired_bootstrap(y, new, base, "auc", n_boot=200)
    assert b["gain"] > 0 and b["z"] > 2 and b["p_better"] > 0.95
    g = paired_bootstrap(y, new, base, "auc", n_boot=100, groups=np.arange(n) // 3)
    assert g["n_boot"] == 100 and g["se"] > 0
    m = paired_bootstrap(y, new, base, "auc", n_boot=50, mask=np.arange(n) < 1000)
    assert m["n_rows"] == 1000


# ----------------------------------------------------------------------------- ledger decisions
def test_baseline_is_last_accepted_not_best(tmp_path):
    S.CompetitionState(slug="x", metric="auc").save(tmp_path)
    led = Ledger(tmp_path)
    a = led.log("base", 0.80, fold_scores=[0.8] * 5, decision="baseline")
    b = led.log("lucky", 0.83, fold_scores=[0.83] * 5, parent=a["id"])  # higher CV, never accepted
    c = led.log("te", 0.805, fold_scores=[0.805] * 5, parent=a["id"])
    assert led.baseline()["id"] == a["id"]
    led.decide(b["id"], "discard", "leaky", parent=a["id"])
    led.decide(c["id"], "keep", "+0.005 5/5")
    assert led.baseline()["id"] == c["id"]
    assert [r["id"] for r in led.lineage(c["id"])] == [a["id"], c["id"]]
    assert led.decisions_on_folds(None) == 2
    with pytest.raises(ValueError):
        led.decide(c["id"], "maybe")
    # an unknown parent (e.g. a local id named inside a remote Kaggle run) is kept verbatim
    d = led.log("remote", 0.81, parent="0042")
    assert d["parent"] == "0042" and [r["id"] for r in led.lineage(d["id"])] == [d["id"]]


def test_baseline_falls_back_to_best(tmp_path):
    S.CompetitionState(slug="x", metric="rmse", greater_is_better=False).save(tmp_path)
    led = Ledger(tmp_path)
    led.log("a", 0.5)
    b = led.log("b", 0.4)
    assert led.baseline()["id"] == b["id"] and led.baseline(fallback=False) is None


def test_import_preds_recomputes_cv_on_our_folds(tmp_path):
    S.CompetitionState(slug="x", metric="auc").save(tmp_path)
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    oof = y * 0.6 + rng.random(500) * 0.5
    folds = np.arange(500) % 5
    rec = Ledger(tmp_path).import_preds("teammate_cnn", oof, rng.random(100), y_true=y, folds=folds,
                                         source="teammate")
    assert len(rec["fold_scores"]) == 5 and rec["cv"] > 0.8 and "teammate" in rec["tags"]
    assert (tmp_path / rec["test_path"]).is_file()


# ----------------------------------------------------------------------------- backlog
def test_backlog_ranking_status_and_fidelity(tmp_path):
    bl = Backlog(tmp_path)
    a = bl.add("bigger backbone", gain=3, prob=0.6, cost=6)
    b = bl.add("fix label noise in class 3", gain=4, prob=0.5, cost=1, evidence="40% of top losses", source="error-analysis")
    c = bl.add("TTA hflip", gain=1, prob=0.7, cost=0.5)
    assert bl.add("Bigger backbone ")["id"] == a["id"]  # no duplicates
    assert [i["id"] for i in bl.ranked()] == [b["id"], c["id"], a["id"]]
    assert score(b) == pytest.approx(2.0)
    bl.set(b["id"], status="done", exp="0007", result="KEEP +0.004", screen_gain=0.005, full_gain=0.004)
    assert b["id"] not in [i["id"] for i in bl.ranked()]
    bl.set(c["id"], status="done", screen_gain=0.001, full_gain=-0.0002)
    assert bl.fidelity() is None
    bl.set(a["id"], status="done", screen_gain=0.003, full_gain=0.002)
    f = bl.fidelity()
    assert f["n"] == 3 and f["sign_agreement"] == pytest.approx(2 / 3)
    md = bl.render().read_text(encoding="utf-8")
    assert "fix label noise" in md and "Screen fidelity" in md
    with pytest.raises(ValueError):
        bl.set(a["id"], status="maybe")


# ----------------------------------------------------------------------------- CLI
@pytest.fixture
def ws(tmp_path, plugin_root, binary_df):
    run(["init", "demo", "--metric", "auc", "--target", "target", "--id-col", "id"], tmp_path, plugin_root)
    binary_df.to_csv(tmp_path / "data" / "train.csv", index=False)
    run(["folds", "data/train.csv", "--target", "target", "--id-col", "id", "--out", "data/folds.csv"],
        tmp_path, plugin_root)
    return tmp_path


def test_cli_compare_decide_lineage_import_backlog(ws, plugin_root, binary_df):
    y = binary_df["target"].to_numpy()
    folds = pd.read_csv(ws / "data" / "folds.csv")["fold"].to_numpy()
    rng = np.random.default_rng(3)
    signal = 1.5 * binary_df["x1"] - binary_df["x2"]
    base = signal + rng.normal(0, 1.0, len(y))
    new = signal + rng.normal(0, 0.5, len(y))
    np.save(ws / "base.npy", base)
    np.save(ws / "new.npy", new)
    r = run(["ledger", "import", "base", "base.npy", "--truth", "data/train.csv:target", "--folds",
             "data/folds.csv:fold"], ws, plugin_root)
    assert "logged 0001_base" in r.stdout
    run(["ledger", "decide", "0001", "baseline"], ws, plugin_root)
    run(["ledger", "import", "new", "new.npy", "--truth", "data/train.csv:target", "--folds", "data/folds.csv:fold",
         "--source", "teammate"], ws, plugin_root)
    r = run(["ledger", "compare", "0002", "--truth", "data/train.csv:target", "--folds", "data/folds.csv:fold",
             "--boot", "100"], ws, plugin_root)
    assert "KEEP" in r.stdout and "0001_base" in r.stdout and "bootstrap" in r.stdout
    r = run(["ledger", "compare", "0002", "0001"], ws, plugin_root)  # from stored fold scores
    assert "folds:" in r.stdout
    # a 1-fold screen stored as per-fold artefacts, compared on its fold only
    d = ws / "artifacts" / "screen"
    d.mkdir(parents=True)
    idx = np.flatnonzero(folds == 0)
    np.save(d / "fold0_oof.npy", new[idx])
    np.save(d / "fold0_idx.npy", idx)
    r = run(["ledger", "compare", "artifacts/screen", "0001", "--truth", "data/train.csv:target", "--folds",
             "data/folds.csv:fold", "--fold", "0", "--boot", "100"], ws, plugin_root)
    assert f"{len(idx)} rows" in r.stdout
    run(["ledger", "decide", "0002", "keep", "less", "noise", "--parent", "0001"], ws, plugin_root)
    r = run(["ledger", "lineage"], ws, plugin_root)
    assert "0001_base" in r.stdout and "0002_new" in r.stdout
    r = run(["status"], ws, plugin_root)
    assert "accepted baseline" in r.stdout and "0002_new" in r.stdout
    run(["backlog", "add", "count", "encode", "pairs", "--gain", "3", "--prob", "0.5", "--cost", "1"], ws, plugin_root)
    r = run(["backlog", "list"], ws, plugin_root)
    assert "count encode pairs" in r.stdout
    run(["backlog", "set", "1", "--status", "done", "--exp", "0002", "--result", "KEEP"], ws, plugin_root)
    run(["backlog", "render"], ws, plugin_root)
    assert "count encode pairs" in (ws / "reports" / "backlog.md").read_text(encoding="utf-8")
    data = json.loads((ws / ".kaggle-gm" / "backlog.json").read_text(encoding="utf-8"))
    assert data[0]["exp_ids"] == ["0002"]


def test_cli_blend_stack_residual_prune(ws, plugin_root, binary_df):
    y = binary_df["target"].to_numpy()
    rng = np.random.default_rng(5)
    signal = 1.5 * binary_df["x1"] - binary_df["x2"] + (binary_df["cat"] == "a")
    for i in range(3):
        np.save(ws / f"m{i}.npy", 1 / (1 + np.exp(-(signal + rng.normal(0, 1.0, len(y))))))
        np.save(ws / f"t{i}.npy", rng.random(50))
        run(["ledger", "import", f"m{i}", f"m{i}.npy", "--test", f"t{i}.npy", "--truth", "data/train.csv:target",
             "--folds", "data/folds.csv:fold"], ws, plugin_root)
    np.save(ws / "dup.npy", np.load(ws / "m0.npy"))
    run(["ledger", "import", "dup", "dup.npy", "--test", "t0.npy", "--truth", "data/train.csv:target"], ws, plugin_root)
    common = ["--exp", "0001", "0002", "0003", "0004", "--truth", "data/train.csv:target", "--folds", "data/folds.csv:fold"]
    r = run(["blend", *common, "--method", "stack", "--levels", "2", "--prune", "10", "--name", "stk"], ws, plugin_root)
    assert "pruned 0004: corr" in r.stdout and "level 3" in r.stdout and "honest" in r.stdout
    r = run(["blend", *common, "--method", "residual", "--name", "res"], ws, plugin_root)
    assert "residual stack on base" in r.stdout
    recs = [json.loads(l) for l in (ws / ".kaggle-gm" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert recs[-2]["params"]["method"] == "stack" and recs[-1]["params"]["method"] == "residual"
    r = run(["blend", "--exp", "0001", "0002", "--truth", "data/train.csv:target", "--method", "stack"], ws,
            plugin_root, check=False)
    assert r.returncode != 0 and "--folds" in (r.stdout + r.stderr)
