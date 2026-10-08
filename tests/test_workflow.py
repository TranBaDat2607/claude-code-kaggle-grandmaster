"""End-to-end: init a workspace, make folds, log experiments, blend, validate a submission."""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from kgkit import state as S
from kgkit.experiment import Ledger
from kgkit.submission import validate_submission


def run(args, cwd, plugin_root, check=True):
    env = dict(os.environ, PYTHONPATH=str(plugin_root))
    r = subprocess.run([sys.executable, "-m", "kgkit", *args], cwd=cwd, env=env, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(r.stdout + r.stderr)
    return r


@pytest.fixture
def workspace(tmp_path, plugin_root, binary_df):
    run(["init", "demo-comp", "--metric", "auc", "--task", "tabular", "--target", "target", "--id-col", "id"],
        tmp_path, plugin_root)
    binary_df.to_csv(tmp_path / "data" / "train.csv", index=False)
    test = binary_df.drop(columns="target").iloc[:300].copy()
    test["id"] = np.arange(10_000, 10_300)
    test.to_csv(tmp_path / "data" / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "target": 0.5}).to_csv(tmp_path / "data" / "sample_submission.csv", index=False)
    return tmp_path


def test_init_creates_state_and_claude_md(workspace):
    st = S.load(workspace)
    assert st.slug == "demo-comp" and st.metric == "auc" and st.greater_is_better
    assert (workspace / "CLAUDE.md").exists()
    assert "demo-comp" in (workspace / "CLAUDE.md").read_text(encoding="utf-8")
    assert S.find_root(workspace / "src") == workspace.resolve()


def test_full_cycle(workspace, plugin_root):
    r = run(["folds", "data/train.csv", "--target", "target", "--group", "grp", "--id-col", "id",
             "--out", "data/folds.csv"], workspace, plugin_root)
    assert "GROUP LEAKAGE" not in r.stdout
    r = run(["eda", "data/train.csv", "--test", "data/test.csv", "--target", "target", "--id-col", "id",
             "--out", "reports/eda.md"], workspace, plugin_root)
    assert (workspace / "reports" / "eda.md").exists()

    train = pd.read_csv(workspace / "data" / "train.csv")
    y = train["target"].to_numpy()
    rng = np.random.default_rng(0)
    led = Ledger(workspace)
    for name in ["lgbm", "catboost"]:
        oof = 1 / (1 + np.exp(-(y + rng.normal(0, 1, len(y)))))
        test_pred = rng.random(300)
        from kgkit import metrics as M

        led.log(name, M.score("auc", y, oof), fold_scores=[0.8, 0.81], oof=oof, test_pred=test_pred, model=name)
    assert len(led.records()) == 2

    r = run(["blend", "--exp", "1", "2", "--truth", "data/train.csv:target", "--folds", "data/folds.csv:fold",
             "--sample", "data/sample_submission.csv", "--name", "blend-v1"], workspace, plugin_root)
    assert "honest" in r.stdout
    blend = led.records()[-1]
    assert blend["model"] == "blend" and blend["submission"]
    sub_path = workspace / blend["submission"]
    assert validate_submission(sub_path, workspace / "data" / "sample_submission.csv")["ok"]

    run(["ledger", "lb", "1", "0.79"], workspace, plugin_root)
    assert led.get("1")["lb_public"] == 0.79
    status = run(["status"], workspace, plugin_root).stdout
    assert "demo-comp" in status and "best CV" in status


def test_validate_catches_errors(tmp_path):
    sample = pd.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.5, 0.5]})
    bad = pd.DataFrame({"Unnamed: 0": [0, 1], "id": [1, 1], "target": [np.nan, 0.2]})
    res = validate_submission(bad, sample, prob_cols=["target"])
    assert not res["ok"]
    joined = " ".join(res["errors"])
    assert "Unnamed" in joined and "row count" in joined and "NaN" in joined and "duplicated" in joined


def test_ledger_best_respects_direction(tmp_path):
    S.CompetitionState(slug="x", metric="rmse", greater_is_better=False).save(tmp_path)
    led = Ledger(tmp_path)
    led.log("a", 1.0)
    led.log("b", 0.5)
    assert led.best(1)[0]["name"] == "b"
    rec = json.loads((tmp_path / ".kaggle-gm" / "ledger.jsonl").read_text().splitlines()[0])
    assert rec["metric"] == "rmse"


def test_blend_warns_on_mismatched_folds(workspace, plugin_root):
    train = pd.read_csv(workspace / "data" / "train.csv")
    y = train["target"].to_numpy()
    rng = np.random.default_rng(1)
    led = Ledger(workspace)
    a = led.log("a", 0.8, oof=rng.random(len(y)), test_pred=rng.random(300), folds=np.arange(len(y)) % 5)
    b = led.log("b", 0.8, oof=rng.random(len(y)), test_pred=rng.random(300), folds=(np.arange(len(y)) + 1) % 5)
    assert a["folds_hash"] != b["folds_hash"]
    r = run(["blend", "--exp", a["id"], b["id"], "--truth", "data/train.csv:target"], workspace, plugin_root)
    assert "DIFFERENT fold splits" in r.stdout
    c = led.log("c", 0.8)  # no explicit folds: fingerprint of data/folds.csv when present, else None
    assert c["folds_hash"] is None or len(c["folds_hash"]) == 12
