"""Run the training templates end to end in a throwaway competition workspace."""

import importlib.util
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from kgkit.experiment import Ledger
from kgkit.submission import validate_submission


def _run(args, cwd, plugin_root):
    env = dict(os.environ, KGKIT_HOME=str(plugin_root),
               PYTHONPATH=os.pathsep.join([str(plugin_root), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def _make_ws(tmp_path, plugin_root, kind):
    rng = np.random.default_rng(0)
    n = 1200
    df = pd.DataFrame({"row_id": np.arange(n), "a": rng.normal(size=n), "b": rng.normal(size=n),
                       "c": rng.choice(list("xyz"), n)})
    signal = df["a"] * 2 + (df["c"] == "x") - df["b"]
    if kind == "binary":
        df["y"] = (signal + rng.normal(0, 1, n) > 0).astype(int)
        metric = "auc"
    elif kind == "multiclass":
        df["y"] = pd.cut(signal + rng.normal(0, 0.5, n), 3, labels=["lo", "mid", "hi"]).astype(str)
        metric = "accuracy"
    else:
        df["y"] = np.exp(signal / 2 + rng.normal(0, 0.3, n))
        metric = "rmsle"
    train, test = df.iloc[:1000].copy(), df.iloc[1000:].drop(columns="y")
    (tmp_path / "data").mkdir()
    train.to_csv(tmp_path / "data" / "train.csv", index=False)
    test.to_csv(tmp_path / "data" / "test.csv", index=False)
    sample = pd.DataFrame({"row_id": test["row_id"][::-1].to_numpy(),  # deliberately different order
                           "y": "lo" if kind == "multiclass" else 0.0})
    sample.to_csv(tmp_path / "data" / "sample_submission.csv", index=False)
    _run(["-m", "kgkit", "init", "t", "--metric", metric, "--target", "y", "--id-col", "row_id"], tmp_path, plugin_root)
    _run(["-m", "kgkit", "folds", "data/train.csv", "--target", "y", "--id-col", "row_id", "--out", "data/folds.csv"],
         tmp_path, plugin_root)
    (tmp_path / "src").mkdir(exist_ok=True)
    shutil.copy(plugin_root / "templates" / "train_gbdt.py", tmp_path / "src" / "train_gbdt.py")
    return tmp_path


@pytest.mark.parametrize("kind", ["binary", "multiclass", "regression"])
def test_gbdt_template_hgb(tmp_path, plugin_root, kind):
    ws = _make_ws(tmp_path, plugin_root, kind)
    extra = ["--params", '{"max_iter": 100}']
    script = "src/train_gbdt.py"
    if kind == "regression":
        # enable log-target training through a tiny CONFIG edit, as a user would
        p = ws / script
        p.write_text(p.read_text(encoding="utf-8").replace("log_target=False", "log_target=True"), encoding="utf-8")
    out = _run([script, "--model", "hgb", "--name", f"hgb_{kind}", *extra], ws, plugin_root)
    assert "CV" in out
    rec = Ledger(ws).records()[-1]
    assert rec["model"] == "hgb" and rec["submission"]
    res = validate_submission(ws / rec["submission"], ws / "data" / "sample_submission.csv")
    assert res["ok"], res
    sub = pd.read_csv(ws / rec["submission"])
    if kind == "binary":
        assert rec["cv"] > 0.8 and sub["y"].between(0, 1).all()
    if kind == "multiclass":
        assert set(sub["y"]) <= {"lo", "mid", "hi"} and rec["cv"] > 0.5
    if kind == "regression":
        assert (sub["y"] > 0).all() and rec["cv"] < 0.6


@pytest.mark.skipif(importlib.util.find_spec("lightgbm") is None, reason="lightgbm not installed")
def test_gbdt_template_lgbm_smoke_and_full(tmp_path, plugin_root):
    ws = _make_ws(tmp_path, plugin_root, "binary")
    assert "smoke run OK" in _run(["src/train_gbdt.py", "--smoke"], ws, plugin_root)
    _run(["src/train_gbdt.py", "--params", '{"n_estimators": 300, "learning_rate": 0.1}', "--seeds", "1", "2"], ws, plugin_root)
    rec = Ledger(ws).records()[-1]
    assert rec["cv"] > 0.8 and len(rec["fold_scores"]) == 5
