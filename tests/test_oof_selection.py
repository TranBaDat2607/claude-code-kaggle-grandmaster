"""#109: an OOF prediction is honest only when nothing about the model (stopping round, epoch, checkpoint) was chosen
on the rows it predicts. The training templates early-stop / pick epochs off the scored fold, record how the OOF was
selected (``oof_selection``), and ``ledger compare`` says when two records were selected differently."""

import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kgkit import state as S
from kgkit.experiment import Ledger

SKILLS = Path(__file__).resolve().parents[1] / "plugins" / "kaggle-grandmaster" / "skills"


# ----------------------------------------------------------------------------- GBDT template, in process
def _tabular_ws(root: Path, plugin_root: Path, n=500, n_folds=5) -> Path:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"row_id": np.arange(n), "a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["y"] = (df["a"] + rng.normal(0, 1, n) > 0.6).astype(int)  # imbalanced: ~30% positives
    (root / "data").mkdir()
    df.iloc[:400].to_csv(root / "data" / "train.csv", index=False)
    df.iloc[400:].drop(columns="y").to_csv(root / "data" / "test.csv", index=False)
    pd.DataFrame({"row_id": df["row_id"].iloc[400:], "y": 0.0}).to_csv(root / "data" / "sample_submission.csv",
                                                                      index=False)
    pd.DataFrame({"row_id": np.arange(400), "fold": np.arange(400) % n_folds}).to_csv(root / "data" / "folds.csv",
                                                                                    index=False)
    S.CompetitionState(slug="t", metric="auc", target="y", id_col="row_id").save(root)
    (root / "src").mkdir()
    shutil.copy(plugin_root / "templates" / "train_gbdt.py", root / "src" / "train_gbdt.py")
    return root


def _load_template(path: Path):
    spec = importlib.util.spec_from_file_location(f"train_gbdt_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Fake:
    """Stands in for a GBDT: predicts from column a, remembers the params it was made with."""

    def __init__(self, params):
        self.params = params

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-X["a"].to_numpy()))
        return np.c_[1 - p, p]


@pytest.fixture
def gbdt(tmp_path, plugin_root, monkeypatch):
    ws = _tabular_ws(tmp_path, plugin_root)
    monkeypatch.chdir(ws)
    mod = _load_template(ws / "src" / "train_gbdt.py")
    calls = {"fit": [], "make": []}

    def make_model(kind, task, params, seed, n_classes):
        calls["make"].append(dict(params))
        return _Fake(params)

    def fit(model, kind, X_fit, y_fit, X_es, y_es, cats, es):
        calls["fit"].append((set(X_fit.index), None if X_es is None else set(X_es.index),
                             None if y_es is None else np.asarray(y_es)))
        return model, 100

    monkeypatch.setattr(mod, "make_model", make_model)
    monkeypatch.setattr(mod, "fit", fit)

    def run(*argv):
        calls["fit"].clear()
        calls["make"].clear()
        monkeypatch.setattr(sys, "argv", ["train_gbdt.py", *argv])
        mod.main()
        return Ledger(ws).records()[-1]

    folds = pd.read_csv(ws / "data" / "folds.csv")["fold"].to_numpy()
    y = pd.read_csv(ws / "data" / "train.csv")["y"].to_numpy()
    return run, calls, folds, y


def test_gbdt_early_stops_on_an_inner_slice_never_the_scored_fold(gbdt, capsys):
    run, calls, folds, y = gbdt
    rec = run("--model", "lgbm", "--name", "inner")
    assert len(calls["fit"]) == 5
    for f, (fit_rows, es_rows, y_es) in enumerate(calls["fit"]):
        va = set(np.flatnonzero(folds == f))
        tr = set(np.flatnonzero(folds != f))
        assert es_rows and not es_rows & va and not fit_rows & va  # the scored fold chose nothing
        assert fit_rows | es_rows == tr and not fit_rows & es_rows
        assert len(es_rows) == pytest.approx(0.1 * len(tr), abs=2)
        assert y_es.mean() == pytest.approx(y[sorted(tr)].mean(), abs=0.03)  # stratified
    assert rec["oof_selection"] == "independent" and rec["params"]["early_stopping_on"] == "inner"
    assert len(rec["train_scores"]) == 5 and all(0.5 < s <= 1 for s in rec["train_scores"])
    assert re.search(r"train [0-9.]+", capsys.readouterr().out)

    # the slice is drawn with the run seed: the same seed draws it again, another seed another one
    first = [c[1] for c in calls["fit"]]
    run("--model", "lgbm", "--name", "again")
    assert [c[1] for c in calls["fit"]] == first
    run("--model", "lgbm", "--name", "seed7", "--seeds", "7")
    assert [c[1] for c in calls["fit"]] != first


def test_gbdt_es_on_valid_keeps_the_old_behaviour_and_says_so(gbdt):
    run, calls, folds, _ = gbdt
    rec = run("--model", "lgbm", "--name", "old", "--es-on-valid")
    for f, (fit_rows, es_rows, _) in enumerate(calls["fit"]):
        assert es_rows == set(np.flatnonzero(folds == f)) and fit_rows == set(np.flatnonzero(folds != f))
    assert rec["oof_selection"] == "scored-fold" and rec["params"]["early_stopping_on"] == "scored-fold"


def test_gbdt_refit_trains_on_the_whole_training_part(gbdt):
    run, calls, folds, _ = gbdt
    rec = run("--model", "lgbm", "--name", "refit", "--refit")
    assert len(calls["fit"]) == 10  # early-stopping fit, then a refit, per fold
    for f in range(5):
        _, es_rows, _ = calls["fit"][2 * f]
        refit_rows, refit_es, _ = calls["fit"][2 * f + 1]
        assert es_rows and refit_es is None and refit_rows == set(np.flatnonzero(folds != f))
    assert calls["make"][1]["n_estimators"] == math.ceil(100 / 0.9)
    assert rec["oof_selection"] == "independent" and rec["params"]["refit"] is True


def test_gbdt_parent_flag_and_models_without_early_stopping(gbdt):
    run, calls, _, _ = gbdt
    base = run("--model", "linear", "--name", "lin")
    assert all(es is None for _, es, _ in calls["fit"])  # nothing to stop: no rows held back
    assert base["oof_selection"] == "independent" and base["parent"] is None
    rec = run("--model", "lgbm", "--name", "child", "--parent", base["id"])
    assert rec["parent"] == base["id"]


# ----------------------------------------------------------------------------- GBDT template, real libraries
@pytest.mark.parametrize("model,n_key", [("lgbm", "n_estimators"), ("xgb", "n_estimators"), ("cat", "iterations")])
def test_gbdt_template_real_library_inner_early_stopping(tmp_path, plugin_root, model, n_key):
    lib = {"lgbm": "lightgbm", "xgb": "xgboost", "cat": "catboost"}[model]
    if importlib.util.find_spec(lib) is None:
        pytest.skip(f"{lib} not installed")
    ws = _tabular_ws(tmp_path, plugin_root)
    env = {"KGKIT_HOME": str(plugin_root), "PYTHONPATH": str(plugin_root)}
    params = json.dumps({n_key: 300, "learning_rate": 0.1})
    for extra in ([], ["--refit"]):
        r = subprocess.run([sys.executable, "src/train_gbdt.py", "--model", model, "--params", params, *extra],
                           cwd=ws, capture_output=True, text=True, timeout=600, env={**__import__("os").environ, **env})
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        rec = Ledger(ws).records()[-1]
        assert rec["oof_selection"] == "independent" and len(rec["train_scores"]) == 5 and rec["cv"] > 0.6
        assert all(t >= v - 0.05 for t, v in zip(rec["train_scores"], rec["fold_scores"]))


# ----------------------------------------------------------------------------- ledger compare
def test_ledger_compare_says_when_the_two_oofs_were_selected_differently(tmp_path, monkeypatch, capsys):
    from kgkit import cli
    from kgkit import experiment as E

    rng = np.random.default_rng(0)
    S.CompetitionState(slug="t", metric="auc", target="target", id_col="id").save(tmp_path)
    (tmp_path / "data").mkdir()
    y = (rng.random(200) < 0.5).astype(int)
    pd.DataFrame({"id": np.arange(200), "target": y}).to_csv(tmp_path / "data" / "train.csv", index=False)
    pd.DataFrame({"id": np.arange(200), "fold": np.arange(200) % 5}).to_csv(tmp_path / "data" / "folds.csv",
                                                                          index=False)
    monkeypatch.chdir(tmp_path)
    led = Ledger(tmp_path)
    kw = dict(fold_scores=[0.7] * 5)
    old = led.log("old", 0.70, oof=rng.random(200), **kw)  # logged before kgkit recorded the selection
    biased = led.log("biased", 0.70, oof=rng.random(200), oof_selection="scored-fold", **kw)
    honest = led.log("honest", 0.70, oof=rng.random(200), oof_selection="independent", **kw)
    honest2 = led.log("honest2", 0.70, oof=rng.random(200), oof_selection="independent", **kw)

    assert E.selection_note(honest2, honest) is None
    assert "optimistic" in E.selection_note(honest, biased) and biased["id"] in E.selection_note(honest, biased)
    assert "re-run" in E.selection_note(honest, old) and old["id"] in E.selection_note(honest, old)
    assert E.selection_note(old, old) is None  # neither says: nothing to compare

    cli.main(["ledger", "compare", honest["id"], biased["id"]])
    out = capsys.readouterr().out
    assert "WARNING" in out and "scored fold" in out
    cli.main(["ledger", "compare", honest2["id"], honest["id"]])
    assert "scored fold" not in capsys.readouterr().out


# ----------------------------------------------------------------------------- skills
SCORED_FOLD_SELECTION = re.compile(
    r"best[- ]by[- ]metric|best checkpoints? (?:often|per fold)|(?:checkpoint|evaluate|keep) the best checkpoint"
    r"|checkpoint the best|save best"
    r"|eval_set=\[\(X(?:\.iloc|\.loc)?\[(?:va|val|valid)\w*\]", re.IGNORECASE)


def test_no_skill_teaches_selecting_on_the_scored_fold():
    """#128: skills that told Claude to keep the best checkpoint, or early-stop on the scored fold, contradicted the
    templates and kg-experiment (#109)."""
    for p in sorted(SKILLS.glob("**/*.md")):
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            m = SCORED_FOLD_SELECTION.search(line)
            assert not m, f"{p.relative_to(SKILLS)}:{n} selects on the scored fold: {line.strip()}"


def test_no_skill_repeats_a_paragraph():
    """A paragraph or bullet pasted twice into a skill costs tokens on every load and reads as a mistake."""
    for p in sorted(SKILLS.glob("*/**/*.md")):
        blocks = [b.strip() for b in re.split(r"\n\s*\n|\n(?=\s*[-*] |\s*\d+\. )", p.read_text(encoding="utf-8"))]
        long = [b for b in blocks if len(b) >= 120 and not b.startswith(("|", "```"))]
        dup = {b[:80] for b in long if long.count(b) > 1}
        assert not dup, f"{p.relative_to(SKILLS)} repeats: {sorted(dup)}"
