"""The training templates honour a screen (#134): under KG_SCREEN they train and score only the screen's rows and
folds, log the OOF with NaN everywhere else, write no submission (a screen is never submitted), and keep their fold
files apart from a full run of the same name. A full run is unchanged."""

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from kgkit import screen as SC
from kgkit.experiment import Ledger
from test_templates import _make_ws, _tiny_local_bert

SCREEN_VARS = ("KG_SCREEN", "KG_FIDELITY", "KG_CONTROL_OF", "KG_SWEEP", "KG_SWEEP_VARIANT", "KG_RUNG")


def _run(args, cwd, plugin_root, **env):
    full = {k: v for k, v in os.environ.items() if k not in SCREEN_VARS}
    full.update(KGKIT_HOME=str(plugin_root), PYTHONPATH=os.pathsep.join([str(plugin_root),
                                                                         os.environ.get("PYTHONPATH", "")]), **env)
    r = subprocess.run([sys.executable, *args], cwd=cwd, env=full, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def _expected_rows(ws, id_col, screen, folds):
    train = pd.read_csv(ws / "data" / "train.csv")
    fold = train[[id_col]].merge(pd.read_csv(ws / "data" / "folds.csv")[[id_col, "fold"]], on=id_col)["fold"]
    return SC.load(ws, screen).mask(train[id_col]) & np.isin(fold.to_numpy(), folds)


def test_gbdt_template_trains_and_scores_only_the_screen(tmp_path, plugin_root):
    ws = _make_ws(tmp_path, plugin_root, "binary")
    _run(["-m", "kgkit", "screen", "new", "half", "--frac", "0.5", "--folds", "0", "1"], ws, plugin_root)
    gbdt = ["src/train_gbdt.py", "--model", "hgb", "--params", '{"max_iter": 60}']
    _run([*gbdt, "--name", "base"], ws, plugin_root)
    led = Ledger(ws)
    base = led.records()[-1]
    led.decide(base["id"], "baseline")
    assert "fidelity" not in base and np.isfinite(np.load(ws / base["oof_path"])).all() and base["submission"]

    out = _run([*gbdt, "--name", "ctl", "--parent", base["id"]], ws, plugin_root, KG_SCREEN="half",
               KG_CONTROL_OF=base["id"])
    ctl = Ledger(ws).records()[-1]
    assert ctl["control_of"] == base["id"] and "control of" in out

    out = _run([*gbdt, "--name", "v1", "--parent", base["id"], "--params", '{"max_iter": 150}'], ws, plugin_root,
               KG_SCREEN="half")
    v1 = Ledger(ws).records()[-1]
    assert v1["fidelity"]["screen"] == "half" and len(v1["fold_scores"]) == 2
    oof = np.load(ws / v1["oof_path"])
    assert (np.isfinite(oof) == _expected_rows(ws, "row_id", "half", [0, 1])).all()
    assert not v1.get("submission") and not v1.get("test_path") and "screen half" in out and "no submission" in out
    assert f"{v1['id']} vs {ctl['id']}" in out  # compared with the control when it logs


def _image_ws(tmp_path, plugin_root):
    from PIL import Image

    rng = np.random.default_rng(0)
    colors = {"red": (220, 30, 30), "green": (30, 200, 30), "blue": (30, 30, 220)}
    for d in ("train_images", "test_images"):
        (tmp_path / "data" / d).mkdir(parents=True)
    rows, trows = [], []
    for i in range(48):
        lab = list(colors)[i % 3]
        arr = np.clip(np.array(colors[lab]) + rng.normal(0, 25, (32, 32, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(tmp_path / "data" / "train_images" / f"{i}.png")
        rows.append({"image_id": i, "image_path": f"{i}.png", "label": lab})
    for i in range(6):
        Image.fromarray(np.full((32, 32, 3), 128, np.uint8)).save(tmp_path / "data" / "test_images" / f"t{i}.png")
        trows.append({"image_id": 1000 + i, "image_path": f"t{i}.png"})
    pd.DataFrame(rows).to_csv(tmp_path / "data" / "train.csv", index=False)
    pd.DataFrame(trows).to_csv(tmp_path / "data" / "test.csv", index=False)
    pd.DataFrame({"image_id": [1000 + i for i in range(6)], "label": "red"}).to_csv(
        tmp_path / "data" / "sample_submission.csv", index=False)
    _run(["-m", "kgkit", "init", "img", "--metric", "accuracy", "--target", "label", "--id-col", "image_id"],
         tmp_path, plugin_root)
    _run(["-m", "kgkit", "folds", "data/train.csv", "--target", "label", "--id-col", "image_id", "--n-splits", "3",
          "--out", "data/folds.csv"], tmp_path, plugin_root)
    (tmp_path / "src").mkdir(exist_ok=True)
    src = (plugin_root / "templates" / "train_image.py").read_text(encoding="utf-8")
    (tmp_path / "src" / "train_image.py").write_text(src.replace("num_workers=4", "num_workers=0"), encoding="utf-8")
    return tmp_path


@pytest.mark.skipif(importlib.util.find_spec("timm") is None, reason="needs torch + timm")
def test_image_template_on_a_screen_keeps_its_folds_apart(tmp_path, plugin_root):
    ws = _image_ws(tmp_path, plugin_root)
    _run(["-m", "kgkit", "screen", "new", "f0", "--folds", "0", "--frac", "0.5"], ws, plugin_root)
    args = ["src/train_image.py", "--backbone", "resnet18", "--no-pretrained", "--img-size", "32", "--batch-size", "8",
            "--lr", "3e-3", "--epochs", "1", "--name", "r18"]
    _run(args, ws, plugin_root)
    full = Ledger(ws).records()[-1]
    assert len(full["fold_scores"]) == 3 and "fidelity" not in full and full["submission"]
    out = _run(args, ws, plugin_root, KG_SCREEN="f0", KG_FIDELITY='{"epochs": 1}')  # the same name
    rec = Ledger(ws).records()[-1]
    assert rec["id"] != full["id"] and rec["fidelity"] == {**SC.load(ws, "f0").fidelity(), "epochs": 1}
    assert len(rec["fold_scores"]) == 1 and not rec.get("submission") and "no submission" in out
    oof = np.load(ws / rec["oof_path"])
    assert (np.isfinite(oof).all(1) == _expected_rows(ws, "image_id", "f0", [0])).all()
    dirs = sorted(p.name for p in (ws / "artifacts").iterdir() if p.name.startswith("r18"))
    assert len(dirs) == 2 and "r18" in dirs  # the screen's fold files are not the full run's


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None or importlib.util.find_spec("tokenizers") is None,
                    reason="needs transformers + tokenizers")
def test_transformer_template_on_a_screen(tmp_path, plugin_root):
    rng = np.random.default_rng(0)
    words = ["good", "bad", "movie", "very", "plot", "the", "was", "not"]
    rows = []
    for i in range(80):
        toks = list(rng.choice(words, size=8))
        rows.append({"id": i, "text": " ".join(toks), "score": toks.count("good") - toks.count("bad")})
    df = pd.DataFrame(rows)
    (tmp_path / "data").mkdir()
    df.iloc[:60].to_csv(tmp_path / "data" / "train.csv", index=False)
    df.iloc[60:].drop(columns="score").to_csv(tmp_path / "data" / "test.csv", index=False)
    pd.DataFrame({"id": df["id"].iloc[60:], "score": 0.0}).to_csv(tmp_path / "data" / "sample_submission.csv",
                                                                   index=False)
    _tiny_local_bert(tmp_path / "tinybert")
    _run(["-m", "kgkit", "init", "nlp", "--metric", "rmse", "--target", "score", "--id-col", "id"], tmp_path,
         plugin_root)
    _run(["-m", "kgkit", "folds", "data/train.csv", "--target", "score", "--id-col", "id", "--n-splits", "3",
          "--out", "data/folds.csv"], tmp_path, plugin_root)
    (tmp_path / "src").mkdir(exist_ok=True)
    src = (plugin_root / "templates" / "train_transformer.py").read_text(encoding="utf-8")
    (tmp_path / "src" / "train_transformer.py").write_text(src.replace("num_workers=2", "num_workers=0"),
                                                          encoding="utf-8")
    _run(["-m", "kgkit", "screen", "new", "f1", "--folds", "1", "--frac", "0.6"], tmp_path, plugin_root)
    out = _run(["src/train_transformer.py", "--model", "tinybert", "--max-len", "16", "--batch-size", "8", "--lr",
                "1e-3", "--epochs", "1", "--name", "tiny"], tmp_path, plugin_root, KG_SCREEN="f1")
    rec = Ledger(tmp_path).records()[-1]
    assert rec["fidelity"]["screen"] == "f1" and len(rec["fold_scores"]) == 1
    assert not rec.get("submission") and "no submission" in out
    oof = np.load(tmp_path / rec["oof_path"])
    assert (np.isfinite(oof.reshape(len(oof), -1)).all(1) == _expected_rows(tmp_path, "id", "f1", [1])).all()


def test_a_sweep_over_the_gbdt_template(tmp_path, plugin_root):
    """The sweep runner and a template together: the screens train on the screen, the finalist is a full run with a
    submission, compared with the baseline."""
    import json

    ws = _make_ws(tmp_path, plugin_root, "binary")
    _run(["-m", "kgkit", "screen", "new", "half", "--frac", "0.5", "--folds", "0", "1"], ws, plugin_root)
    _run(["src/train_gbdt.py", "--model", "hgb", "--name", "base"], ws, plugin_root)
    led = Ledger(ws)
    base = led.records()[-1]
    led.decide(base["id"], "baseline")
    cmd = f'"{sys.executable}" src/train_gbdt.py --parent {{base}} --name {{sweep}}_{{variant}} {{args}}'
    _run(["-m", "kgkit", "sweep", "new", "fam", "--cmd", cmd, "--base-args", "--model hgb",
          "--variant", "mlp=--model mlp", "--variant", "hgb7=--model hgb --seeds 7", "--rung", "screen=half",
          "--eta", "2"], ws, plugin_root)
    out = _run(["-m", "kgkit", "sweep", "run", "fam"], ws, plugin_root)
    st = json.loads((ws / ".kaggle-gm" / "sweeps" / "fam.json").read_text())
    assert st["state"] == "done" and len(st["survivors"]["1"]) == 1
    recs = {r["id"]: r for r in Ledger(ws).records()}
    ctl = recs[st["controls"]["0"]]
    assert ctl["control_of"] == base["id"] and ctl["model"] == "hgb" and ctl["fidelity"]["screen"] == "half"
    for v in ("mlp", "hgb7"):
        r = recs[st["runs"][v]["0"]["record"]]
        assert r["fidelity"]["screen"] == "half" and len(r["fold_scores"]) == 2 and not r.get("submission")
    final = recs[st["runs"][st["survivors"]["1"][0]]["1"]["record"]]
    assert "fidelity" not in final and len(final["fold_scores"]) == 5 and final["submission"]
    assert f"{final['id']} vs {base['id']}" in out and "best of 2" in out
