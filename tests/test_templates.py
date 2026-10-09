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


@pytest.mark.parametrize("model,kind", [("linear", "binary"), ("knn", "multiclass"), ("svm", "regression"),
                                        ("mlp", "binary")])
def test_tabular_template_diverse_families(tmp_path, plugin_root, model, kind):
    ws = _make_ws(tmp_path, plugin_root, kind)
    extra = ["--params", '{"max_iter": 300}'] if model == "mlp" else []
    out = _run(["src/train_gbdt.py", "--model", model, "--name", f"{model}_{kind}", *extra], ws, plugin_root)
    assert "CV" in out
    rec = Ledger(ws).records()[-1]
    assert rec["model"] == model and rec["oof_path"] and rec["test_path"]
    res = validate_submission(ws / rec["submission"], ws / "data" / "sample_submission.csv")
    assert res["ok"], res
    if kind == "binary":
        assert rec["cv"] > 0.8
    if kind == "multiclass":
        assert rec["cv"] > 0.5


@pytest.mark.skipif(importlib.util.find_spec("lightgbm") is None, reason="lightgbm not installed")
def test_gbdt_template_lgbm_smoke_and_full(tmp_path, plugin_root):
    ws = _make_ws(tmp_path, plugin_root, "binary")
    assert "smoke run OK" in _run(["src/train_gbdt.py", "--smoke"], ws, plugin_root)
    _run(["src/train_gbdt.py", "--params", '{"n_estimators": 300, "learning_rate": 0.1}', "--seeds", "1", "2"], ws, plugin_root)
    rec = Ledger(ws).records()[-1]
    assert rec["cv"] > 0.8 and len(rec["fold_scores"]) == 5


@pytest.mark.skipif(importlib.util.find_spec("timm") is None or importlib.util.find_spec("torch") is None,
                    reason="needs torch + timm")
def test_image_template_tiny(tmp_path, plugin_root):
    from PIL import Image

    rng = np.random.default_rng(0)
    colors = {"red": (220, 30, 30), "green": (30, 200, 30), "blue": (30, 30, 220)}
    (tmp_path / "data" / "train_images").mkdir(parents=True)
    (tmp_path / "data" / "test_images").mkdir(parents=True)
    rows, trows = [], []
    for i in range(60):
        lab = list(colors)[i % 3]
        arr = np.clip(np.array(colors[lab]) + rng.normal(0, 25, (32, 32, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(tmp_path / "data" / "train_images" / f"{i}.png")
        rows.append({"image_id": i, "image_path": f"{i}.png", "label": lab})
    for i in range(9):
        lab = list(colors)[i % 3]
        arr = np.clip(np.array(colors[lab]) + rng.normal(0, 25, (32, 32, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(tmp_path / "data" / "test_images" / f"t{i}.png")
        trows.append({"image_id": 1000 + i, "image_path": f"t{i}.png"})
    pd.DataFrame(rows).to_csv(tmp_path / "data" / "train.csv", index=False)
    pd.DataFrame(trows).to_csv(tmp_path / "data" / "test.csv", index=False)
    pd.DataFrame({"image_id": [1000 + i for i in range(9)], "label": "red"}).to_csv(
        tmp_path / "data" / "sample_submission.csv", index=False)
    _run(["-m", "kgkit", "init", "img", "--metric", "accuracy", "--target", "label", "--id-col", "image_id"],
         tmp_path, plugin_root)
    _run(["-m", "kgkit", "folds", "data/train.csv", "--target", "label", "--id-col", "image_id", "--n-splits", "3",
          "--out", "data/folds.csv"], tmp_path, plugin_root)
    (tmp_path / "src").mkdir(exist_ok=True)
    src = (plugin_root / "templates" / "train_image.py").read_text(encoding="utf-8")
    (tmp_path / "src" / "train_image.py").write_text(src.replace("num_workers=4", "num_workers=0"), encoding="utf-8")
    args = ["src/train_image.py", "--backbone", "resnet18", "--no-pretrained", "--img-size", "32",
            "--batch-size", "8", "--lr", "3e-3"]
    assert "smoke run OK" in _run([*args, "--smoke"], tmp_path, plugin_root)
    # split folds across two "GPUs": first call must not log, second completes the set
    out1 = _run([*args, "--epochs", "3", "--name", "r18", "--folds", "0"], tmp_path, plugin_root)
    assert "the run that completes the set" in out1
    _run([*args, "--epochs", "3", "--name", "r18", "--folds", "1", "2"], tmp_path, plugin_root)
    rec = Ledger(tmp_path).records()[-1]
    assert rec["name"] == "r18" and len(rec["fold_scores"]) == 3
    res = validate_submission(tmp_path / rec["submission"], tmp_path / "data" / "sample_submission.csv")
    assert res["ok"], res


_TINY_BERT = """
import sys, torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
path = sys.argv[1]
vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "good", "bad", "movie", "very",
                                      "plot", "the", "was", "not"])}
tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
tk.pre_tokenizer = pre_tokenizers.Whitespace()
PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                        sep_token="[SEP]").save_pretrained(path)
torch.manual_seed(0)
cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                 intermediate_size=64, max_position_embeddings=64)
BertModel(cfg).save_pretrained(path)
"""


def _tiny_local_bert(path):
    """Build a tiny BERT + word-level tokenizer on disk (no network). Runs in a subprocess: importing torch
    inside the pytest process after scipy/sklearn can hit DLL-order conflicts on Windows."""
    r = subprocess.run([sys.executable, "-c", _TINY_BERT, str(path)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None or importlib.util.find_spec("tokenizers") is None,
                    reason="needs transformers + tokenizers")
def test_transformer_template_tiny(tmp_path, plugin_root):
    rng = np.random.default_rng(0)
    words = ["good", "bad", "movie", "very", "plot", "the", "was", "not"]
    rows = []
    for i in range(120):
        toks = list(rng.choice(words, size=8))
        rows.append({"id": i, "text": " ".join(toks), "score": toks.count("good") - toks.count("bad")})
    df = pd.DataFrame(rows)
    (tmp_path / "data").mkdir()
    df.iloc[:100].to_csv(tmp_path / "data" / "train.csv", index=False)
    df.iloc[100:].drop(columns="score").to_csv(tmp_path / "data" / "test.csv", index=False)
    pd.DataFrame({"id": df["id"].iloc[100:], "score": 0.0}).to_csv(tmp_path / "data" / "sample_submission.csv", index=False)
    _tiny_local_bert(tmp_path / "tinybert")
    _run(["-m", "kgkit", "init", "nlp", "--metric", "rmse", "--target", "score", "--id-col", "id"], tmp_path, plugin_root)
    _run(["-m", "kgkit", "folds", "data/train.csv", "--target", "score", "--id-col", "id", "--n-splits", "2",
          "--out", "data/folds.csv"], tmp_path, plugin_root)
    (tmp_path / "src").mkdir(exist_ok=True)
    src = (plugin_root / "templates" / "train_transformer.py").read_text(encoding="utf-8")
    (tmp_path / "src" / "train_transformer.py").write_text(src.replace("num_workers=2", "num_workers=0"), encoding="utf-8")
    args = ["src/train_transformer.py", "--model", "tinybert", "--max-len", "16", "--batch-size", "8", "--lr", "1e-3"]
    assert "smoke run OK" in _run([*args, "--smoke"], tmp_path, plugin_root)
    _run([*args, "--epochs", "10", "--name", "tiny"], tmp_path, plugin_root)
    rec = Ledger(tmp_path).records()[-1]
    assert rec["name"] == "tiny" and len(rec["fold_scores"]) == 2
    assert rec["cv"] < df["score"].std()  # learned something beyond predicting the mean
    res = validate_submission(tmp_path / rec["submission"], tmp_path / "data" / "sample_submission.csv")
    assert res["ok"], res


def test_inference_kernel_dry_run(tmp_path, plugin_root):
    comp = tmp_path / "input" / "COMPETITION-SLUG"
    comp.mkdir(parents=True)
    (tmp_path / "working").mkdir()
    pd.DataFrame({"id": [5, 6, 7], "x": [1, 2, 3]}).to_csv(comp / "test.csv", index=False)
    pd.DataFrame({"id": [5, 6, 7], "target": 0.5}).to_csv(comp / "sample_submission.csv", index=False)
    env = dict(os.environ, KAGGLE_INPUT=str(tmp_path / "input"), KAGGLE_WORKING=str(tmp_path / "working"))
    r = subprocess.run([sys.executable, str(plugin_root / "templates" / "inference_kernel.py")], env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    sub = pd.read_csv(tmp_path / "working" / "submission.csv")
    assert list(sub.columns) == ["id", "target"] and len(sub) == 3
