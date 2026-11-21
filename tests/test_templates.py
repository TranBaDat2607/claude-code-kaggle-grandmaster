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


@pytest.mark.parametrize("kind", ["binary", "multiclass"])
def test_gbdt_template_compares_the_run_with_its_parent_when_it_logs(tmp_path, plugin_root, kind):
    """#127: the training output ends with the paired verdict against --parent (else the accepted baseline), so the
    experiment needs no separate `ledger compare` call."""
    ws = _make_ws(tmp_path, plugin_root, kind)
    run = ["src/train_gbdt.py", "--model", "hgb"]
    out = _run([*run, "--name", "base", "--params", '{"max_iter": 30}'], ws, plugin_root)
    assert "no accepted baseline yet" in out and "kgkit exp finish 0001_base baseline" in out
    _run(["-m", "kgkit", "exp", "finish", "0001", "baseline", "first model"], ws, plugin_root)
    out = _run([*run, "--name", "more", "--params", '{"max_iter": 120}', "--parent", "0001"], ws, plugin_root)
    tail = out[out.index("logged 0002_more"):]
    assert "0002_more vs 0001_base:" in tail and "folds:" in tail
    # accuracy scores decoded labels while the ledger keeps raw probabilities: per-fold scores, no row-level test
    assert ("rows)" in tail) == (kind == "binary")
    assert "kgkit exp finish 0002_more" in tail and out.isascii()
    comps = (ws / ".kaggle-gm" / "comparisons.jsonl").read_text(encoding="utf-8")
    assert '"new": "0002_more"' in comps and '"base": "0001_base"' in comps
    # no --parent: compared with the accepted baseline
    out = _run([*run, "--name", "less", "--params", '{"max_iter": 10}'], ws, plugin_root)
    assert "0003_less vs 0001_base:" in out


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


@pytest.mark.skipif(importlib.util.find_spec("lightgbm") is None, reason="lightgbm not installed")
def test_gbdt_template_lgbm_device_cuda(tmp_path, plugin_root):
    """--device cuda with LightGBM: its GPU build when one trains here, else the CPU with a note (never a crash)."""
    ws = _make_ws(tmp_path, plugin_root, "binary")
    out = _run(["src/train_gbdt.py", "--smoke", "--device", "cuda"], ws, plugin_root)
    assert "smoke run OK" in out and ("note: LightGBM trains on the GPU" in out or "trains on the CPU" in out)


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
    # --bench: data loader alone vs the training step alone vs the real loop, recorded; logs no experiment
    n_logged = len(Ledger(tmp_path).records())
    out = _run([*args, "--bench", "3"], tmp_path, plugin_root)
    assert "[bench]" in out and "real loop" in out and "verdict:" in out
    out = _run([*args, "--bench", "3", "--batch-size", "4"], tmp_path, plugin_root)
    assert "vs previous bench of this name" in out and "batch_size 8 -> 4" in out
    assert len(Ledger(tmp_path).records()) == n_logged and (tmp_path / "reports" / "bench.jsonl").is_file()
    # split folds across two "GPUs": first call must not log, second completes the set
    out1 = _run([*args, "--epochs", "3", "--name", "r18", "--folds", "0"], tmp_path, plugin_root)
    assert "the run that completes the set" in out1
    _run([*args, "--epochs", "3", "--name", "r18", "--folds", "1", "2"], tmp_path, plugin_root)
    rec = Ledger(tmp_path).records()[-1]
    assert rec["name"] == "r18" and len(rec["fold_scores"]) == 3
    # #109: the OOF is the last epoch's (EMA) weights, not the epoch that scored best on the fold it predicts
    assert rec["oof_selection"] == "independent" and rec["params"]["select"] == "last"
    out = _run([*args, "--epochs", "2", "--name", "r18_best", "--select", "best", "--parent", rec["id"]], tmp_path,
               plugin_root)
    best = Ledger(tmp_path).records()[-1]
    assert best["oof_selection"] == "scored-fold" and "optimistic" in out
    assert best["parent"] == rec["id"] and f"{best['id']} vs {rec['id']}:" in out  # #127: compared when logged
    # #119: early stopping done legally - the epoch is chosen on rows held out of the training part
    out = _run([*args, "--epochs", "3", "--name", "r18_es", "--es-holdout", "0.2", "--es-patience", "1"], tmp_path,
               plugin_root)
    es = Ledger(tmp_path).records()[-1]
    assert "select holdout" in out and "holdout accuracy" in out
    assert es["oof_selection"] == "independent" and es["params"]["select"] == "holdout"
    assert es["params"]["es_holdout"] == 0.2 and rec["params"]["es_holdout"] == 0.0
    bad = subprocess.run([sys.executable, *args, "--smoke", "--select", "best", "--es-holdout", "0.1"], cwd=tmp_path,
                         env=dict(os.environ, PYTHONPATH=str(plugin_root)), capture_output=True, text=True, timeout=600)
    assert bad.returncode != 0 and "does not combine with --select best" in bad.stdout + bad.stderr
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
    out = _run([*args, "--bench", "3"], tmp_path, plugin_root)
    assert "[bench]" in out and "verdict:" in out and "padding:" in out
    out = _run([*args, "--bench", "3", "--no-bucket"], tmp_path, plugin_root)
    assert "length_buckets True -> False" in out
    out = _run([*args, "--epochs", "10", "--name", "tiny"], tmp_path, plugin_root)
    rec = Ledger(tmp_path).records()[-1]
    assert rec["name"] == "tiny" and len(rec["fold_scores"]) == 2
    assert rec["oof_selection"] == "independent" and rec["params"]["select"] == "last"  # #109
    # #119: the learning curve holds each epoch's last evaluation (what select=last keeps), not the running best, so
    # pruning can see a run that peaks and then declines
    import json
    import re
    last_eval = {int(e): float(s) for e, st, n, s in
                 re.findall(r"fold 0 epoch (\d+) step (\d+)/(\d+) val \w+ ([\d.]+)", out) if st == n}
    curve = json.loads((tmp_path / "artifacts" / "tiny" / "curve_fold0.json").read_text(encoding="utf-8"))
    assert len(curve["points"]) == 10
    assert all(pt["score"] == pytest.approx(last_eval[pt["epoch"]], abs=1e-5) for pt in curve["points"])
    out = _run([*args, "--epochs", "4", "--name", "tiny_es", "--es-holdout", "0.2"], tmp_path, plugin_root)
    es = Ledger(tmp_path).records()[-1]
    assert "select holdout" in out and "holdout rmse" in out
    assert es["oof_selection"] == "independent" and es["params"]["select"] == "holdout"
    out = _run([*args, "--epochs", "2", "--name", "tiny_best", "--select", "best", "--parent", rec["id"]], tmp_path,
               plugin_root)
    best = Ledger(tmp_path).records()[-1]
    assert best["oof_selection"] == "scored-fold" and "optimistic" in out
    assert best["parent"] == rec["id"] and f"{best['id']} vs {rec['id']}:" in out  # #127: compared when logged
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


def _tiny_image_ws(tmp_path, plugin_root):
    from PIL import Image

    rng = np.random.default_rng(0)
    colors = {"red": (220, 30, 30), "green": (30, 200, 30), "blue": (30, 30, 220)}
    (tmp_path / "data" / "train_images").mkdir(parents=True)
    rows = []
    for i in range(60):
        lab = list(colors)[i % 3]
        arr = np.clip(np.array(colors[lab]) + rng.normal(0, 25, (32, 32, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(tmp_path / "data" / "train_images" / f"{i}.png")
        rows.append({"image_id": i, "image_path": f"{i}.png", "label": lab})
    pd.DataFrame(rows).to_csv(tmp_path / "data" / "train.csv", index=False)
    _run(["-m", "kgkit", "init", "img", "--metric", "accuracy", "--target", "label", "--id-col", "image_id"],
         tmp_path, plugin_root)
    _run(["-m", "kgkit", "folds", "data/train.csv", "--target", "label", "--id-col", "image_id", "--n-splits", "3",
          "--out", "data/folds.csv"], tmp_path, plugin_root)
    (tmp_path / "src").mkdir(exist_ok=True)
    src = (plugin_root / "templates" / "train_image.py").read_text(encoding="utf-8")
    (tmp_path / "src" / "train_image.py").write_text(src.replace("num_workers=4", "num_workers=0"), encoding="utf-8")
    return ["src/train_image.py", "--backbone", "resnet18", "--no-pretrained", "--img-size", "32", "--batch-size", "8",
            "--lr", "3e-3"]


def test_image_template_heartbeat_and_checkpoint_every(tmp_path, plugin_root):
    """#125: a local run says which process runs it (run_<pid>.json, a heartbeat each epoch), so a new session can
    tell a live run from a dead one; --checkpoint-every N lets a killed fold resume mid-schedule."""
    import json
    import time

    args = _tiny_image_ws(tmp_path, plugin_root)
    _run([*args, "--epochs", "2", "--name", "hb", "--folds", "0"], tmp_path, plugin_root)
    (run,) = list((tmp_path / "artifacts" / "hb").glob("run_*.json"))
    info = json.loads(run.read_text())
    assert info["finished"] is True and info["folds"] == [0] and info["all_folds"] == [0, 1, 2]
    assert info["heartbeat"] >= info["started"] and "--name hb" in info["cmd"]
    assert not (tmp_path / "artifacts" / "hb" / "fold0_resume.pt").exists()  # off by default: no checkpoint
    # --checkpoint-every 1: kill the run once a checkpoint exists, run it again: it resumes, not restarts
    env = dict(os.environ, KGKIT_HOME=str(plugin_root),
               PYTHONPATH=os.pathsep.join([str(plugin_root), os.environ.get("PYTHONPATH", "")]))
    cmd = [sys.executable, *args, "--epochs", "40", "--name", "ck", "--folds", "0", "--checkpoint-every", "1"]
    p = subprocess.Popen(cmd, cwd=tmp_path, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ck = tmp_path / "artifacts" / "ck" / "fold0_resume.pt"
    deadline = time.time() + 300
    while not ck.exists() and p.poll() is None and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.3)  # let the checkpoint write finish
    p.kill()
    p.wait()
    assert ck.exists(), "no checkpoint was written"
    out = _run([*args, "--epochs", "40", "--name", "ck", "--folds", "0", "--checkpoint-every", "1"], tmp_path, plugin_root)
    assert "resumed at epoch" in out and not ck.exists()
