"""Compute efficiency (#91): measuring a run (kgkit.bench), static slow-pattern checks (kgkit.speedcheck), GPU options
of the feature search, and the compute-efficiency skill being the one home that the domain skills point to."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kgkit import bench as B
from kgkit import featsearch as FS
from kgkit import speedcheck as SC

PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "kaggle-grandmaster"
SKILLS = PLUGIN / "skills"
SKILL = SKILLS / "compute-efficiency"


# ----------------------------------------------------------------------------- kgkit.bench
def test_verdict_names_the_limit():
    assert B.verdict({"data": 100.0, "compute": 400.0, "full": 95.0}) == B.INPUT_BOUND
    assert B.verdict({"data": 900.0, "compute": 300.0, "full": 290.0}) == B.COMPUTE_BOUND
    assert B.verdict({"data": 900.0, "compute": 300.0, "full": 100.0}) == B.STALLS
    assert B.verdict({"data": 900.0}) == ""


def test_measure_counts_items_per_second():
    calls = {"a": 0}

    def step():
        calls["a"] += 1
        return 8

    rates = B.measure({"a": step, "b": lambda: 4}, n=5, warmup=2)
    assert calls["a"] == 7 and set(rates) == {"a", "b"} and rates["a"] > 0 and rates["b"] > 0
    with pytest.raises(ValueError):
        next(B.forever([]))
    it = B.forever([1, 2])
    assert [next(it) for _ in range(5)] == [1, 2, 1, 2, 1]


def test_report_records_and_prints_the_speed_up(tmp_path):
    lines = []
    B.report("cnx", {"data": 50.0, "compute": 200.0, "full": 48.0}, config={"batch_size": 32, "image_dir": "raw"},
             root=tmp_path, kind="train_image", out=lines.append)
    assert any("INPUT-BOUND" in x for x in lines) and not any("vs previous" in x for x in lines)
    lines.clear()
    B.report("cnx", {"data": 400.0, "compute": 200.0, "full": 192.0}, config={"batch_size": 32, "image_dir": "npy"},
             root=tmp_path, kind="train_image", out=lines.append)
    text = "\n".join(lines)
    assert "COMPUTE-BOUND" in text and "x4.00 faster" in text and "image_dir raw -> npy" in text
    # another run name of the same script is compared too, and says which run it is compared with
    lines.clear()
    B.report("cnx_448", {"data": 400.0, "compute": 100.0, "full": 96.0}, config={"batch_size": 32, "image_dir": "npy"},
             root=tmp_path, kind="train_image", out=lines.append)
    assert any("this script (cnx)" in x and "x0.50 slower" in x for x in lines)
    recs = B.records(tmp_path / "reports" / "bench.jsonl")
    assert [r["name"] for r in recs] == ["cnx", "cnx", "cnx_448"] and recs[0]["verdict"] == B.INPUT_BOUND


def _fake_smi(tmp_path, rows):
    p = tmp_path / "fake_smi.py"
    p.write_text("print(%r)\n" % "\n".join(rows), encoding="utf-8")
    return str(p)


def _kgkit(args, cwd, **env):
    plugin = str(PLUGIN)
    e = dict(os.environ, PYTHONPATH=os.pathsep.join([plugin, os.environ.get("PYTHONPATH", "")]), **env)
    return subprocess.run([sys.executable, "-m", "kgkit", *args], cwd=cwd, env=e, capture_output=True, text=True,
                          timeout=300)


def _sleep_cmd(seconds):
    return f'"{sys.executable}" -c "import time; time.sleep({seconds})"'


def test_bench_command_idle_gpu_and_speed_up(tmp_path):
    smi = _fake_smi(tmp_path, ["0, 2, 300", "1, 0, 0"])
    r = _kgkit(["bench", "--run", _sleep_cmd(1.5), "--run", _sleep_cmd(0.2), "--sample-s", "0.1"], tmp_path,
               KGKIT_NVIDIA_SMI=smi)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "GPU IDLE" in r.stdout and "gpu0 util mean 2%" in r.stdout
    m = re.search(r"run 2 vs run 1: x([\d.]+) faster", r.stdout)
    assert m and float(m.group(1)) > 1.3, r.stdout  # interpreter start-up is in both wall times
    # the same command again: compared with its previous bench
    r = _kgkit(["bench", "--run", _sleep_cmd(0.2), "--sample-s", "0.1"], tmp_path, KGKIT_NVIDIA_SMI=smi)
    assert "vs previous bench of this command" in r.stdout
    assert len(B.records(tmp_path / "reports" / "bench.jsonl")) == 3


def test_bench_command_busy_one_gpu_and_failures(tmp_path):
    smi = _fake_smi(tmp_path, ["0, 97, 14000", "1, 0, 3"])
    r = _kgkit(["bench", "--run", _sleep_cmd(0.5), "--sample-s", "0.1", "--no-record"], tmp_path,
               KGKIT_NVIDIA_SMI=smi)
    assert r.returncode == 0 and "GPU BUSY" in r.stdout and "gpu1 idle while another GPU worked" in r.stdout
    assert not (tmp_path / "reports" / "bench.jsonl").exists()
    bad = f'"{sys.executable}" -c "import sys; sys.exit(3)"'
    r = _kgkit(["bench", "--run", _sleep_cmd(0.1), "--run", bad, "--no-record"], tmp_path,
               KGKIT_NVIDIA_SMI=_fake_smi(tmp_path, []))
    assert r.returncode == 1 and "exit code 3" in r.stdout and "not compared" in r.stdout
    assert "no GPU sampled" in r.stdout
    assert _kgkit(["bench"], tmp_path).returncode != 0


# ----------------------------------------------------------------------------- kgkit.speedcheck
SLOW_TRAIN = """
import timm, torch
from torch.utils.data import DataLoader
model = timm.create_model("resnet18")
dl = DataLoader(ds, batch_size=32, shuffle=True)
for x, y in dl:
    loss = model(x.cuda()).sum()
    loss.backward()
df = df.apply(lambda r: r.a + r.b, axis=1)
"""


def test_speedcheck_flags_slow_training_code():
    assert SC.kind(SLOW_TRAIN) == "train" and SC.domain(SLOW_TRAIN) == "vision"
    codes = {f.code for f in SC.check(SLOW_TRAIN) if f.level == "warn"}
    assert codes == {"NO_AMP", "NO_CUDNN_BENCHMARK", "LOADER_IN_MAIN_PROCESS", "ROW_LOOP"}
    fixed = SLOW_TRAIN.replace("shuffle=True)", "shuffle=True, num_workers=4, persistent_workers=True)")
    fixed = fixed.replace("loss = model", "torch.backends.cudnn.benchmark = True\n    with torch.autocast('cuda'):\n"
                                          "        loss = model")
    assert {f.code for f in SC.check(fixed) if f.level == "warn"} == {"ROW_LOOP"}


def test_speedcheck_gbdt_and_inference():
    xgb = "import xgboost as xgb\nm = xgb.XGBClassifier(tree_method='gpu_hist')\nm.fit(X, y)\n"
    (f,) = SC.check(xgb)
    assert f.code == "XGB_TREE_METHOD" and "device='cuda'" in f.message and SC.domain(xgb) == "tabular"
    assert SC.check(xgb.replace("gpu_hist", "hist")) == []
    svr = "from sklearn.svm import SVR\nSVR().fit(X, y)\n"
    assert [f.code for f in SC.check(svr)] == ["SKLEARN_ON_GPU"] and SC.check(svr)[0].level == "info"
    infer = "import torch\nmodel = torch.load('m.pt')\nmodel.eval()\npred = model(x)\n"
    assert SC.kind(infer) == "infer" and [f.code for f in SC.check(infer)] == ["NO_INFERENCE_MODE"]
    assert SC.check(infer.replace("pred = model(x)", "with torch.inference_mode():\n    pred = model(x)")) == []
    assert SC.kind("print('hello')") is None and SC.check("def broken(:\n") == []
    # bf16 chosen with is_bf16_supported(): True on a T4/P100 too (emulated, slow); a comment about it is fine
    bf16 = ("import torch\n# not torch.cuda.is_bf16_supported(): emulated on T4\n"
            "dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16\n")
    assert [(f.code, f.line) for f in SC.check(bf16)] == [("BF16_EMULATED", 3)]
    assert SC.check(bf16.replace("is_bf16_supported()", "is_bf16_supported(including_emulation=False)")) == []


def test_speedcheck_flags_a_model_trained_in_its_checkpoint_dtype():
    """transformers 5 loads the checkpoint's dtype (DeBERTa-v3: fp16), which crashed train_transformer.py's GradScaler
    on a Kaggle T4 ('Attempting to unscale FP16 gradients'): full fine-tuning needs fp32 master weights."""
    def hf(src):
        return [(f.code, f.line) for f in SC.check(src) if f.code == "HF_CHECKPOINT_DTYPE"]

    head = "import torch\nfrom transformers import AutoConfig, AutoModel, AutoTokenizer\n"
    loop = "\nloss = model(x).sum()\nloss.backward()\n"
    assert hf(head + "model = AutoModel.from_pretrained(name, config=cfg)" + loop) == [("HF_CHECKPOINT_DTYPE", 3)]
    assert hf(head + "model = AutoModel.from_pretrained(name).to(device)" + loop) == [("HF_CHECKPOINT_DTYPE", 3)]
    for fine in ("model = AutoModel.from_pretrained(name).float()",
                 "model = AutoModel.from_pretrained(name).to(torch.float32)",
                 "model = AutoModel.from_pretrained(name, dtype=torch.float32)",
                 "model = AutoModel.from_pretrained(name, **kw)",
                 "tok = AutoTokenizer.from_pretrained(name)\ncfg = AutoConfig.from_pretrained(name)",
                 "from peft import LoraConfig\nmodel = AutoModel.from_pretrained(name)"):  # a frozen fp16/4-bit base
        assert hf(head + fine + loop) == [], fine
    assert hf(head + "model = AutoModel.from_pretrained(name)\nwith torch.inference_mode():\n    p = model(x)\n") == []


@pytest.mark.parametrize("template", ["train_image.py", "train_transformer.py", "train_gbdt.py", "inference_kernel.py"])
def test_templates_follow_the_compute_defaults(template):
    """What the templates do matches the guidance: no slow pattern the write hook would report."""
    text = (PLUGIN / "templates" / template).read_text(encoding="utf-8")
    warn = [f for f in SC.check(text) if f.level == "warn"]
    assert warn == [], SC.format_findings(template, warn)


def test_speedcheck_command(tmp_path):
    (tmp_path / "train.py").write_text(SLOW_TRAIN, encoding="utf-8")
    r = _kgkit(["speedcheck", "train.py"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "train.py:5: [warn] LOADER_IN_MAIN_PROCESS" in r.stdout
    assert "defaults for vision" in r.stdout and "compute-efficiency/references/vision.md" in r.stdout


# ----------------------------------------------------------------------------- feature search on the GPU
def _small_problem(n=600):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    y = (X["a"] + 0.5 * X["b"] + rng.normal(0, 0.5, n) > 0).astype(int).to_numpy() + 1  # labels 1/2, not 0/1
    return X, y, np.arange(n) % 3


@pytest.mark.skipif(not FS._have("xgboost"), reason="needs xgboost")
def test_feature_search_screens_with_xgboost():
    X, y, folds = _small_problem()
    res = FS.cv_scores(X, y, folds, "auc", model="xgb", n_estimators=60)
    assert res["model"] == "xgb" and len(res["scores"]) == 3 and min(res["scores"]) > 0.7
    assert res["importance"]["a"] > res["importance"]["c"]


def test_lightgbm_device_params_on_cpu():
    assert FS.lightgbm_device_params("cpu") == ({}, "")


def test_lightgbm_gpu_probe_never_breaks_a_run():
    """--device cuda: a GPU build that works is used, anything else falls back to the CPU with a note."""
    if not FS._have_lgbm():
        pytest.skip("needs lightgbm")
    code = "from kgkit import featsearch as FS; p, n = FS.lightgbm_device_params('cuda'); print(repr(p)); print(n)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(PLUGIN), os.environ.get("PYTHONPATH", "")])))
    assert r.returncode == 0, r.stderr
    params, note = r.stdout.splitlines()[0], r.stdout.splitlines()[1]
    assert ("device_type" in params and "on the GPU" in note) or (params == "{}" and "on the CPU" in note)


# ----------------------------------------------------------------------------- one home, linked from every domain
DOMAIN_SKILLS = ["deep-learning-training", "computer-vision", "nlp-and-llm", "tabular-mastery", "audio-and-signal",
                 "recsys-and-ranking", "time-series", "code-competitions", "simulation-and-optimization", "kaggle-gpu"]


def test_compute_skill_references_exist_and_have_contents():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert len(text.splitlines()) < 500
    refs = sorted(set(re.findall(r"\(references/([\w-]+\.md)", text)))
    assert set(refs) == {p.name for p in (SKILL / "references").glob("*.md")}
    for name in refs:
        lines = (SKILL / "references" / name).read_text(encoding="utf-8").splitlines()
        if len(lines) > 100:
            assert any(x.startswith("Contents:") for x in lines[:6]), f"{name}: long reference without contents"


def test_compute_skill_links_resolve():
    for md in [SKILL / "SKILL.md", *(SKILL / "references").glob("*.md")]:
        for target in re.findall(r"\]\(([^)#\s]+\.md)(?:#[^)]*)?\)", md.read_text(encoding="utf-8")):
            assert (md.parent / target).is_file(), f"{md.name} links to missing {target}"


@pytest.mark.parametrize("name", DOMAIN_SKILLS)
def test_domain_skills_point_to_the_compute_skill(name):
    assert "compute-efficiency" in (SKILLS / name / "SKILL.md").read_text(encoding="utf-8")


def test_no_second_home_for_speed_guidance():
    assert not (SKILLS / "kaggle-gpu" / "references" / "throughput.md").exists()
    for md in SKILLS.rglob("*.md"):
        assert "throughput.md" not in md.read_text(encoding="utf-8"), md
    # the old contradiction: num_workers = all CPU cores while two folds share them
    dl = (SKILLS / "deep-learning-training" / "SKILL.md").read_text(encoding="utf-8")
    assert "`num_workers` =\n  CPU cores," not in dl and "CPU cores ÷" in dl


def test_bench_is_never_pre_approved():
    """`kgkit bench --run` runs any shell command, so no kg-* command may pre-approve it."""
    for p in SKILLS.glob("kg-*/SKILL.md"):
        front = p.read_text(encoding="utf-8").split("---")[1]
        assert "kgkit bench" not in front, p.parent.name


def test_bench_main_needs_a_command():
    with pytest.raises(SystemExit):
        B.main_bench(SimpleNamespace(run=[], name=None, sample_s=0.5, no_record=True))


def test_bench_record_is_json(tmp_path):
    B.report("x", {"full": 10.0}, config={"a": [1, 2]}, root=tmp_path, out=lambda s: None)
    rec = json.loads((tmp_path / "reports" / "bench.jsonl").read_text(encoding="utf-8"))
    assert rec["config"] == {"a": [1, 2]} and rec["verdict"] == ""


def test_merge_brings_remote_records_home(tmp_path):
    """A Kaggle session's bench records join the local history, with the speed-up against it, once."""
    ws, remote = tmp_path / "ws", tmp_path / "remote"
    B.report("cnx", {"data": 50.0, "compute": 200.0, "full": 48.0}, config={"image_dir": "raw"}, root=ws,
             kind="train_image", out=lambda s: None)
    B.report("cnx", {"data": 400.0, "compute": 200.0, "full": 192.0}, config={"image_dir": "npy"}, root=remote,
             kind="train_image", out=lambda s: None)
    local, src = ws / "reports" / "bench.jsonl", remote / "reports" / "bench.jsonl"
    text = "\n".join(B.merge(src, local))
    assert "[bench] remote cnx" in text and "COMPUTE-BOUND" in text
    assert "x4.00 faster" in text and "image_dir raw -> npy" in text
    assert [r["config"]["image_dir"] for r in B.records(local)] == ["raw", "npy"]
    assert B.merge(src, local) == [] and B.merge(tmp_path / "missing.jsonl", local) == []
