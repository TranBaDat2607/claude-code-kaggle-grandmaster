"""Remote Kaggle GPU training: budget guard, quota accounting, kernel build, runner and collect.

The runner is executed for real in a simulated /kaggle layout (KAGGLE_INPUT / KAGGLE_WORKING / KG_RUN_DIR,
two fake GPUs), so the whole build -> run -> collect path is covered without network or a GPU.
"""

import base64
import datetime as dt
import io
import json
import os
import re
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import pytest

from kgkit import gpu as G
from kgkit import state as S
from kgkit.budget import TrainBudget, baseline_best_at
from kgkit.experiment import Ledger

FAKE_TRAIN = r'''
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from kgkit.budget import TrainBudget
from kgkit.experiment import Ledger

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--folds", type=int, nargs="+")
ap.add_argument("--assemble", action="store_true")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--quality", type=float, default=0.1)
a = ap.parse_args()
train = pd.read_csv("data/train.csv")
folds = pd.read_csv("data/folds.csv")["fold"].to_numpy()
out = Path("artifacts") / a.name
out.mkdir(parents=True, exist_ok=True)
for f in ([] if a.assemble else a.folds):
    tb = TrainBudget(out, f, a.epochs, True)
    stopped = None
    for e in range(a.epochs):
        tb.start_epoch()
        time.sleep(0.05)
        act = tb.end_epoch(e, 0.5 + a.quality * e)
        if act != "continue":
            stopped = act
            tb.write_status("pruned" if act == "prune" else act)
            break
    if stopped:
        print("stopped", stopped)
        sys.exit(0)
    va = np.flatnonzero(folds == f)
    np.save(out / f"fold{f}_oof.npy", np.full(len(va), 0.5 + a.quality * (a.epochs - 1)))
    np.save(out / f"fold{f}_idx.npy", va)
    np.save(out / f"fold{f}.pt", np.zeros(3))  # stands in for weights
    tb.write_status("done")
    print("fold", f, "done")
if os.environ.get("KG_DEFER_ASSEMBLE") and not a.assemble:
    sys.exit(0)
all_folds = sorted(set(folds.tolist()))
if all(( out / f"fold{f}_oof.npy").exists() for f in all_folds):
    oof = np.zeros(len(train))
    scores = []
    for f in all_folds:
        idx = np.load(out / f"fold{f}_idx.npy")
        oof[idx] = np.load(out / f"fold{f}_oof.npy")
        scores.append(float(oof[idx].mean()))
    rec = Ledger().log(a.name, float(np.mean(scores)), fold_scores=scores, oof=oof, test_pred=np.zeros(4))
    print("logged", rec["id"])
'''


# ----------------------------------------------------------------------------- budget
def test_budget_timeout(tmp_path):
    tb = TrainBudget(tmp_path, 0, total_epochs=10, env={"KG_DEADLINE": str(time.time() + 30)})
    tb.start_epoch()
    assert tb.end_epoch(0, 0.5) == "timeout"  # 29 s left < one epoch + 60 s margin
    tb2 = TrainBudget(tmp_path, 1, total_epochs=10, env={"KG_DEADLINE": str(time.time() + 3600)})
    tb2.start_epoch()
    assert tb2.end_epoch(0, 0.5) == "continue"
    assert json.loads((tmp_path / "curve_fold1.json").read_text())["points"][0]["score"] == 0.5


def test_budget_prune_and_resume_curve(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "curve_fold0.json").write_text(json.dumps(
        {"fold": 0, "epochs": 10, "greater_is_better": True,
         "points": [{"epoch": e, "score": 0.6 + 0.02 * e, "seconds": 1} for e in range(10)]}))
    assert baseline_best_at(json.loads((base / "curve_fold0.json").read_text()), 0.3, True) == pytest.approx(0.64)
    env = {"KG_PRUNE_BASELINE": str(base), "KG_PRUNE_MARGIN": "0.01", "KG_PRUNE_MIN_FRAC": "0.3"}
    out = tmp_path / "cand"
    tb = TrainBudget(out, 0, total_epochs=10, env=env)
    actions = [tb.end_epoch(e, 0.55 + 0.005 * e) for e in range(3)]
    assert actions == ["continue", "continue", "prune"]  # never before 30% of the schedule
    good = TrainBudget(tmp_path / "good", 0, total_epochs=10, env=env)
    assert all(good.end_epoch(e, 0.6 + 0.02 * e) == "continue" for e in range(9))
    # a resumed fold reloads its curve and best-so-far
    again = TrainBudget(out, 0, total_epochs=10, env={})
    assert len(again.points) == 3 and again.best == pytest.approx(0.56)
    low_env = {"KG_PRUNE_BASELINE": str(base), "KG_PRUNE_MARGIN": "0.01"}
    lower = TrainBudget(tmp_path / "low", 0, total_epochs=10, greater_is_better=False, env=low_env)
    assert lower.end_epoch(4, 0.9) == "prune"  # lower is better: 0.9 vs the baseline's best (min) 0.6
    ok = TrainBudget(tmp_path / "low_ok", 0, total_epochs=10, greater_is_better=False, env=low_env)
    assert ok.end_epoch(4, 0.55) == "continue"


# ----------------------------------------------------------------------------- quota
QUOTA_JSON = json.dumps([
    {"resource": "GPU", "used": "19.55h", "remaining": "10.45h", "total": "30.00h", "refreshAt": "2099-01-03T00:00:00"},
    {"resource": "TPU", "used": "0.00h", "remaining": "20.00h", "total": "20.00h", "refreshAt": "2099-01-03T00:00:00"},
])


def test_parse_quota_and_in_flight(tmp_path):
    q = G.parse_quota(QUOTA_JSON)
    assert q["remaining_h"] == pytest.approx(10.45) and q["total_h"] == 30 and q["refresh_at"].startswith("2099-01-03")
    S.CompetitionState(slug="demo").save(tmp_path)
    G.quota_path(tmp_path).write_text(json.dumps(q))
    assert G.available_hours(tmp_path) == pytest.approx(10.45)
    now = G._iso(G._now())
    G.log_run(tmp_path, {"kernel": "u/a", "hours": 4.0, "pushed_at": now, "status": "running"})
    G.log_run(tmp_path, {"kernel": "u/b", "hours": 4.0, "pushed_at": now, "status": "collected"})
    assert G.available_hours(tmp_path) == pytest.approx(6.45, abs=0.05)
    # a snapshot taken after the week rolled over is treated as a full quota
    stale = dict(q, refresh_at="2000-01-01T00:00:00Z", remaining_h=0.0)
    G.quota_path(tmp_path).write_text(json.dumps(stale))
    assert G.cached_quota(tmp_path)["remaining_h"] == 30


def test_normalize_status():
    assert G.normalize_status('u/k has status "KernelWorkerStatus.COMPLETE"') == "complete"
    assert G.normalize_status('u/k has status "running"') == "running"
    assert G.normalize_status('u/k has status "KernelWorkerStatus.ERROR"\nFailure message: "x"') == "error"
    assert G.normalize_status('u/k has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"') == "cancelled"


# ----------------------------------------------------------------------------- build + run + collect
@pytest.fixture
def gws(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    S.CompetitionState(slug="demo-comp", metric="auc", target="y", id_col="id").save(root)
    (root / "data").mkdir()
    n = 40
    pd.DataFrame({"id": range(n), "y": np.arange(n) % 2}).to_csv(root / "data" / "train.csv", index=False)
    pd.DataFrame({"id": range(n), "fold": np.arange(n) % 4}).to_csv(root / "data" / "folds.csv", index=False)
    (root / "src").mkdir()
    (root / "src" / "fake_train.py").write_text(FAKE_TRAIN, encoding="utf-8")
    (root / "src" / "big.npy").write_bytes(b"0" * 1000)  # must not be bundled
    return root


def _decode(script: str, var: str) -> bytes:
    return base64.b64decode(re.search(var + r' = "([^"]+)"', script).group(1))


def _cmd(name, extra=""):
    return f'"{sys.executable}" src/fake_train.py --name {name} --folds {{fold}}{extra}'


def test_build_kernel(gws):
    kdir = G.build_kernel(gws, "fk", _cmd("fk"), hours=2.5, user="tester", datasets=["tester/demo-512"])
    meta = json.loads((kdir / "kernel-metadata.json").read_text())
    assert meta["id"] == "tester/fk-train" and meta["machine_shape"] == "NvidiaTeslaT4" and meta["enable_gpu"]
    assert meta["competition_sources"] == ["demo-comp"] and meta["dataset_sources"] == ["tester/demo-512"]
    assert meta["is_private"] and meta["kernel_type"] == "script"
    script = (kdir / meta["code_file"]).read_text()
    assert "__KG_" not in script
    cfg = json.loads(_decode(script, "CONFIG_B64"))
    job = cfg["jobs"][0]
    assert job["folds"] == [0, 1, 2, 3] and cfg["hours"] == 2.5
    assert job["assemble_cmd"].endswith("--name fk --assemble") and cfg["sources"] == ["demo-comp", "tester/demo-512"]
    names = zipfile.ZipFile(io.BytesIO(_decode(script, "PAYLOAD_B64"))).namelist()
    assert "src/fake_train.py" in names and "kgkit/gpu.py" in names and "data/folds.csv" in names
    assert ".kaggle-gm/competition.json" in names and "src/big.npy" not in names
    assert not any(n.startswith("data/train") for n in names)
    with pytest.raises(SystemExit):
        G.build_kernel(gws, "fk", _cmd("fk"), hours=13, user="tester")
    with pytest.raises(SystemExit):  # duplicate job names would share artifacts/<name>/
        G.build_kernel(gws, "fk", _cmd("fk"), hours=1, user="tester", extra_jobs=[("fk", _cmd("fk"))])


def _run_kernel(kdir, base, gpus="2"):
    meta = json.loads((kdir / "kernel-metadata.json").read_text())
    inp, work = base / "input", base / "working"
    env = dict(os.environ, KAGGLE_INPUT=str(inp), KAGGLE_WORKING=str(work), KG_RUN_DIR=str(base / "run"),
               KG_FAKE_GPUS=gpus)
    env.pop("PYTHONPATH", None)
    r = subprocess.run([sys.executable, str(kdir / meta["code_file"])], env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout, work


def _kaggle_input(base, gws):
    comp = base / "input" / "competitions" / "demo-comp"  # nested layout: found by name, not by fixed path
    comp.mkdir(parents=True)
    (comp / "train.csv").write_bytes((gws / "data" / "train.csv").read_bytes())


def test_runner_end_to_end_and_collect(gws, tmp_path):
    kdir = G.build_kernel(gws, "fk", _cmd("fk"), hours=2, user="tester")
    _kaggle_input(tmp_path / "k1", gws)
    out, work = _run_kernel(kdir, tmp_path / "k1")
    assert "KG_RUN_STATUS fk: complete" in out
    rep = json.loads((work / "kg_run.json").read_text())
    job = rep["jobs"][0]
    assert job["folds_done"] == [0, 1, 2, 3] and job["assembled"] is True and len(rep["gpus"]) == 2
    assert {s["gpu"] for s in job["fold_status"].values()} == {0, 1}  # both GPUs used
    assert len(rep["ledger"]) == 1 and (work / "artifacts" / "fk" / "curve_fold0.json").is_file()

    # one local experiment already exists: the import must get a fresh id and keep artefacts consistent
    Ledger(gws).log("local", 0.7, fold_scores=[0.7] * 4)
    G.log_run(gws, {"kernel": "tester/fk-train", "version": 1, "name": "fk", "hours": 2, "status": "running",
                    "pushed_at": G._iso(G._now())})
    res = G.collect(gws, "tester/fk-train", from_dir=work)
    assert res["status"] == "complete" and len(res["imported"]) == 1
    rec = Ledger(gws).records()[-1]
    assert rec["id"].startswith("0002_") and rec["remote"]["kernel"] == "tester/fk-train"
    assert rec["remote"]["gpus"] == ["fake-gpu", "fake-gpu"] and rec["gpu_hours"] >= 0
    assert Ledger(gws).load_oof(rec["id"]).shape == (40,)
    assert rec["folds_hash"] == Ledger(gws)._folds_hash(None)  # same frozen split remotely and locally
    run = G.runs(gws)[-1]
    assert run["status"] == "collected" and run["exp_ids"] == [rec["id"]] and run["gpu_hours"] is not None
    # collecting again is idempotent
    G.collect(gws, "tester/fk-train", from_dir=work)
    assert len(Ledger(gws).records()) == 2
    assert "fk" in G.run_table(gws)


def test_runner_resume_skips_done_folds(gws, tmp_path):
    prev = tmp_path / "k2" / "input" / "fk-train" / "artifacts" / "fk"
    prev.mkdir(parents=True)
    va = np.flatnonzero(np.arange(40) % 4 == 0)
    np.save(prev / "fold0_oof.npy", np.full(len(va), 0.7))
    np.save(prev / "fold0_idx.npy", va)
    (prev / "fold0_status.json").write_text(json.dumps({"fold": 0, "status": "done", "best": 0.7}))
    kdir = G.build_kernel(gws, "fk", _cmd("fk"), hours=2, user="tester", resume_from=["tester/fk-train"],
                          slug="fk-train-r2")
    meta = json.loads((kdir / "kernel-metadata.json").read_text())
    assert meta["kernel_sources"] == ["tester/fk-train"] and meta["id"] == "tester/fk-train-r2"
    _kaggle_input(tmp_path / "k2", gws)
    out, work = _run_kernel(kdir, tmp_path / "k2", gpus="1")
    job = json.loads((work / "kg_run.json").read_text())["jobs"][0]
    assert "fk fold 0 already done (resumed)" in out
    assert job["folds_done"] == [0, 1, 2, 3] and job["fold_status"]["0"].get("seconds") is None
    assert job["assembled"] is True


def test_runner_prune_screen(gws, tmp_path):
    base = gws / "artifacts" / "good"
    base.mkdir(parents=True)
    (base / "curve_fold0.json").write_text(json.dumps(
        {"fold": 0, "epochs": 5, "greater_is_better": True,
         "points": [{"epoch": e, "score": 0.5 + 0.1 * e, "seconds": 1} for e in range(5)]}))
    Ledger(gws).log("good", 0.9, fold_scores=[0.9, 0.9, 0.9, 0.9])
    kdir = G.build_kernel(gws, "bad", _cmd("bad", " --epochs 5 --quality 0.01"), hours=1, folds=[0], user="tester",
                          prune_against="good", prune_margin=0.05)
    _kaggle_input(tmp_path / "k3", gws)
    out, work = _run_kernel(kdir, tmp_path / "k3")
    job = json.loads((work / "kg_run.json").read_text())["jobs"][0]
    assert job["fold_status"]["0"]["status"] == "pruned" and job["folds_incomplete"] == [0]
    res = G.collect(gws, "tester/bad-train", from_dir=work)
    assert res["status"] == "failed" and not res["imported"]
    assert "pruned on folds" in res["summary"] and "vs 0001_good fold 0" in res["summary"]


def test_two_screens_share_one_session(gws, tmp_path):
    """Two 1-fold screens of different ideas run side by side on the two T4s of one session."""
    Ledger(gws).log("base", 0.6, fold_scores=[0.6, 0.6, 0.6, 0.6])
    kdir = G.build_kernel(gws, "idea-a", _cmd("idea-a", " --quality 0.2"), hours=1, folds=[0], user="tester",
                          extra_jobs=[("idea-b", _cmd("idea-b", " --quality 0.01"))])
    assert json.loads((kdir / "kg-train.json").read_text())["jobs"] == ["idea-a", "idea-b"]
    _kaggle_input(tmp_path / "k4", gws)
    out, work = _run_kernel(kdir, tmp_path / "k4")
    rep = json.loads((work / "kg_run.json").read_text())
    gpus_used = {j["name"]: j["fold_status"]["0"]["gpu"] for j in rep["jobs"]}
    assert sorted(gpus_used.values()) == [0, 1]  # one screen per GPU
    res = G.collect(gws, "tester/idea-a-train", from_dir=work)
    assert "idea-a" in res["summary"] and "idea-b" in res["summary"] and "vs 0001_base fold 0" in res["summary"]
    assert res["status"] == "complete" and not res["imported"]  # screens never log a (fake) full-CV ledger entry
    assert all(j["assembled"] is None for j in rep["jobs"])  # a fold subset is not assembled
    assert G.runs(gws)[-1]["fold_scores"]["idea-a"]["0"] == pytest.approx(0.9)
    assert "idea-a f0:0.9000" in G.run_table(gws)


def test_diagnose_flags_idle_gpu_and_input_bound():
    rep = {"gpus": ["T4", "T4"], "gpu_util": {"0": {"util_mean": 41.0}, "1": {"util_mean": 0.0}},
           "fold_status": {"0": {"status": "done", "seconds": 3600}}}
    msgs = " ".join(G.diagnose(rep))
    assert "second T4 was idle" in msgs and "input-bound" in msgs


def test_plan_mentions_reserve_and_reset(gws):
    st = S.load(gws)
    st.deadline = (dt.date.today() + dt.timedelta(days=20)).isoformat()
    st.save(gws)
    soon = G._iso(G._now() + dt.timedelta(hours=20))
    G.quota_path(gws).write_text(json.dumps({"used_h": 20, "remaining_h": 10, "total_h": 30, "refresh_at": soon,
                                             "fetched_at": G._iso(G._now())}))
    txt = G.plan(gws, refresh=False)
    assert "USE IT OR LOSE IT" in txt and "Reserve" in txt and "GPU-hours in total" in txt


# ----------------------------------------------------------------------------- templates
@pytest.mark.skipif(__import__("importlib").util.find_spec("timm") is None, reason="needs torch + timm")
def test_image_template_timeout_resume_and_assemble(tmp_path, plugin_root):
    from test_templates import _run
    from PIL import Image

    rng = np.random.default_rng(0)
    (tmp_path / "data" / "train_images").mkdir(parents=True)
    rows = []
    for i in range(30):
        lab = ["a", "b", "c"][i % 3]
        arr = np.clip(np.array([200 * (lab == "a"), 200 * (lab == "b"), 200 * (lab == "c")]) +
                      rng.normal(0, 20, (32, 32, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(tmp_path / "data" / "train_images" / f"{i}.png")
        rows.append({"image_id": i, "image_path": f"{i}.png", "label": lab})
    pd.DataFrame(rows).to_csv(tmp_path / "data" / "train.csv", index=False)
    _run(["-m", "kgkit", "init", "img", "--metric", "accuracy", "--target", "label", "--id-col", "image_id"],
         tmp_path, plugin_root)
    _run(["-m", "kgkit", "folds", "data/train.csv", "--target", "label", "--id-col", "image_id", "--n-splits", "2",
          "--out", "data/folds.csv"], tmp_path, plugin_root)
    (tmp_path / "src").mkdir(exist_ok=True)
    src = (plugin_root / "templates" / "train_image.py").read_text(encoding="utf-8")
    (tmp_path / "src" / "train_image.py").write_text(src.replace("num_workers=4", "num_workers=0"), encoding="utf-8")
    args = ["src/train_image.py", "--backbone", "resnet18", "--no-pretrained", "--img-size", "32", "--batch-size", "8",
            "--epochs", "3", "--name", "r18"]
    env_deadline = {"KG_DEADLINE": str(time.time() + 20), "KG_DEFER_ASSEMBLE": "1"}
    old = dict(os.environ)
    try:
        os.environ.update(env_deadline)
        out = _run([*args, "--folds", "0"], tmp_path, plugin_root)
    finally:
        os.environ.clear()
        os.environ.update(old)
    assert "checkpointed after epoch 0" in out and "not logged" in out
    art = tmp_path / "artifacts" / "r18"
    assert (art / "fold0_resume.pt").is_file()
    assert json.loads((art / "fold0_status.json").read_text())["status"] == "timeout"
    os.environ["KG_DEFER_ASSEMBLE"] = "1"
    try:
        out = _run([*args, "--folds", "0", "1"], tmp_path, plugin_root)
    finally:
        os.environ.pop("KG_DEFER_ASSEMBLE")
    assert "resumed at epoch 1" in out and "assembly deferred" in out
    assert not (art / "fold0_resume.pt").exists()
    assert len(Ledger(tmp_path).records()) == 0
    out = _run([*args, "--assemble"], tmp_path, plugin_root)
    rec = Ledger(tmp_path).records()[-1]
    assert rec["name"] == "r18" and len(rec["fold_scores"]) == 2
    assert len(json.loads((art / "curve_fold0.json").read_text())["points"]) == 3


@pytest.mark.parametrize("rel", ["kgkit/gpu.py", "kgkit/budget.py", "kgkit/state.py", "templates/kaggle_gpu_runner.py"])
def test_stdlib_only(plugin_root, rel):
    """Hooks import gpu/budget/state, and the runner executes inside Kaggle: standard library only."""
    import ast

    tree = ast.parse((plugin_root / rel).read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add(node.module.split(".")[0])
    assert mods <= set(sys.stdlib_module_names) | {"__future__"}, mods - set(sys.stdlib_module_names)


def test_runner_prints_ascii(plugin_root):
    src = (plugin_root / "templates" / "kaggle_gpu_runner.py").read_text(encoding="utf-8")
    assert src.isascii()
