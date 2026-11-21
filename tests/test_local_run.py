"""`gpu push --backend local`: the `gpu build` runner on this machine, as a background process, with no quota.

The runner runs for real (one fake GPU, KG_FAKE_GPUS), so build -> push -> wait -> collect is covered on any OS
without a GPU."""

import json
import sys
import time

import numpy as np
import pandas as pd
import pytest

from kgkit import gpu as G
from kgkit import localrun as LR
from kgkit import state as S
from kgkit.experiment import Ledger

from test_gpu import FAKE_TRAIN


@pytest.fixture
def lws(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    S.CompetitionState(slug="demo-comp", metric="auc", target="y", id_col="id").save(root)
    (root / "data").mkdir()
    n = 40
    pd.DataFrame({"id": range(n), "y": np.arange(n) % 2}).to_csv(root / "data" / "train.csv", index=False)
    pd.DataFrame({"id": range(n), "fold": np.arange(n) % 4}).to_csv(root / "data" / "folds.csv", index=False)
    (root / "src").mkdir()
    (root / "src" / "fake_train.py").write_text(FAKE_TRAIN, encoding="utf-8")
    monkeypatch.chdir(root)
    real = LR.shutil.which
    monkeypatch.setattr(LR.shutil, "which", lambda name, *a, **k: "/usr/bin/nvidia-smi" if name == "nvidia-smi"
                        else real(name, *a, **k))
    monkeypatch.setattr(G, "cpu_torch_line", lambda *a, **k: "")
    monkeypatch.setenv("KG_FAKE_GPUS", "1")
    return root


def _cmd(name):
    return f'"{sys.executable}" src/fake_train.py --name {name} --folds {{fold}}'


def _wait(root, kernel, timeout=180):
    t = time.time() + timeout
    while time.time() < t:
        s = LR.status(G.latest_run(root, kernel))
        if s != "running":
            return s
        time.sleep(0.5)
    raise AssertionError(f"{kernel} still running after {timeout}s:\n" + "\n".join(LR.log_tail(G.latest_run(root, kernel))))


def test_local_run_end_to_end(lws):
    kdir = G.build_kernel(lws, "fk", _cmd("fk"), hours=2, user="tester")
    rec = G.push_local(lws, kdir, hours=20)  # past Kaggle's 12 h session limit: a local cap
    assert rec["kernel"] == "local/fk-train" and rec["resource"] == "LOCAL" and rec["hours"] == 20
    vdir = lws / ".kaggle-gm" / "local" / "fk-train" / "v1"
    comp = vdir / "input" / "competitions" / "demo-comp"
    assert (comp / "train.csv").is_file() and not (comp / "folds.csv").exists()  # the runner writes the split
    # no Kaggle quota: the GPU arithmetic never sees it, the run log does
    assert G.active_runs(lws, "GPU") == [] and G.in_flight_hours(lws) == 0
    assert _wait(lws, rec["kernel"]) == "complete", "\n".join(LR.log_tail(G.latest_run(lws, rec["kernel"])))
    assert G.kernel_status(rec["kernel"])[0] == "complete"
    rep = json.loads((vdir / "working" / "kg_run.json").read_text())
    assert rep["budget_h"] == 20 and rep["jobs"][0]["folds_done"] == [0, 1, 2, 3]

    res = G.collect(lws, rec["kernel"])
    assert res["status"] == "complete" and len(res["imported"]) == 1
    exp = Ledger(lws).records()[-1]
    assert exp["folds_hash"] == Ledger(lws)._folds_hash(None)  # same frozen split as the workspace
    assert (lws / "artifacts" / "fk" / "curve_fold0.json").is_file()
    run = G.runs(lws)[-1]
    assert run["status"] == "collected" and run["exp_ids"] == [exp["id"]]
    assert "local/" in (lws / ".kaggle-gm" / ".gitignore").read_text()


def test_a_second_run_waits_for_the_first_and_cancel_stops_it(lws):
    slow = f'"{sys.executable}" -c "import time; time.sleep(120)"'
    kdir = G.build_kernel(lws, "slow", slow, hours=1, user="tester")
    rec = G.push_local(lws, kdir)
    assert LR.status(G.latest_run(lws, rec["kernel"])) == "running"
    other = G.build_kernel(lws, "fk", _cmd("fk"), hours=1, user="tester")
    with pytest.raises(SystemExit, match="share its memory"):
        G.push_local(lws, other)
    with pytest.raises(SystemExit, match="is running"):
        G.collect(lws, rec["kernel"])
    msg = LR.cancel(G.latest_run(lws, rec["kernel"]))
    assert msg.startswith("stopped local/slow-train v1")
    G.note_status(lws, rec["kernel"], "cancelled")
    assert _wait(lws, rec["kernel"], 30) == "cancelled"


def test_an_input_with_no_local_copy_is_refused_until_it_is_there(lws):
    kdir = G.build_kernel(lws, "fk", _cmd("fk"), hours=1, user="tester", datasets=["tester/extra"],
                          input_layout="nested")
    with pytest.raises(SystemExit, match="kaggle datasets download -d tester/extra -p data/ext/extra"):
        G.push_local(lws, kdir)
    assert not (lws / ".kaggle-gm" / "local" / "fk-train" / "v1").exists()
    (lws / "data" / "ext" / "extra").mkdir(parents=True)
    (lws / "data" / "ext" / "extra" / "more.csv").write_text("a\n1\n")
    cfg = LR.embedded_config(kdir / json.loads((kdir / "kernel-metadata.json").read_text())["code_file"])
    said, missing = LR.prepare_inputs(lws, cfg, lws / "inp", [], ".kaggle-gm", G.kernel_output_dir, "demo-comp")
    assert missing == {}
    assert (lws / "inp" / "datasets" / "tester" / "extra" / "more.csv").is_file()
    assert not (lws / "inp" / "competitions" / "demo-comp" / "ext").exists()  # linked as the dataset, not twice


def test_resume_from_an_earlier_local_run(lws):
    kdir = G.build_kernel(lws, "fk", _cmd("fk"), hours=1, user="tester")
    first = G.push_local(lws, kdir)
    assert _wait(lws, first["kernel"]) == "complete"
    again = G.build_kernel(lws, "fk", _cmd("fk"), hours=1, user="tester", resume_from=["local/fk-train"])
    rec = G.push_local(lws, again)
    assert rec["version"] == 2 and _wait(lws, rec["kernel"]) == "complete"
    log = "\n".join(LR.log_tail(rec, 400))
    assert "fold 0 already done (resumed)" in log


def test_without_a_gpu_or_a_build_output_it_refuses(lws, monkeypatch, tmp_path):
    kdir = G.build_kernel(lws, "fk", _cmd("fk"), hours=1, user="tester")
    monkeypatch.setattr(LR.shutil, "which", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="no NVIDIA GPU"):
        G.push_local(lws, kdir)
    (tmp_path / "nb").mkdir()
    with pytest.raises(SystemExit, match="not a `gpu build` output"):
        G.push_local(lws, tmp_path / "nb")


def test_pid_alive():
    import os
    assert LR.pid_alive(os.getpid()) and not LR.pid_alive(None) and not LR.pid_alive(0)


def test_left_out_of_the_kaggle_bundle_and_gpu_imports_without_it(lws, plugin_root):
    import base64
    import io
    import re
    import subprocess
    import zipfile

    kdir = G.build_kernel(lws, "fk", _cmd("fk"), hours=1, user="tester")
    script = (kdir / json.loads((kdir / "kernel-metadata.json").read_text())["code_file"]).read_text()
    payload = base64.b64decode(re.search(r'PAYLOAD_B64 = "([^"]+)"', script).group(1))
    names = zipfile.ZipFile(io.BytesIO(payload)).namelist()
    assert "kgkit/gpu.py" in names and "kgkit/localrun.py" not in names
    code = ("import sys\nclass Block:\n    def find_spec(self, name, *a):\n"
            "        if name == 'kgkit.localrun': raise ImportError(name)\n"
            "sys.meta_path.insert(0, Block())\nfrom kgkit import gpu\nprint(gpu.LR, gpu.is_local('local/x'))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(plugin_root), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.split() == ["None", "True"], r.stderr
