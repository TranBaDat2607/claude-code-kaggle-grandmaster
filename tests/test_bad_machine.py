"""#113: Kaggle sometimes starts a 2x T4 push on a machine with one GPU (or none). The session checks the machine it
got against the GPUs its shape promises and stops within seconds on a mismatch (the kgkit runner, and the monitor
cell `gpu push` and the push hook add to hand-written kernels); `gpu wait` reads a running GPU kernel's log and,
like a stopped run's report and the /kaggle-monitor pane, names a bad machine and the remedy (repush)."""

import datetime as dt
import json
import os
import subprocess
import sys

import pytest

from kgkit import gpu as G
from kgkit import monitor as M
from kgkit import state as S

CELL = M.CELL


def _cp(stdout="", rc=0, stderr=""):
    return subprocess.CompletedProcess([], rc, stdout, stderr)


def _log(*lines):
    """A session log as `kaggle kernels logs` prints it."""
    return json.dumps([{"stream_name": "stdout", "time": float(i), "data": l + "\n"} for i, l in enumerate(lines)])


def _reading(n_gpus, want=None):
    snap = {"v": 1, "t": 5, "t0": 1763136000, "cores": 4.0,
            "gpus": [[i, 0, 300, 15360, "Tesla T4"] for i in range(n_gpus)]}
    if want is not None:
        snap["want"] = want
    return "[kgmon] " + json.dumps(snap, separators=(",", ":"))


@pytest.fixture
def ws(tmp_path, monkeypatch):
    S.CompetitionState(slug="demo", metric="auc", deadline=(dt.date.today() + dt.timedelta(days=20)).isoformat()
                       ).save(tmp_path)
    q = {"used_h": 2.0, "remaining_h": 28.0, "total_h": 30.0, "refresh_at": "2099-01-03T00:00:00Z",
         "fetched_at": G._iso(G._now())}
    G.quota_path(tmp_path).write_text(json.dumps(q))
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ----------------------------------------------------------------------------- the rule
def test_expected_gpus_per_shape():
    assert G.expected_gpus("NvidiaTeslaT4") == 2 and G.expected_gpus("NvidiaTeslaP100") == 1
    assert G.expected_gpus("TpuV5E8") is None and G.expected_gpus("none") is None and G.expected_gpus(None) is None


def test_machine_problem_reads_the_log():
    t4 = "NvidiaTeslaT4"
    assert G.machine_problem(_log(_reading(2, want=2), "epoch 1"), t4) is None
    assert G.machine_problem(_log(_reading(1, want=1)), "NvidiaTeslaP100") is None
    # one GPU on a 2x T4 push: from the reading's own `want`, or from the shape the run was pushed with
    assert "1 GPU" in G.machine_problem(_log(_reading(1, want=2)), None)
    assert "2x T4" in G.machine_problem(_log(_reading(1)), t4)
    # the line the runner and the cell print when they stop on a mismatch
    assert "1 GPU" in G.machine_problem(_log("[kg    0.1m] KG_BAD_MACHINE: NvidiaTeslaT4 promises 2 GPUs, "
                                             "nvidia-smi lists 1"), t4)
    # a process that found no GPU at all (a hand-written kernel without the monitor)
    assert "No CUDA GPUs are available" in G.machine_problem(
        _log("RuntimeError: No CUDA GPUs are available"), t4)
    # a reading without GPUs (a CPU kernel, an opted-out cell) and a raw text log
    assert G.machine_problem(_log('[kgmon] {"v":1,"t":5,"cores":4}'), "none") is None
    assert G.machine_problem("not json\nNo CUDA GPUs are available\n", t4) is not None
    assert G.machine_problem("", t4) is None


# ----------------------------------------------------------------------------- A: the monitor cell stops too
def _run_cell(path, tmp_path, **env):
    e = dict(os.environ, KAGGLE_WORKING=str(tmp_path), KG_MONITOR_S="0", **env)
    return subprocess.run([sys.executable, str(path)], env=e, capture_output=True, text=True, timeout=60)


def test_monitor_cell_stops_on_fewer_gpus_than_expected(tmp_path):
    assert CELL.read_text(encoding="utf-8").isascii()
    r = _run_cell(CELL, tmp_path, KG_FAKE_GPUS="1", KG_EXPECT_GPUS="2")
    assert r.returncode != 0
    assert "KG_BAD_MACHINE: this machine shape promises 2 GPUs, nvidia-smi lists 1" in r.stdout + r.stderr
    reading = [l for l in r.stdout.splitlines() if l.startswith("[kgmon] ")]
    assert reading and json.loads(reading[0][8:])["want"] == 2  # the evidence reaches the log first
    for env in ({"KG_FAKE_GPUS": "2", "KG_EXPECT_GPUS": "2"}, {"KG_FAKE_GPUS": "0"},
                {"KG_FAKE_GPUS": "1", "KG_EXPECT_GPUS": "0"}):  # 0: the check is off
        r = _run_cell(CELL, tmp_path, **env)
        assert r.returncode == 0 and "KG_BAD_MACHINE" not in r.stdout, (env, r.stdout, r.stderr)


# ----------------------------------------------------------------------------- B: gpu push adds the monitor
def _kernel_dir(ws, shape="NvidiaTeslaT4", code="print('hi')\n", code_file="run.py"):
    d = ws / "kern"
    d.mkdir(exist_ok=True)
    (d / "kernel-metadata.json").write_text(json.dumps({"id": "me/x-train", "enable_gpu": True,
                                                         "machine_shape": shape, "code_file": code_file}))
    if code is not None:
        (d / code_file).write_text(code)
    return d


def _fake_push(monkeypatch):
    def run(args, timeout=600, env=None):
        if args[:2] == ["kernels", "push"]:
            return _cp("Kernel version 1 successfully pushed.  Please check progress at https://www.kaggle.com/code/me/x")
        if args[:2] == ["kernels", "list"]:
            return _cp("[]")
        return _cp("", rc=1, stderr="not faked")
    monkeypatch.setattr(G, "run_kaggle", run)


def test_gpu_push_adds_the_monitor_with_the_gpus_the_shape_promises(ws, monkeypatch, capsys, tmp_path):
    _fake_push(monkeypatch)
    d = _kernel_dir(ws)
    G.push(ws, d, hours=0.75)
    code = (d / "run.py").read_text(encoding="utf-8")
    assert M.BEGIN in code and "KGMON_EXPECT_GPUS = 2" in code and code.endswith("print('hi')\n")
    assert "added the hardware monitor" in capsys.readouterr().out
    r = _run_cell(d / "run.py", tmp_path, KG_FAKE_GPUS="1")
    assert r.returncode != 0 and "KG_BAD_MACHINE" in r.stdout and "hi" not in r.stdout.splitlines()
    # pushed again for the P100: the block is brought up to date, not added twice
    _kernel_dir(ws, shape="NvidiaTeslaP100", code=None)
    G.push(ws, d, hours=0.75)
    code = (d / "run.py").read_text(encoding="utf-8")
    assert code.count(M.BEGIN) == 1 and "KGMON_EXPECT_GPUS = 1" in code
    assert "updated the hardware monitor" in capsys.readouterr().out
    before = code
    G.push(ws, d, hours=0.75)  # up to date: untouched, nothing said
    assert (d / "run.py").read_text(encoding="utf-8") == before
    assert "hardware monitor" not in capsys.readouterr().out


def test_the_check_never_stops_a_code_competition_scoring_run(tmp_path):
    """#121: Kaggle re-runs a submitted notebook on the hidden test; a slower machine still gives a valid submission,
    an exception fails it."""
    r = _run_cell(CELL, tmp_path, KG_FAKE_GPUS="1", KG_EXPECT_GPUS="2", KAGGLE_IS_COMPETITION_RERUN="True")
    assert r.returncode == 0 and "KG_BAD_MACHINE" not in r.stdout, (r.stdout, r.stderr)


@pytest.mark.parametrize("internet", [False, "false"])
def test_a_submission_kernel_gets_the_monitor_without_the_check(ws, monkeypatch, capsys, tmp_path, internet):
    _fake_push(monkeypatch)
    d = _kernel_dir(ws)
    meta = json.loads((d / "kernel-metadata.json").read_text())
    meta.update(competition_sources=["demo"], enable_internet=internet)
    (d / "kernel-metadata.json").write_text(json.dumps(meta))
    G.push(ws, d, hours=0.75)
    code = (d / "run.py").read_text(encoding="utf-8")
    assert M.BEGIN in code and "KGMON_EXPECT_GPUS = 0" in code
    assert "looks like a code-competition submission" in capsys.readouterr().out
    r = _run_cell(d / "run.py", tmp_path, KG_FAKE_GPUS="1")
    assert r.returncode == 0 and "KG_BAD_MACHINE" not in r.stdout and "hi" in r.stdout.splitlines()


def test_a_training_kernel_that_attaches_the_competition_keeps_the_check(ws, monkeypatch):
    _fake_push(monkeypatch)
    d = _kernel_dir(ws)
    meta = json.loads((d / "kernel-metadata.json").read_text())
    meta.update(competition_sources=["demo"], enable_internet=True)
    (d / "kernel-metadata.json").write_text(json.dumps(meta))
    G.push(ws, d, hours=0.75)
    assert "KGMON_EXPECT_GPUS = 2" in (d / "run.py").read_text(encoding="utf-8")
    assert not M.is_submission_kernel({"competition_sources": [], "enable_internet": False})


def test_gpu_push_leaves_opted_out_and_built_kernels_alone(ws, monkeypatch):
    _fake_push(monkeypatch)
    d = _kernel_dir(ws, code="# kgmon: off\nprint('hi')\n")
    G.push(ws, d, hours=0.75)
    assert (d / "run.py").read_text(encoding="utf-8") == "# kgmon: off\nprint('hi')\n"


def test_the_cell_lives_in_the_package_and_the_template_is_gone(plugin_root):
    """kgkit fills the cell in (the GPUs the shape promises), so it ships inside the package: `kgkit vendor` copies
    have it too."""
    assert CELL.parent.name == "kgkit" and CELL.is_file()
    assert not (plugin_root / "templates" / "kaggle_monitor_cell.py").exists()
    assert "KGMON_EXPECT_GPUS = None" in CELL.read_text(encoding="utf-8")
    assert {"monitor.py", "kaggle_monitor_cell.py"} <= G.BUNDLE_KGKIT_LOCAL_ONLY


# ----------------------------------------------------------------------------- C: gpu wait and the stop report
class Clock:
    def __init__(self, monkeypatch):
        self.t = G._now()
        monkeypatch.setattr(G, "_now", lambda: self.t)
        monkeypatch.setattr(G.time, "sleep", self.sleep)

    def sleep(self, s):
        self.t += dt.timedelta(seconds=s)


def _wait_setup(ws, monkeypatch, log, statuses):
    G.record_push(ws, {"kernel": "me/x-train", "version": 1, "name": "x", "accelerator": "NvidiaTeslaT4",
                       "resource": "GPU", "hours": 9.5, "pushed_at": G._iso(G._now()), "status": "pushed",
                       "dir": "kern"})
    it, last = iter(statuses), [None]

    def status(kernel):
        last[0] = next(it, last[0])
        return last[0], ""
    monkeypatch.setattr(G, "kernel_status", status)
    calls = []

    def run(args, timeout=600, env=None):
        calls.append(args[:2])
        if args[:2] == ["kernels", "logs"]:
            return _cp(log)
        return _cp("", rc=1, stderr="not faked")
    monkeypatch.setattr(G, "run_kaggle", run)
    monkeypatch.setattr(G, "session_output", lambda kernel, timeout=0: (None, False, log, None))
    return calls


def test_wait_names_a_bad_machine_while_the_kernel_runs(ws, monkeypatch, capsys):
    Clock(monkeypatch)
    log = _log(_reading(1, want=2), "[gpu1] RuntimeError: No CUDA GPUs are available", "epoch 1/10")
    calls = _wait_setup(ws, monkeypatch, log, ["running"] * 6 + ["complete"])
    assert G.wait(ws, "me/x-train", interval=60) == "complete"
    out = capsys.readouterr().out
    assert out.count("bad machine") == 1  # said once, not at every poll
    assert "1 GPU" in out and "gpu repush me/x-train --yes" in out and "ask the user" in out
    assert calls.count(["kernels", "logs"]) == 1  # a verdict stops the log reads


def test_wait_stops_reading_the_log_once_the_machine_looks_right(ws, monkeypatch, capsys):
    Clock(monkeypatch)
    calls = _wait_setup(ws, monkeypatch, _log(_reading(2, want=2)), ["running"] * 10 + ["complete"])
    G.wait(ws, "me/x-train", interval=60)
    assert "bad machine" not in capsys.readouterr().out and calls.count(["kernels", "logs"]) == 1


def test_wait_gives_up_on_a_log_without_a_verdict(ws, monkeypatch, capsys):
    """A kernel without readings (opted out): the log is read every few minutes in the first
    MACHINE_CHECK_MIN minutes of the run, not for the whole run."""
    Clock(monkeypatch)
    calls = _wait_setup(ws, monkeypatch, _log("epoch 1/10"), ["running"] * 60 + ["complete"])
    G.wait(ws, "me/x-train", interval=60)
    n = calls.count(["kernels", "logs"])
    assert 2 <= n <= G.MACHINE_CHECK_MIN * 60 / G.MACHINE_CHECK_EVERY_S + 1


def test_a_run_stopped_by_a_bad_machine_says_so(ws, monkeypatch, capsys):
    Clock(monkeypatch)
    log = _log(_reading(1, want=2), "[kg    0.1m] KG_BAD_MACHINE: NvidiaTeslaT4 promises 2 GPUs, nvidia-smi lists 1")
    _wait_setup(ws, monkeypatch, log, ["error"])
    G.wait(ws, "me/x-train", interval=60)
    out = capsys.readouterr().out
    assert "bad machine" in out and "gpu repush me/x-train --yes" in out


# ----------------------------------------------------------------------------- E: the skill says so
def test_kaggle_gpu_skill_tells_of_bad_machines(plugin_root):
    text = (plugin_root / "skills" / "kaggle-gpu" / "SKILL.md").read_text(encoding="utf-8")
    assert "KG_BAD_MACHINE" in text and "gpu repush" in text
