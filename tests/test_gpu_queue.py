"""#110: a GPU job left in Kaggle's queue. GPU sessions normally start within a minute or two, so `gpu wait` warns
once a GPU job has been queued for longer than GPU_QUEUE_WARN_MIN and says what helps; `gpu status` shows the queue
time; `gpu repush` deletes and pushes again (only with --yes, as deleting is irreversible). A kernel deleted and
pushed again under the same slug starts again at version 1: the log keeps that run as a new row."""

import datetime as dt
import json
import subprocess
import sys

import pytest

from kgkit import gpu as G
from kgkit import state as S


def _cp(stdout="", rc=0, stderr=""):
    return subprocess.CompletedProcess([], rc, stdout, stderr)


@pytest.fixture
def ws(tmp_path, monkeypatch):
    S.CompetitionState(slug="demo", metric="auc", deadline=(dt.date.today() + dt.timedelta(days=20)).isoformat()
                       ).save(tmp_path)
    q = {"used_h": 2.0, "remaining_h": 28.0, "total_h": 30.0, "refresh_at": "2099-01-03T00:00:00Z",
         "fetched_at": G._iso(G._now())}
    G.quota_path(tmp_path).write_text(json.dumps(q))
    monkeypatch.chdir(tmp_path)
    return tmp_path


class Clock:
    """G._now() that a patched sleep moves forward."""

    def __init__(self, monkeypatch):
        self.t = G._now()
        monkeypatch.setattr(G, "_now", lambda: self.t)
        monkeypatch.setattr(G.time, "sleep", self.sleep)

    def sleep(self, s):
        self.t += dt.timedelta(seconds=s)


def _push(ws, kernel="me/x-train", acc="NvidiaTeslaT4", ago_min=0.0, now=None, **kw):
    now = now or G._now()
    rec = {"kernel": kernel, "version": 1, "name": "x", "accelerator": acc, "resource": G.resource_of(acc),
           "hours": 0.75, "pushed_at": G._iso(now - dt.timedelta(minutes=ago_min)), "status": "pushed", **kw}
    return G.record_push(ws, rec)


def _statuses(monkeypatch, seq):
    it = iter(seq)
    last = [None]

    def status(kernel):
        last[0] = next(it, last[0])
        return last[0], ""
    monkeypatch.setattr(G, "kernel_status", status)


# ----------------------------------------------------------------------------- gpu wait
def test_wait_warns_once_when_a_gpu_job_stays_queued(ws, monkeypatch, capsys):
    clock = Clock(monkeypatch)
    _push(ws, ago_min=5, now=clock.t)
    _statuses(monkeypatch, ["queued"] * 8 + ["running", "complete"])  # 90 s polls: queued from 5 to ~16 min
    assert G.wait(ws, "me/x-train", interval=90) == "complete"
    out = capsys.readouterr().out
    assert out.count("queued for") == 1 and f"more than {G.GPU_QUEUE_WARN_MIN} min" in out
    assert "gpu repush me/x-train" in out and "ask the user" in out and "P100" in out


def test_wait_quiet_for_a_short_gpu_queue_and_any_tpu_queue(ws, monkeypatch, capsys):
    clock = Clock(monkeypatch)
    _push(ws, ago_min=1, now=clock.t)
    _statuses(monkeypatch, ["queued", "queued", "running", "complete"])
    G.wait(ws, "me/x-train", interval=90)
    _push(ws, kernel="me/t-train", acc="TpuV5E8", ago_min=180, now=clock.t)
    _statuses(monkeypatch, ["queued", "queued", "running", "complete"])
    G.wait(ws, "me/t-train", interval=90)
    assert "queued for" not in capsys.readouterr().out  # TPU jobs queue for hours: that is normal


# ----------------------------------------------------------------------------- gpu status
def test_status_shows_how_long_a_job_has_been_queued(ws, monkeypatch, capsys):
    from kgkit import cli

    _push(ws, ago_min=14)
    monkeypatch.setattr(G, "kernel_status", lambda k: ("queued", ""))
    monkeypatch.setattr(G, "confirm_pending", lambda root, kernels=None: [])
    cli.main(["gpu", "status"])
    out = capsys.readouterr().out
    assert "me/x-train: queued (for 14 min" in out and "gpu repush me/x-train" in out


# ----------------------------------------------------------------------------- gpu repush
def _kernel_dir(ws, kernel="me/x-train"):
    d = ws / "kern"
    d.mkdir()
    (d / "kernel-metadata.json").write_text(json.dumps({"id": kernel, "enable_gpu": True,
                                                         "machine_shape": "NvidiaTeslaT4", "code_file": "run.py"}))
    (d / "run.py").write_text("print('hi')\n")
    return d


def _fake_kaggle(monkeypatch, status="queued", version=1):
    calls = []

    def run(args, timeout=600, env=None):
        calls.append(args[:2])
        if args[:2] == ["kernels", "status"]:
            return _cp(f'{args[2]} has status "KernelWorkerStatus.{status.upper()}"')
        if args[:2] == ["kernels", "delete"]:
            return _cp(f"Kernel {args[2]} deleted successfully")
        if args[:2] == ["kernels", "list"]:
            return _cp("[]")
        if args[:2] == ["kernels", "push"]:
            return _cp(f"Kernel version {version} successfully pushed.  Please check progress at "
                       f"https://www.kaggle.com/code/me/x-train")
        return _cp("", rc=1, stderr="not faked")
    monkeypatch.setattr(G, "run_kaggle", run)
    return calls


def test_repush_without_yes_deletes_nothing(ws, monkeypatch, capsys):
    from kgkit import cli

    d = _kernel_dir(ws)
    _push(ws, ago_min=20, dir=d.relative_to(ws).as_posix())
    calls = _fake_kaggle(monkeypatch)
    with pytest.raises(SystemExit) as e:
        cli.main(["gpu", "repush", "me/x-train"])
    assert e.value.code not in (0, None)
    out = capsys.readouterr().out
    assert "kaggle kernels delete me/x-train" in out and "ask the user" in out and "--yes" in out
    assert ["kernels", "delete"] not in calls and ["kernels", "push"] not in calls
    assert G.latest_run(ws, "me/x-train")["status"] == "pushed"


def test_repush_deletes_and_pushes_the_same_folder_again(ws, monkeypatch, capsys):
    from kgkit import cli

    d = _kernel_dir(ws)
    calls = _fake_kaggle(monkeypatch)
    first = G.push(ws, d, hours=0.75)  # records the folder it pushed
    assert first["dir"] == "kern" and first["version"] == 1
    cli.main(["gpu", "repush", "me/x-train", "--yes"])
    assert [c for c in calls if c[1] in ("delete", "push")][-2:] == [["kernels", "delete"], ["kernels", "push"]]
    rows = [r for r in G.runs(ws) if r["kernel"] == "me/x-train"]
    assert [(r["version"], r["status"]) for r in rows] == [(1, "deleted"), (1, "pushed")]  # a new row for the re-push
    assert rows[-1]["hours"] == 0.75 and rows[-1]["dir"] == "kern"
    assert "pushed me/x-train v1" in capsys.readouterr().out


def test_repush_refuses_a_finished_kernel_or_an_unknown_folder(ws, monkeypatch):
    from kgkit import cli

    _push(ws, ago_min=20)  # logged by the hook: no folder recorded
    _fake_kaggle(monkeypatch)
    with pytest.raises(SystemExit, match="pass the kernel folder"):
        cli.main(["gpu", "repush", "me/x-train", "--yes"])
    _fake_kaggle(monkeypatch, status="complete")
    with pytest.raises(SystemExit, match="complete: nothing to repush"):
        cli.main(["gpu", "repush", "me/x-train", str(_kernel_dir(ws)), "--yes"])


# ----------------------------------------------------------------------------- the run log after a delete
def test_a_kernel_deleted_and_pushed_again_keeps_both_runs(ws):
    _push(ws)
    assert G.record_push(ws, dict(G.runs(ws)[0])) is not None and len(G.runs(ws)) == 1  # same version: logged once
    assert G.note_deleted(ws, "me/x-train") == 1
    assert G.runs(ws)[0]["status"] == "deleted"
    _push(ws)  # Kaggle numbers the new kernel's versions from 1 again
    assert [(r["version"], r["status"]) for r in G.runs(ws)] == [(1, "deleted"), (1, "pushed")]


def test_post_tool_hook_notes_a_kernel_deleted_by_hand(ws, plugin_root):
    _push(ws)
    payload = {"hook_event_name": "PostToolUse", "tool_name": "Bash", "cwd": str(ws),
               "tool_input": {"command": "kaggle kernels delete me/x-train -y"},
               "tool_response": {"stdout": "Kernel me/x-train deleted successfully\n"}}
    r = subprocess.run([sys.executable, str(plugin_root / "hooks" / "post_tool.py")], input=json.dumps(payload),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    ctx = json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "me/x-train" in ctx and "deleted" in ctx
    assert G.runs(ws)[0]["status"] == "deleted"


def test_post_tool_hook_logs_delete_then_push_in_one_command(ws, plugin_root):
    d = _kernel_dir(ws)
    _push(ws)
    cmd = f"kaggle kernels delete me/x-train -y && kaggle kernels push -p {d.as_posix()} -t 2700"
    out = ("Kernel me/x-train deleted successfully\nKernel version 1 successfully pushed.  Please check progress at "
           "https://www.kaggle.com/code/me/x-train\n")
    payload = {"hook_event_name": "PostToolUse", "tool_name": "Bash", "cwd": str(ws), "tool_input": {"command": cmd},
               "tool_response": {"stdout": out}}
    r = subprocess.run([sys.executable, str(plugin_root / "hooks" / "post_tool.py")], input=json.dumps(payload),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert [(x["version"], x["status"]) for x in G.runs(ws)] == [(1, "deleted"), (1, "pushed")]
