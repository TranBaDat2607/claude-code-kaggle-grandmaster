"""#114: a kernel deleted on Kaggle (where kgkit did not see it: the website, another terminal) and pushed again under
its slug starts again at version 1. The run log took the re-push for the push it already had, so the run that
actually ran had no row of its own and was charged the old push's few queued minutes (7.67 h logged as 0.12 h).
A push is the same run only while that entry is still in flight and was pushed moments before; a version Kaggle
numbers at or below one already logged is a new kernel's. `collect` charges a run Kaggle reports finished its
session time, and says why when it cannot."""

import datetime as dt
import json
import subprocess

import pytest

from kgkit import gpu as G
from kgkit import state as S

K = "me/hand"


@pytest.fixture
def ws(tmp_path, monkeypatch):
    S.CompetitionState(slug="demo", metric="auc", deadline=(dt.date.today() + dt.timedelta(days=20)).isoformat()
                       ).save(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def clock(monkeypatch):
    now = [dt.datetime(2026, 11, 14, 15, 49, tzinfo=dt.timezone.utc)]
    monkeypatch.setattr(G, "_now", lambda: now[0])

    def tick(**kw):
        now[0] += dt.timedelta(**kw)
        return now[0]
    tick.now = lambda: now[0]
    return tick


def _push(ws, version=1, **kw):
    rec = {"kernel": K, "version": version, "name": "hand", "accelerator": "NvidiaTeslaT4", "resource": "GPU",
           "hours": 9.5, "pushed_at": G._iso(G._now()), "status": "pushed", **kw}
    return G.record_push(ws, rec)


def _rows(ws):
    return [(r["version"], r["status"], bool(r.get("deleted_at"))) for r in G.runs(ws) if r["kernel"] == K]


# ----------------------------------------------------------------------------- 1. which push is which run
def test_the_issues_run_deleted_unseen_released_and_pushed_again_is_charged_its_session(ws, clock):
    _push(ws)
    G.note_status(ws, K, "queued")
    clock(minutes=7)
    G.release(ws, K)  # deleted on the website: kgkit saw only the release
    _push(ws)  # Kaggle answers "version 1" again
    assert _rows(ws) == [(1, "released", True), (1, "pushed", False)]
    G.note_status(ws, K, "running")
    clock(hours=7.67)
    G.note_status(ws, K, "complete")
    assert G.used_hours(G.latest_run(ws, K)) == pytest.approx(7.67, abs=0.01)
    assert G.used_hours(G.runs(ws)[0]) in (None, 0.0)  # the first push never ran


def test_a_push_logged_twice_moments_apart_is_one_run(ws, clock):
    """The same push seen twice (a hook re-run on the same command) is logged once."""
    _push(ws, version=2)
    clock(seconds=20)
    _push(ws, version=2)
    assert _rows(ws) == [(2, "pushed", False)]


def test_a_stale_entry_still_in_flight_is_not_the_new_push(ws, clock):
    """Deleted on the website while queued, pushed again half an hour later: the queued entry spent nothing."""
    _push(ws)
    G.note_status(ws, K, "queued")
    clock(minutes=30)
    _push(ws)
    assert _rows(ws) == [(1, "deleted", True), (1, "pushed", False)]
    assert G.runs(ws)[0]["gpu_hours"] == 0.0


def test_a_version_below_one_logged_is_a_new_kernel(ws, clock):
    _push(ws, version=3)
    G.note_status(ws, K, "running")
    clock(hours=1)
    G.note_status(ws, K, "complete")
    clock(hours=1)
    _push(ws, version=1)
    assert _rows(ws) == [(3, "complete", True), (1, "pushed", False)]
    assert G.runs(ws)[0]["gpu_hours"] == pytest.approx(1.0, abs=0.01)  # its charge stands
    clock(minutes=5)
    _push(ws, version=2)  # the new kernel's next version: an ordinary push
    assert _rows(ws) == [(3, "complete", True), (1, "superseded", False), (2, "pushed", False)]


# ----------------------------------------------------------------------------- 2. collect charges what Kaggle reports
def _session(tmp_path, seconds):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    rows = [{"stream_name": "stdout", "time": 1.0, "data": "start\n"}]
    if seconds is not None:
        rows.append({"stream_name": "stdout", "time": float(seconds), "data": "done\n"})
    (out / "hand.log").write_text(json.dumps(rows), encoding="utf-8")
    (out / "preds.npy").write_text("x")
    return out


def test_collect_charges_a_finished_run_whose_entry_was_closed_by_hand(ws, clock, tmp_path, capsys):
    """The re-push never reached the log (pushed from the website): the released entry is the kernel's latest, and
    Kaggle reports the kernel complete with a 7.67 h session log. The quota was spent: it is charged."""
    _push(ws)
    G.note_status(ws, K, "queued")
    clock(minutes=7)
    G.release(ws, K)
    clock(hours=7.7)
    G._collect_without_report(ws, K, _session(tmp_path, 27606), "complete", staged=False, weights=False)
    out = capsys.readouterr().out
    run = G.latest_run(ws, K)
    assert run["status"] == "complete" and run["gpu_hours"] == pytest.approx(7.668, abs=0.01)
    assert "7.67 GPU-hours (session wall time)" in out and "shows no time" not in out


def test_collect_says_why_no_hours_are_charged(ws, clock, tmp_path, capsys):
    _push(ws)
    G.note_status(ws, K, "running")
    clock(hours=2)
    # Kaggle could not be asked, and the run is still in flight in the log: its session may not be over
    G._collect_without_report(ws, K, _session(tmp_path, 7200), "unknown", staged=False, weights=False)
    out = capsys.readouterr().out
    assert "no hours accounted yet" in out and "not reported it finished" in out and "shows no time" not in out
    capsys.readouterr()
    # finished, but neither the log nor the session log tells how long it ran (a TPU job never seen running)
    G.log_run(ws, {"kernel": "me/tpu", "version": 1, "accelerator": "TpuV5E8", "resource": "TPU", "hours": 9,
                   "pushed_at": G._iso(G._now()), "status": "complete"})
    out_dir = tmp_path / "tpu"
    out_dir.mkdir()
    G._collect_without_report(ws, "me/tpu", out_dir, "complete", staged=False, weights=False)
    out = capsys.readouterr().out
    assert "no hours accounted" in out and "its session log shows no time" in out


def test_gpu_wait_collect_of_the_issues_run_end_to_end(ws, clock, tmp_path, monkeypatch, capsys):
    """push, stuck in the queue, deleted unseen + released, pushed again by hand (logged by the hook), run 7.67 h,
    `gpu wait --collect`: the run is charged its session."""
    _push(ws)
    G.note_status(ws, K, "queued")
    clock(minutes=7)
    G.release(ws, K)
    _push(ws)
    statuses = iter(["running"] * 10 + ["complete"])
    monkeypatch.setattr(G, "kernel_status", lambda k: (next(statuses, "complete"), ""))
    monkeypatch.setattr(G.time, "sleep", lambda s: clock(seconds=max(s, 2763)))  # 10 polls ~ 7.67 h
    monkeypatch.setattr(G, "run_kaggle", lambda args, timeout=600, env=None: subprocess.CompletedProcess(
        [], 1, "", "not faked"))
    assert G.wait(ws, K, interval=60) == "complete"
    G._collect_without_report(ws, K, _session(tmp_path, 27606), "complete", staged=False, weights=False)
    assert G.latest_run(ws, K)["gpu_hours"] == pytest.approx(7.67, abs=0.05)
    assert "GPU-hours (session wall time)" in capsys.readouterr().out
