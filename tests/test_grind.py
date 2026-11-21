"""#125: a /kg-grind loop outlives the conversation that runs it. Its budget, progress and the runs it started are
kept in .kaggle-gm/grind.json, so a compacted conversation or a new session continues it instead of starting over;
one grind at a time per workspace; time spent counts only while a session drives it; on resume every run it
started is checked against the ledger, its artifacts (is the process still alive?) and the Kaggle run log."""

import json
import os
import socket
import subprocess
import sys
import time

import pytest

from kgkit import grind as GR
from kgkit import state as S
from kgkit.experiment import Ledger

T0 = 1_800_000_000.0  # a fixed clock: every call gets `now`


def _ws(tmp_path):
    S.CompetitionState(slug="demo", metric="auc").save(tmp_path)
    return tmp_path


def test_start_records_the_budget_and_one_grind_at_a_time(tmp_path):
    ws = _ws(tmp_path)
    g = GR.start(ws, "sess-a", hours=4, focus="features", now=T0)
    assert g["budget"] == {"hours": 4.0} and g["owner"] == "sess-a" and g["focus"] == "features"
    with pytest.raises(GR.GrindError, match="already in progress"):
        GR.start(ws, "sess-b", experiments=10, now=T0 + 60)  # a second loop would race the first
    with pytest.raises(GR.GrindError, match="already in progress"):
        GR.start(ws, "sess-a", experiments=10, now=T0 + 60)  # the same session too: resume, or stop it first
    assert GR.load(ws)["budget"] == {"hours": 4.0}  # untouched


def test_time_spent_counts_only_while_a_session_drives_it(tmp_path):
    ws = _ws(tmp_path)
    GR.start(ws, "sess-a", hours=4, now=T0)
    GR.status(ws, "sess-a", now=T0 + 3600)  # a heartbeat an hour in
    # the session ends; a new one resumes the next morning: the night does not count
    GR.start(ws, "sess-b", resume=True, now=T0 + 15 * 3600)
    s = GR.status(ws, "sess-b", now=T0 + 15 * 3600 + 1800)
    assert s["active_hours"] == pytest.approx(1.5) and s["hours_left"] == pytest.approx(2.5)
    assert s["owner"] == "sess-b" and len(s["segments"]) == 2


def test_a_live_owner_is_not_taken_over_unless_asked(tmp_path):
    ws = _ws(tmp_path)
    GR.start(ws, "sess-a", hours=4, now=T0)
    GR.status(ws, "sess-a", now=T0 + 600)
    with pytest.raises(GR.GrindError, match="driven by another session"):
        GR.start(ws, "sess-b", resume=True, now=T0 + 900)  # sess-a was active 5 min ago
    g = GR.start(ws, "sess-b", resume=True, takeover=True, now=T0 + 900)
    assert g["owner"] == "sess-b"
    # a heartbeat from a session that does not own it changes nothing
    GR.status(ws, "sess-a", now=T0 + 1200)
    assert GR.load(ws)["segments"][-1]["last"] == T0 + 900


def test_experiments_done_are_counted_from_the_ledger(tmp_path, monkeypatch):
    ws = _ws(tmp_path)
    led = Ledger(ws)
    led.log("before", 0.70)  # logged before the grind: not its work
    start = time.time() + 5
    GR.start(ws, "sess-a", experiments=3, now=start)
    for name in ("a", "b"):
        rec = led.log(name, 0.71)
        recs = [json.loads(l) for l in led.path.read_text().splitlines()]
        recs[-1]["time"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start + 60))
        led.path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    s = GR.status(ws, "sess-a", now=start + 120)
    assert s["experiments_done"] == 2 and s["experiments_left"] == 1


def _write_run(art, pid, host=None, finished=False, folds=(0, 1, 2)):
    art.mkdir(parents=True, exist_ok=True)
    (art / f"run_{pid}.json").write_text(json.dumps({
        "pid": pid, "host": host or socket.gethostname(), "cmd": "python src/train_image.py --name cnx",
        "started": T0, "heartbeat": T0 + 60, "folds": list(folds), "all_folds": [0, 1, 2], "finished": finished}))


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_pid_alive_tells_a_live_process_from_a_finished_one():
    assert GR.pid_alive(os.getpid()) is True
    assert GR.pid_alive(_dead_pid()) is False


def test_resume_reconciles_every_run_the_grind_started(tmp_path):
    ws = _ws(tmp_path)
    GR.start(ws, "sess-a", hours=4, now=T0)
    for name in ("done_run", "live_run", "dead_run"):
        GR.note_started(ws, "sess-a", name, f"python src/train_image.py --name {name}", now=T0 + 10)
    # done_run reached the ledger; live_run's process still runs; dead_run's died after fold 0 and mid fold 1
    led = Ledger(ws)
    led.log("done_run", 0.8)
    recs = [json.loads(l) for l in led.path.read_text().splitlines()]
    recs[-1]["time"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(T0 + 100))
    led.path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    _write_run(ws / "artifacts" / "live_run", os.getpid())
    dead = ws / "artifacts" / "dead_run"
    _write_run(dead, _dead_pid())
    (dead / "fold0_status.json").write_text(json.dumps({"fold": 0, "status": "done"}))
    (dead / "fold0_oof.npy").write_bytes(b"x")
    (dead / "fold1_resume.pt").write_bytes(b"x")
    (dead / "curve_fold1.json").write_text(json.dumps({"points": [{"epoch": 0, "score": 0.5}, {"epoch": 1, "score": 0.6}]}))
    # a Kaggle run the grind pushed, finished and not collected yet
    (ws / S.STATE_DIR / "gpu_runs.jsonl").write_text(json.dumps(
        {"kernel": "me/cnx-train", "name": "cnx", "status": "complete", "pushed_at": "2027-01-15T08:05:00Z"}) + "\n")
    by = {r["name"]: r for r in GR.reconcile(ws, GR.load(ws))}
    assert by["done_run"]["state"] == "done"
    assert by["live_run"]["state"] == "running"
    assert by["dead_run"]["state"] == "stopped" and by["dead_run"]["folds_done"] == [0]
    assert by["dead_run"]["folds_left"] == [1, 2] and "--folds 1 2" in by["dead_run"]["resume"]
    assert "fold 1 resumes from its checkpoint" in by["dead_run"]["detail"]
    kag = [r for r in GR.reconcile(ws, GR.load(ws)) if r.get("kernel")]
    assert kag and kag[0]["state"] == "finished" and "kgkit gpu collect me/cnx-train" in kag[0]["resume"]


def test_a_run_of_another_machine_is_not_judged_by_its_pid(tmp_path):
    ws = _ws(tmp_path)
    GR.start(ws, "sess-a", hours=4, now=T0)
    GR.note_started(ws, "sess-a", "remote_box", "python src/train_image.py --name remote_box", now=T0 + 10)
    _write_run(ws / "artifacts" / "remote_box", 1, host="some-other-host")
    (r,) = [x for x in GR.reconcile(ws, GR.load(ws)) if x["name"] == "remote_box"]
    assert r["state"] == "unknown" and "some-other-host" in r["detail"]


def test_stop_archives_the_grind_and_the_brief_shows_an_active_one(tmp_path):
    ws = _ws(tmp_path)
    assert GR.brief_line(ws, now=T0) == ""
    GR.start(ws, "sess-a", hours=4, now=T0)
    line = GR.brief_line(ws, now=T0 + 2400)
    assert "grind in progress" in line.lower() and "/kg-grind resume" in line and "kgkit grind stop" in line
    GR.stop(ws, "sess-a", reason="budget spent", now=T0 + 3000)
    assert GR.load(ws) is None and GR.brief_line(ws, now=T0 + 3000) == ""
    hist = [json.loads(l) for l in (ws / S.STATE_DIR / "grinds.jsonl").read_text().splitlines()]
    assert hist[-1]["stopped_reason"] == "budget spent" and hist[-1]["budget"] == {"hours": 4.0}
    GR.start(ws, "sess-b", experiments=5, now=T0 + 4000)  # a new grind after the last one stopped
    assert GR.load(ws)["budget"] == {"experiments": 5}
