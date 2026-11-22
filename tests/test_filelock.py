"""#126: several processes write the ledger and the backlog at once (a run logging its record, `gpu collect`,
`ledger decide` meanwhile, two experiments finishing together). A lock file serialises the writers: ids stay unique
and a rewrite never drops a record appended while it ran."""

import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

from kgkit import filelock as FL
from kgkit import state as S
from kgkit.backlog import Backlog
from kgkit.experiment import Ledger


def _ws(tmp_path):
    S.CompetitionState(slug="demo", metric="auc", target="y", id_col="id").save(tmp_path)
    return tmp_path


def _spawn(tmp_path, plugin_root, code, n):
    """Start ``n`` processes running ``code`` (``I`` is each one's index) at the same moment."""
    start = time.time() + 2.0
    script = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(plugin_root)!r})
        I = int(sys.argv[1])
        while time.time() < {start}:
            time.sleep(0.001)
    """) + textwrap.dedent(code)
    env = dict(os.environ, KG_NO_SNAPSHOT="1")
    procs = [subprocess.Popen([sys.executable, "-c", script, str(i)], cwd=tmp_path, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(n)]
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err[-2000:]


def test_runs_logging_at_once_get_distinct_ids(tmp_path, plugin_root):
    _ws(tmp_path)
    _spawn(tmp_path, plugin_root, """
        from kgkit.experiment import Ledger
        Ledger(".").log(f"run{I}", 0.5 + I / 100, fold_scores=[0.5, 0.5])
    """, n=6)
    recs = Ledger(tmp_path).records()
    nums = [r["id"].split("_")[0] for r in recs]
    assert len(recs) == 6 and len(set(nums)) == 6, nums
    assert not FL.lock_path(Ledger(tmp_path).path).exists()  # released


def test_a_rewrite_waits_for_an_append_and_keeps_it(tmp_path, plugin_root):
    _ws(tmp_path)
    led = Ledger(tmp_path)
    first = led.log("base", 0.7)
    with FL.locked(led.path):  # a decide in progress here: the other process must wait for it
        p = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(plugin_root)!r})
            from kgkit.experiment import Ledger
            Ledger(".").log("late", 0.71)
        """)], cwd=tmp_path, env=dict(os.environ, KG_NO_SNAPSHOT="1"))
        time.sleep(1.5)
        assert [r["name"] for r in led.records()] == ["base"]  # it is waiting, not appending
        led.decide(first["id"], "baseline")  # re-entrant: decide -> update in the same process
    assert p.wait(timeout=60) == 0
    recs = {r["name"]: r for r in Ledger(tmp_path).records()}
    assert set(recs) == {"base", "late"} and recs["base"]["decision"] == "baseline"


def test_a_number_two_records_share_is_refused(tmp_path):
    _ws(tmp_path)
    led = Ledger(tmp_path)
    led.path.parent.mkdir(parents=True, exist_ok=True)
    led.path.write_text("".join(json.dumps({"id": i, "name": n, "cv": 0.5}) + "\n"
                                for i, n in (("0011_a", "a"), ("0012_cnx", "cnx"), ("0012_effb3", "effb3"))))
    with pytest.raises(KeyError, match="ambiguous: 0012_cnx, 0012_effb3"):
        led.get("12")
    assert led.get("0012_effb3")["name"] == "effb3" and led.get("cnx")["id"] == "0012_cnx"
    assert led.get("11")["id"] == "0011_a"
    assert led.ids("11..12") == ["0011_a", "0012_cnx", "0012_effb3"]  # a range still takes both


def test_a_stale_lock_is_taken_over_and_a_live_one_times_out(tmp_path):
    target = tmp_path / "ledger.jsonl"
    lock = FL.lock_path(target)
    lock.write_text("pid 1 on elsewhere since long ago\n")
    old = time.time() - FL.STALE_S - 60
    os.utime(lock, (old, old))
    with FL.locked(target, timeout=5):  # its holder died long ago: taken over at once
        assert "pid %d" % os.getpid() in lock.read_text()
    assert not lock.exists()
    lock.write_text("pid 1 on elsewhere since now\n")  # fresh: someone holds it
    with pytest.raises(FL.LockTimeout, match="held by another process"):
        with FL.locked(target, timeout=0.3):
            pass
    lock.unlink()


def test_backlog_writers_from_parallel_agents_keep_every_item(tmp_path, plugin_root):
    _ws(tmp_path)
    _spawn(tmp_path, plugin_root, """
        from kgkit.backlog import Backlog
        Backlog(".").add(f"idea {I}", gain=3, prob=0.5, cost=1)
    """, n=6)
    items = Backlog(tmp_path).items()
    assert sorted(i["idea"] for i in items) == [f"idea {i}" for i in range(6)]
    assert len({i["id"] for i in items}) == 6


def test_lock_and_temp_files_stay_out_of_git(tmp_path):
    (tmp_path / S.STATE_DIR).mkdir()
    S.ensure_state_ignores(tmp_path)
    ignores = (tmp_path / S.STATE_DIR / ".gitignore").read_text(encoding="utf-8").split()
    assert "*.lock" in ignores and "*.tmp" in ignores
