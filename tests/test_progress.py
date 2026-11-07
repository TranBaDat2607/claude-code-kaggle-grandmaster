"""Plateau and crowd lines (UAT #88: the honest stack was flat for the last 7 days of a Playground, inside a crowd of
315 teams within 1e-5 of its score, with most of the GPU quota unused; nothing said so)."""

import datetime as dt
import json
from pathlib import Path

import pytest

from kgkit import cli
from kgkit import evalset as E
from kgkit import finals as FN
from kgkit import progress as PG
from kgkit import state as S

NOW = dt.datetime(2026, 10, 25, 12, 0)


@pytest.fixture
def ws(tmp_path):
    st = S.CompetitionState(slug="pg", metric="auc", deadline=(dt.date.today() + dt.timedelta(days=20)).isoformat())
    st.save(tmp_path)
    return tmp_path, st


def _rec(i, cv, days_ago, decision=None, tags=(), lb=None, decided_days_ago=None):
    t = NOW - dt.timedelta(days=days_ago)
    r = {"id": f"{i:04d}_m{i}", "cv": cv, "time": t.strftime("%Y-%m-%dT%H:%M:%S"), "tags": list(tags),
         "decision": decision, "lb_public": lb}
    if decision:
        d = NOW - dt.timedelta(days=decided_days_ago if decided_days_ago is not None else days_ago)
        r["decided_at"] = d.strftime("%Y-%m-%dT%H:%M:%S")
    return r


def _flat(n_after=5):
    """A gain 8 days ago, then experiments that add ~0 (and a within-noise keep)."""
    recs = [_rec(1, 0.9600, 20, "baseline"), _rec(2, 0.9622, 8, "keep")]
    recs += [_rec(3 + k, 0.9622 + 1e-7 * (k % 2), 7 - k, "discard") for k in range(n_after)]
    return recs


def _noise(root, se=2e-5, se_public=None):
    entry = {"question": "paired", "eval": E.current_key(root, S.load(root)), "se": se, "method": "bootstrap"}
    if se_public:
        entry["se_public"] = se_public
    E.log_comparison(root, entry)


def test_plateau_after_days_of_experiments_without_a_gain(ws):
    root, st = ws
    _noise(root)
    line = PG.plateau_line(root, st, _flat(), now=NOW)
    assert line.startswith("PLATEAU: the accepted baseline (0002_m2") and "for 8 days" in line
    assert "5 experiment(s) since" in line and "/kg-ideas" in line and "model family" in line
    assert "`python -m kgkit standing`" in line  # no public board cached yet
    # a keep within noise moves the reference, not the clock
    recs = _flat() + [_rec(20, 0.96221, 1, "keep")]
    assert "for 8 days" in PG.plateau_line(root, st, recs, now=NOW)
    # a real gain resets it
    recs = _flat() + [_rec(20, 0.9630, 1, "keep")]
    assert PG.plateau_line(root, st, recs, now=NOW) == ""


def test_no_plateau_when_idle_recent_or_in_the_final_days(ws):
    root, st = ws
    assert PG.plateau_line(root, st, _flat(n_after=2), now=NOW) == ""  # a pause, not a plateau
    assert PG.plateau_line(root, st, _flat(), now=NOW - dt.timedelta(days=6)) == ""  # 2 days flat
    st.deadline = (dt.date.today() + dt.timedelta(days=2)).isoformat()
    assert PG.plateau_line(root, st, _flat(), now=NOW) == ""
    # fitted records and borrowed CVs are not the honest track
    st.deadline = (dt.date.today() + dt.timedelta(days=20)).isoformat()
    fitted = [_rec(1, 0.99, 1, "keep", tags=["public-lb-fitted"])]
    assert PG.plateau(root, st, fitted, now=NOW) is None


def test_plateau_mentions_the_gpu_quota_left(ws):
    from kgkit import gpu as G

    root, st = ws
    G.quota_path(root).write_text(json.dumps({"used_h": 2, "remaining_h": 28, "total_h": 30.0,
                                              "refresh_at": "2099-01-03T00:00:00Z", "fetched_at": G._iso(G._now())}))
    assert "28h of Kaggle GPU quota are usable before the deadline" in PG.plateau_line(root, st, _flat(), now=NOW)


def _board(root, scores, private=False):
    data = {"leaderboard": {"team": "us", "shown": {"scores": scores, "ranks": [1], "complete": True}},
            "scored": [{"private_score": 0.9 if private else None}]}
    assert FN.save_board(root, "pg", data)


def test_crowd_within_the_board_resolution(ws):
    root, st = ws
    scores = [0.96473, 0.9626] + [0.96238] * 40 + [0.96237] * 30 + [0.96236] * 20 + [0.9620] * 300
    _board(root, scores)
    recs = _flat() + [_rec(30, None, 1, tags=["public-lb-fitted"], lb=0.96473), _rec(31, 0.9622, 1, lb=0.96237)]
    c = PG.crowd(root, st, recs)
    assert c["record"]["id"] == "0031_m31"  # the fitted file's public score is not ours
    assert c["n"] == 90 and c["ranks"] == (3, 92) and c["band"] == pytest.approx(1e-5)
    line = PG.crowd_line(root, st, recs)
    assert line.startswith("CROWD: 90 of 392 teams") and "one unit of the scores Kaggle shows" in line
    assert "ranks 3-92" in line and "what the crowd lacks" in line
    # measured public-LB noise widens the band
    _noise(root, se_public=1e-5)
    assert PG.crowd(root, st, recs)["band"] == pytest.approx(2e-5)
    assert "public-LB SE ~1e-05" in PG.crowd_line(root, st, recs)


def test_no_crowd_when_few_teams_are_close_or_the_board_is_private(ws):
    root, st = ws
    recs = [_rec(1, 0.96, 1, "baseline", lb=0.96237)]
    _board(root, [0.97, 0.96237, 0.96236] + [0.95 - i * 1e-4 for i in range(200)])
    assert PG.crowd_line(root, st, recs) == ""
    _board(root, [0.96237] * 100, private=True)
    assert PG.crowd(root, st, recs) is None


def test_status_and_brief_print_both(ws, capsys, monkeypatch):
    import subprocess
    import sys

    root, st = ws
    from kgkit.experiment import Ledger

    led = Ledger(root)
    led.path.parent.mkdir(parents=True, exist_ok=True)
    recs = _flat()
    for r in recs:  # moved to now, so the plateau is measured against today
        for k in ("time", "decided_at"):
            if r.get(k):
                r[k] = (dt.datetime.fromisoformat(r[k]) + (dt.datetime.now() - NOW)).strftime("%Y-%m-%dT%H:%M:%S")
    recs[-1]["lb_public"] = 0.96237
    led.path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    _board(root, [0.96237] * 60 + [0.95] * 100)
    monkeypatch.chdir(root)
    cli.main(["status", "--offline"])
    out = capsys.readouterr().out
    assert "PLATEAU:" in out and "CROWD: 60 of 160 teams" in out
    plugin = Path(cli.__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(plugin / "hooks" / "session_start.py")], capture_output=True, text=True,
                       input=json.dumps({"hook_event_name": "SessionStart", "cwd": str(root)}), timeout=60)
    ctx = json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "PLATEAU:" in ctx and "CROWD:" in ctx


# ----------------------------------------------------------------------------- #85: public-rank goals, LB fitting
def _log_subs(root, rows):
    S.submissions_log(root).parent.mkdir(parents=True, exist_ok=True)
    S.submissions_log(root).write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_lb_fitting_while_the_honest_track_is_flat(ws):
    root, st = ws
    recs = _flat()
    recs += [_rec(40 + k, None, 1, tags=["public-lb-fitted"], lb=0.9647) for k in range(6)]
    subs = [{"time_utc": f"2026-10-2{k}T10:00:00", "file": f"subs/{40 + k:04d}_tilt.csv", "exp_id": f"{40 + k:04d}_m{40 + k}",
             "status": "complete"} for k in range(6)]
    subs += [{"time_utc": "2026-10-19T10:00:00", "file": "subs/0007.csv", "exp_id": "0007_m7"},
             {"time_utc": "2026-10-19T11:00:00", "file": "subs/x.csv", "message": "blend w0.3 public-tuned"}]
    _log_subs(root, subs)
    assert PG.lb_chase(root, st, recs) == {"n": 8, "k": 7}
    line = PG.fitting_line(root, st, recs, now=NOW)
    assert line.startswith("LB-FITTING: the honest track has been flat for 8 days (accepted baseline 0002_m2) while "
                           "7 of the last 8 submissions went to fitting or probing the public LB")
    assert "cannot raise the private score" in line and "/kg-ideas" in line
    _log_subs(root, subs[:3] + [{"time_utc": "2026-10-23T10:00:00", "file": f"subs/h{k}.csv"} for k in range(5)])
    assert PG.fitting_line(root, st, recs, now=NOW) == ""  # 3 of 8: not most of them


def test_a_public_rank_goal_gets_the_honest_target_once_per_session(ws, plugin_root):
    import subprocess
    import sys

    root, st = ws
    from kgkit.experiment import Ledger

    led = Ledger(root)
    led.attach_lb(led.log("stack", 0.962249)["id"], 0.96185)
    led.decide("0001", "baseline")
    led.attach_lb(led.log("tilt", None, tags=["public-lb-fitted"])["id"], 0.96473)
    import uuid

    sid = uuid.uuid4().hex  # the router remembers what it said per session, in the temp dir

    def ask(prompt, session=sid):
        r = subprocess.run([sys.executable, str(plugin_root / "hooks" / "prompt_router.py")], capture_output=True,
                           text=True, timeout=60, input=json.dumps({"hook_event_name": "UserPromptSubmit",
                                                                     "prompt": prompt, "cwd": str(root),
                                                                     "session_id": session}))
        return json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"] if r.stdout.strip() else ""

    ctx = ask("/goal public top 10")
    assert "names a public-leaderboard rank or score" in ctx and "zero expected private value" in ctx
    assert "above the accepted baseline 0001_stack (0.962249)" in ctx
    assert "better honest public score than 0.96185 (0001_stack)" in ctx  # not the fitted 0.96473
    assert "public-leaderboard rank" not in ask("goal check-in: keep pushing toward public top 10")  # once
    assert "public-leaderboard rank" in ask("let's get top 5 on the public LB", session=sid + "-another-session")
    assert ask("/goal private top 10", session=sid + "-third") == ""
    st.deadline = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    st.save(root)
    assert "public-leaderboard rank" not in ask("public top 10?", session=sid + "-fourth")
