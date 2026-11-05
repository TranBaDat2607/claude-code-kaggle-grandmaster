"""#70: after the deadline. The last-days warning when no finals are confirmed, private scores from Kaggle's list
(`kgkit ledger sync-lb`), the outcome of `kgkit final` (finals, standing, medal zone), and which signal predicted
the private leaderboard (`kgkit ledger predictiveness`)."""

import datetime as dt
import json
import subprocess
import sys
import textwrap
from types import SimpleNamespace as N

import pytest

from kgkit import cli
from kgkit import finals as FN
from kgkit import gpu as G
from kgkit import predictiveness as P
from kgkit import state as S
from kgkit.experiment import Ledger


def _utc_in(days):
    return (dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=days)).isoformat()


def _session_context(plugin_root, cwd):
    r = subprocess.run([sys.executable, str(plugin_root / "hooks" / "session_start.py")], capture_output=True,
                       input=json.dumps({"hook_event_name": "SessionStart", "cwd": str(cwd)}), text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"]


# ----------------------------------------------------------------------------- the last days
def test_the_last_days_warn_whenever_no_selection_is_confirmed(tmp_path, monkeypatch, capsys, plugin_root):
    """The issue's miss: nothing selected, no fitted file, five submissions tied at the cut, and no warning."""
    S.CompetitionState(slug="demo", metric="auc", deadline=_utc_in(1)).save(tmp_path)
    led = Ledger(tmp_path)
    for i, lb in enumerate([0.950, 0.951, 0.958, 0.958, 0.958, 0.958, 0.958]):
        led.attach_lb(led.log(f"a{i}", 0.930 + i / 1000)["id"], lb)
    monkeypatch.chdir(tmp_path)
    cli.main(["status"])
    out = capsys.readouterr().out
    warning = next(l for l in out.splitlines() if l.startswith("WARNING: no finals confirmed on Kaggle (1 day left)."))
    assert "With nothing selected, Kaggle takes the 2 best public scores: 0003_a2 0.95800, 0004_a3 0.95800" in warning
    assert "; 5 tie at the cut and Kaggle picks among them." in warning
    assert f"by {_utc_in(1)} 23:59 UTC: https://www.kaggle.com/competitions/demo/submissions" in warning
    assert "then run `python -m kgkit final`" in warning and "Honest candidates: " in warning
    assert warning in _session_context(plugin_root, tmp_path).splitlines()

    st = S.load(tmp_path)  # confirmed on Kaggle: a note, no warning
    st.final_picks, st.final_picks_confirmed = ["0003_a2", "0001_a0"], "2026-10-21 10:00 UTC"
    st.save(tmp_path)
    assert FN.lines(tmp_path, S.load(tmp_path), led.records())[0].startswith("finals selected on Kaggle (checked ")
    st.final_picks, st.final_picks_confirmed, st.deadline = [], "", _utc_in(S.FINAL_DAYS + 1)
    st.save(tmp_path)  # before the last days, nothing without a fitted file
    assert FN.lines(tmp_path, S.load(tmp_path), led.records()) == []
    st.deadline = _utc_in(0)
    st.save(tmp_path)  # the deadline's own day still warns, with nothing on record but the ledger
    assert FN.lines(tmp_path, S.load(tmp_path), led.records())[0].startswith(
        "WARNING: no finals confirmed on Kaggle (0 days left).")


# ----------------------------------------------------------------------------- helpers after the deadline
def test_submission_rows_carry_the_private_score_and_the_team():
    from kgkit import kaggle_submissions as KS

    r = KS.row(N(ref=7, date=dt.datetime(2026, 10, 22, 13, 47), description="0026_a26-oai | x", file_name="s.csv",
                 public_score="0.957", private_score="0.941", status=N(name="COMPLETE"), team_name="solo"))
    assert (r["public_score"], r["private_score"], r["team_name"], r["date"]) == (0.957, 0.941, "solo",
                                                                                  "2026-10-22T13:47:00")
    assert r["digits"] == 3
    r = KS.row(N(ref=8, date=None, description=None, file_name=None, public_score="", private_score=None, status=None))
    assert r["public_score"] is None and r["private_score"] is None and r["team_name"] == "" and r["digits"] is None
    assert KS.row(N(public_score="0.930", private_score=None))["digits"] == 3  # a float would say 0.93


def test_medal_cutoffs_follow_kaggles_progression_table():
    assert FN.medal_cutoffs(6299) == {"gold": 22, "silver": 314, "bronze": 629}  # the issue's competition
    assert FN.medal_cutoffs(1000) == {"gold": 12, "silver": 50, "bronze": 100}
    assert FN.medal_cutoffs(999) == {"gold": 11, "silver": 50, "bronze": 100}
    assert FN.medal_cutoffs(500) == {"gold": 11, "silver": 50, "bronze": 100}
    assert FN.medal_cutoffs(150) == {"gold": 10, "silver": 30, "bronze": 60}
    assert FN.medal_cutoffs(50) == {"gold": 5, "silver": 10, "bronze": 20}


def test_best_of_n_spans_the_ways_a_tie_at_the_cut_can_be_broken():
    tied = [(0.958, 0.942), (0.958, 0.943), (0.958, 0.941), (0.950, 0.990)]
    assert FN.best_of_n(tied, 1, True) == (0.941, 0.943)
    assert FN.best_of_n(tied, 2, True) == (0.942, 0.943)
    assert FN.best_of_n(tied, 9, True) == (0.990, 0.990)  # more picks than files: all of them
    assert FN.best_of_n([(0.20, 0.30), (0.20, 0.25), (0.10, 0.40)], 2, False) == (0.30, 0.25)  # lower is better
    assert FN.best_of_n([(0.9, None)], 1, True) is None


def test_a_score_typed_with_more_digits_is_the_same_score():
    assert FN.same_score(0.9586, 0.959) and FN.same_score(0.96185, 0.96185)
    assert not FN.same_score(0.9586, 0.957) and not FN.same_score(None, 0.9) and not FN.same_score(0.9, None)
    assert FN.same_score(0.925, 0.93) and not FN.same_score(0.925, 0.93, 3)  # Kaggle showed 0.930: three decimals


def test_spearman_has_an_interval_and_knows_when_to_say_nothing():
    r = P.spearman([1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60])
    assert r["rho"] == pytest.approx(1.0) and r["hi"] == pytest.approx(1.0) and r["lo"] is not None
    assert P.spearman([1, 2], [1, 2])["rho"] is None  # too few points
    assert P.spearman([1, 1, 1, 1, 1], [1, 2, 3, 4, 5])["rho"] is None  # a constant signal ranks nothing
    small = P.spearman([1, 2, 3, 4], [4, 3, 2, 1])
    assert small["rho"] == pytest.approx(-1.0) and small["lo"] is None  # no interval under MIN_CI points


# ----------------------------------------------------------------------------- a fake Kaggle after the deadline
SUBS = [  # ref, date, file, description, public, private
    (1, "2026-10-01T10:00:00", "0001_base.csv", "0001_base | first", "0.910", "0.900"),
    (2, "2026-10-02T10:00:00", "sub.csv", "0002 m2 lgbm", "0.930", "0.915"),  # linked by its leading number
    (3, "2026-10-03T10:00:00", "0003_m3.csv", "m3 again", "0.940", "0.925"),  # linked by its file name
    (4, "2026-10-04T10:00:00", "submission.csv", "0004_a16 | public recipe", "0.958", "0.942"),
    (5, "2026-10-05T10:00:00", "submission.csv", "0005_a17 | + s6fall", "0.958", "0.943"),
    (6, "2026-10-06T10:00:00", "submission.csv", "0006_a20 | + S6T", "0.958", "0.943"),
    (7, "2026-10-07T10:00:00", "submission.csv", "blend v3 by hand", "0.950", "0.935"),  # names no record
    (8, "2026-10-08T10:00:00", "submission.csv", "0003_m3 rerun", "0.941", "0.926"),  # 0003's newer submission
    (9, "2026-10-09T10:00:00", "tilt.csv", "0007 [public-LB fitted] tilt", "0.957", "0.930"),
]

FAKE_KAGGLE = textwrap.dedent("""\
    import datetime, json, os
    from types import SimpleNamespace as N
    DATA = json.load(open(os.environ['FAKE_KAGGLE']))
    class Client:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        @property
        def competitions(self): return N(competition_api_client=self)
        def list_submissions(self, r):
            assert r.competition_name == 'demo'
            rows = [N(ref=ref, date=datetime.datetime.fromisoformat(d), description=desc, file_name=f,
                      public_score=pub, private_score=priv if DATA['private'] else None, status=N(name='COMPLETE'),
                      team_name=DATA['team']) for ref, d, f, desc, pub, priv in DATA['subs']]
            if r.group == 2:
                return N(submissions=[x for x in rows if x.ref in DATA['selected']], next_page_token='')
            return N(submissions=rows, next_page_token='')
        def get_leaderboard(self, r):
            if DATA['lb_fail']: raise RuntimeError('leaderboard unavailable')
            board = DATA['public_board'] if r.override_public else DATA['board']
            page = int(r.page_token or 0)
            rows = board[page * r.page_size:(page + 1) * r.page_size]
            token = str(page + 1) if (page + 1) * r.page_size < len(board) else ''
            return N(submissions=[N(team_name=t, score=s) for t, s in rows], next_page_token=token)
    class KaggleApi:
        def authenticate(self): pass
        def build_kaggle_client(self): return Client()
    """)


def _board(teams, rank, team_score, winning="0.960"):
    """`teams` rows, best first: ours at `rank`; the others fall evenly from the winning score."""
    rows = [[f"team {i}", f"{float(winning) - (i - 1) * 0.017 / 119:.3f}"] for i in range(1, teams + 1)]
    rows[rank - 1] = ["my team", team_score]
    return rows


@pytest.fixture
def fake_kaggle(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("sdk")
    files = {
        "kaggle/__init__.py": "",
        "kaggle/api/__init__.py": "",
        "kaggle/api/kaggle_api_extended.py": FAKE_KAGGLE,
        "kagglesdk/__init__.py": "",
        "kagglesdk/competitions/__init__.py": "",
        "kagglesdk/competitions/types/__init__.py": "",
        "kagglesdk/competitions/types/competition_api_service.py": (
            "class ApiListSubmissionsRequest:\n    page_token = ''\n"
            "class ApiGetLeaderboardRequest:\n    page_token = ''\n    override_public = False\n"),
        "kagglesdk/competitions/types/competition_enums.py": (
            "class SubmissionGroup:\n    SUBMISSION_GROUP_ALL = 0\n    SUBMISSION_GROUP_SUCCESSFUL = 1\n"
            "    SUBMISSION_GROUP_SELECTED = 2\nclass SubmissionSortBy:\n    SUBMISSION_SORT_BY_DATE = 0\n"),
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    monkeypatch.setenv("PYTHONPATH", str(root))
    monkeypatch.setattr(G, "kaggle_python", lambda: sys.executable)

    def configure(selected=(), private=True, team_score="0.943", lb_fail=False):
        data = {"team": "my team", "subs": SUBS, "selected": list(selected), "private": private, "lb_fail": lb_fail,
                "board": _board(1500, 120, team_score), "public_board": _board(1500, 150, "0.958", "0.970")}
        (root / "data.json").write_text(json.dumps(data))
        monkeypatch.setenv("FAKE_KAGGLE", str(root / "data.json"))
    return configure


def _ended_ws(root, days=-2, **state):
    """Six records with their own CV, a public-LB-fitted file, two public scores typed by hand: 0002's disagrees
    with Kaggle's, 0004's has more digits than Kaggle shows."""
    S.CompetitionState(slug="demo", metric="auc", deadline=_utc_in(days), **state).save(root)
    led = Ledger(root)
    for name, cv in [("base", 0.90), ("m2", 0.92), ("m3", 0.93), ("a16", 0.940), ("a17", 0.941), ("a20", 0.939)]:
        led.log(name, cv)
    led.log("tilt", None, tags=["public-lb-fitted"])
    led.attach_lb("0002_m2", 0.925)
    led.attach_lb("0004_a16", 0.9581)
    return led


def test_final_after_the_deadline_reports_the_outcome_when_nothing_was_selected(tmp_path, monkeypatch, capsys,
                                                                              fake_kaggle):
    _ended_ws(tmp_path)
    fake_kaggle()
    monkeypatch.chdir(tmp_path)
    data, err = FN.live("demo", leaderboard=True)
    assert err is None and data["leaderboard"]["team"] == "my team"
    assert data["leaderboard"]["shown"]["ranks"] == [120] and len(data["leaderboard"]["shown"]["scores"]) == 1500
    assert data["leaderboard"]["public"]["ranks"] == [150] and len(data["leaderboard"]["public"]["scores"]) == 200
    cli.main(["final"])
    out = capsys.readouterr().out
    assert f"The competition has ended ({_utc_in(-2)}). Kaggle lists 9 scored submission(s) for demo; 0 selected" in out
    assert ("Private leaderboard: rank 120 of 1,500 teams (top 8.0%), team score 0.94300, winning score 0.96000; "
            "public leaderboard rank 150 (0.95800).") in out
    assert "Medal zone by Kaggle's progression table, computed here: bronze (gold to rank 13 (0.95800), " in out
    assert "silver to rank 75 (0.94900), bronze to rank 150 (0.93900))" in out
    assert ("Nothing was selected, so Kaggle picked the finals itself among the 2 best public score(s); 3 tie at the "
            "cut (public / private):") in out
    rows = [l for l in out.splitlines() if l.startswith("  0.95800 / ")]
    assert [l.split()[3] for l in rows] == ["0004_a16", "0005_a17", "0006_a20"]
    assert ("The team's private score 0.94300 matches 0005_a17, 0006_a20: Kaggle's picks included at least one of "
            "them.") in out
    assert "Best private score of any submission: 0.94300 (0005_a17, 0006_a20). The finals reached it." in out
    assert "Selecting finals (only the website" not in out and "Deadline:" not in out  # no steps after the deadline
    assert "next: `python -m kgkit ledger sync-lb`" in out
    assert S.load(tmp_path).final_picks == []


def test_final_after_the_deadline_with_finals_selected(tmp_path, monkeypatch, capsys, fake_kaggle):
    _ended_ws(tmp_path)
    fake_kaggle(selected=[4, 7], team_score="0.942")
    monkeypatch.chdir(tmp_path)
    cli.main(["final"])
    out = capsys.readouterr().out
    assert "Selected as finals (2 of 2), public / private:" in out
    assert "Kaggle scores a team by the best private score among its finals: 0.94200." in out
    assert ("Best private score of any submission: 0.94300 (0005_a17, 0006_a20). The finals scored 0.94200, 0.00100 "
            "short of it.") in out
    assert "recorded final_picks = 0004_a16, sub:7 (confirmed on Kaggle)" in out


def test_final_after_the_deadline_before_private_scores_and_without_the_leaderboard(tmp_path, monkeypatch, capsys,
                                                                                    fake_kaggle):
    _ended_ws(tmp_path)
    monkeypatch.chdir(tmp_path)
    fake_kaggle(private=False)
    cli.main(["final"])
    out = capsys.readouterr().out
    assert "No private scores yet: Kaggle shows them once the private leaderboard is final" in out
    assert "Medal zone" not in out and "Selecting finals" not in out

    fake_kaggle(lb_fail=True)
    cli.main(["final"])
    out = capsys.readouterr().out
    assert "Leaderboard not read (RuntimeError: leaderboard unavailable)." in out
    assert "Depending on how Kaggle broke the tie, the team scored 0.94300 privately" in out  # any 2 of 3 hold a 0.943

    monkeypatch.setattr(FN, "live", lambda slug, **kw: (None, "not authenticated"))
    cli.main(["final"])
    out = capsys.readouterr().out
    assert "The deadline has passed, but Kaggle could not be asked for the outcome (not authenticated)." in out
    assert "Selecting finals" not in out


# ----------------------------------------------------------------------------- ledger sync-lb and predictiveness
def test_sync_lb_attaches_every_score_kaggle_lists(tmp_path, monkeypatch, capsys, fake_kaggle):
    _ended_ws(tmp_path)
    fake_kaggle()
    monkeypatch.chdir(tmp_path)
    cli.main(["ledger", "sync-lb", "--dry-run"])
    out = capsys.readouterr().out
    assert "Kaggle lists 9 scored submission(s) for demo, 9 with a private score." in out
    assert "8 of them name 7 ledger record(s); would update 7:" in out and "dry run: nothing written" in out
    assert Ledger(tmp_path).get("0005")["lb_private"] is None and not S.kaggle_list(tmp_path).exists()

    cli.main(["ledger", "sync-lb"])
    out = capsys.readouterr().out
    recs = {r["id"]: r for r in Ledger(tmp_path).records()}
    assert (recs["0005_a17"]["lb_public"], recs["0005_a17"]["lb_private"]) == (0.958, 0.943)
    assert recs["0002_m2"]["lb_public"] == 0.93 and "public 0.930 (was 0.925), private 0.915" in out
    assert (recs["0004_a16"]["lb_public"], recs["0004_a16"]["lb_private"]) == (0.9581, 0.942)
    assert "0004_a16: kept lb_public 0.9581; Kaggle shows 0.958, the same score to the digits it gives" in out
    assert recs["0003_m3"]["lb_private"] == 0.926 and "0003_m3: 2 scored submissions name it" in out
    assert recs["0007_tilt"]["lb_private"] == 0.93  # fitted files keep their scores; predictiveness leaves them out
    assert "1 scored submission(s) name no ledger record:" in out and "blend v3 by hand" in out
    assert out.count("CV-LB correlation") == 1 and "next: python -m kgkit ledger predictiveness" in out
    saved = json.loads(S.kaggle_list(tmp_path).read_text(encoding="utf-8"))
    assert len(saved["scored"]) == 9 and saved["selected"] == [] and saved["competition"] == "demo"

    cli.main(["ledger", "sync-lb"])  # a second run changes nothing
    assert "8 of them name 7 ledger record(s); updated 0." in capsys.readouterr().out


def test_predictiveness_ranks_the_signals_and_the_ways_of_picking(tmp_path, monkeypatch, capsys, fake_kaggle):
    _ended_ws(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="run `python -m kgkit ledger sync-lb` first"):
        cli.main(["ledger", "predictiveness"])
    fake_kaggle()
    cli.main(["ledger", "sync-lb"])
    capsys.readouterr()
    cli.main(["ledger", "predictiveness", "--top", "3"])
    out = capsys.readouterr().out
    lines = {l.split("  ")[1].strip(): l for l in out.splitlines() if l.startswith("  ") and "  " in l[2:]}
    assert lines["every scored submission"].split()[3] == "8"  # nine scored, less the fitted file
    assert "records with their own validation score" in out and "top 3 of those by public score" in out
    assert "top 3 of those by validation score" in out
    assert "note: 1 public-LB-fitted submission(s) left out" in out
    assert "note: private scores tie: 5 distinct values among 6, as Kaggle rounds them" in out
    assert "best CV:" in out and "your finals:" in out and "none selected on Kaggle" in out
    assert "; Kaggle's rule when nothing is selected" in out
    assert " ".join(lines["best possible:"].split()) == "best possible: private 0.94300 (0005_a17, 0006_a20)"
    assert " ".join(lines["best CV:"].split()) == "best CV: private 0.94300"  # 0005 and 0004 (0.942), 0005 has 0.943
    assert "Public minus private (positive: private was worse): mean +0.01" in out and "over 8 submissions." in out

    pub = P.spearman([0.910, 0.930, 0.940, 0.958, 0.958, 0.958, 0.950, 0.941],
                     [0.900, 0.915, 0.925, 0.942, 0.943, 0.943, 0.935, 0.926])
    assert pub["rho"] > 0.9 and f"{pub['rho']:+.2f}" in lines["every scored submission"]


def test_predictiveness_figures_or_why_not(tmp_path, monkeypatch, capsys, fake_kaggle):
    _ended_ws(tmp_path, validation="holdout+lb")
    fake_kaggle()
    monkeypatch.chdir(tmp_path)
    cli.main(["ledger", "sync-lb"])
    capsys.readouterr()
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # as if it were not installed
    cli.main(["ledger", "predictiveness", "--plot", "figs"])
    out = capsys.readouterr().out
    assert "holdout -> private" in out and "best holdout:" in out
    assert "figures not written: matplotlib is not installed (pip install matplotlib)" in out


def test_predictiveness_writes_the_figures(tmp_path, monkeypatch, capsys, fake_kaggle):
    pytest.importorskip("matplotlib")
    _ended_ws(tmp_path)
    fake_kaggle()
    monkeypatch.chdir(tmp_path)
    cli.main(["ledger", "sync-lb"])
    cli.main(["ledger", "predictiveness", "--plot", "figs"])
    assert "wrote " in capsys.readouterr().out
    assert {p.name for p in (tmp_path / "figs").iterdir()} == {"public_vs_private.png", "private_over_time.png",
                                                               "cv_vs_private.png"}
