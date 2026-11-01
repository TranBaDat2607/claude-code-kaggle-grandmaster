"""Adopting an existing mid-competition repo: detection, `kgkit adopt`, submission history import."""

import datetime as dt
import json

import pytest

from kgkit import adopt as A
from kgkit import cli
from kgkit import state as S

NOW = dt.datetime.now(dt.timezone.utc)


def _repo(root, internet=False):
    for name, comp in (("infer", ["rsna-knee"]), ("train", ["rsna-knee"]), ("util", [])):
        d = root / "kernels" / name
        d.mkdir(parents=True)
        (d / "kernel-metadata.json").write_text(json.dumps(
            {"id": f"me/{name}", "competition_sources": comp, "enable_internet": "false" if not internet else True}))
    (root / "data" / "deep" / "x").mkdir(parents=True)
    (root / "data" / "deep" / "x" / "kernel-metadata.json").write_text(json.dumps({"competition_sources": ["other"]}))
    return root


def _history(n_today=2):
    ts = [NOW - dt.timedelta(days=3)] + [NOW - dt.timedelta(minutes=10 * (i + 1)) for i in range(n_today)]
    rows = [{"fileName": "submission.csv", "date": t.strftime("%Y-%m-%d %H:%M:%S"),
             "description": f"000{i + 1}_effnet | CV 0.7{i}", "status": "SubmissionStatus.COMPLETE",
             "publicScore": f"0.6{i}", "privateScore": None} for i, t in enumerate(ts)]
    rows.append({"fileName": "submission.csv", "date": NOW.strftime("%Y-%m-%d %H:%M:%S"), "description": "broken",
                 "status": "SubmissionStatus.ERROR", "publicScore": "", "privateScore": ""})
    return rows


def test_detect(tmp_path):
    assert A.detect(tmp_path) is None
    _repo(tmp_path)
    found = A.detect(tmp_path)
    assert found["slugs"] == ["rsna-knee"] and found["internet_off"]  # data/ is not searched
    assert found["kernels"] == ["kernels/infer", "kernels/train", "kernels/util"]
    (tmp_path / "README.md").write_text("see kaggle.com/c/rsna-knee/leaderboard and kaggle.com/competitions/overview")
    assert A.detect(tmp_path)["slugs"] == ["rsna-knee"]


def test_detect_readme_only_and_internet_on(tmp_path):
    (tmp_path / "README.md").write_text("https://www.kaggle.com/competitions/playground-series-s6e2")
    found = A.detect(tmp_path)
    assert found["slugs"] == ["playground-series-s6e2"] and found["kernels"] == [] and not found["internet_off"]
    assert not A.detect(_repo(tmp_path / "x", internet=True))["internet_off"]


def test_parse_submissions_json_csv_and_noise():
    rows = _history()
    text = "Warning: Looks like you're using an outdated API Version\n" + json.dumps(rows)
    recs = A.parse_submissions(text, "rsna-knee")
    assert len(recs) == 4 and recs[0]["exp_id"] == "0001_effnet"
    assert [r["status"] for r in recs].count("failed") == 1 and recs[0]["public_score"] == 0.60
    assert all(r["time_utc"].endswith("Z") and r["competition"] == "rsna-knee" for r in recs)
    csv_text = "fileName,date,description,status,publicScore,privateScore\n" \
               "sub.csv,2026-01-02 03:04:05,0012_lgbm v2,complete,0.81,\n"
    (r,) = A.parse_submissions(csv_text, "x")
    assert r["time_utc"] == "2026-01-02T03:04:05Z" and r["exp_id"] == "0012_lgbm" and r["status"] == "scored"


def test_import_submissions_dedupes_against_hook_log(tmp_path):
    S.CompetitionState(slug="rsna-knee").save(tmp_path)
    recs = A.parse_submissions(json.dumps(_history()), "rsna-knee")
    hook_time = (dt.datetime.strptime(recs[1]["time_utc"], "%Y-%m-%dT%H:%M:%SZ") + dt.timedelta(seconds=40))
    (tmp_path / ".kaggle-gm" / "submissions.jsonl").write_text(json.dumps(
        {"time_utc": hook_time.strftime("%Y-%m-%dT%H:%M:%SZ"), "message": recs[1]["message"], "status": "submitted"}) + "\n")
    assert A.import_submissions(tmp_path, recs) == (3, 1)
    assert A.import_submissions(tmp_path, recs) == (0, 4)


def test_cli_adopt(tmp_path, capsys, monkeypatch):
    _repo(tmp_path)
    (tmp_path / "CLAUDE.md").write_text("my rules\n")
    subs = tmp_path / "subs.json"
    subs.write_text(json.dumps(_history(n_today=3)))
    cli.main(["adopt", "--dir", str(tmp_path), "--metric", "auc", "--daily-subs", "5", "--submissions-file", str(subs)])
    out = capsys.readouterr().out
    st = S.load(tmp_path)
    assert st.slug == "rsna-knee" and not st.internet_allowed and st.metric == "auc"
    assert "competition: rsna-knee (detected)" in out and "--code-competition" in out
    assert "5 on Kaggle, 5 imported" in out and "3 today (UTC)" in out
    assert (tmp_path / "CLAUDE.md").read_text() == "my rules\n"
    assert not (tmp_path / "src").exists() and not (tmp_path / ".gitignore").exists()
    # the hooks now count today's imported submissions (the failed one is excluded)
    monkeypatch.syspath_prepend(str(cli.PLUGIN_ROOT / "hooks"))
    import _common as C

    assert C.submissions_today(tmp_path) == 3
    # re-running keeps competition.json and skips what is already logged
    cli.main(["adopt", "rsna-knee", "--dir", str(tmp_path), "--submissions-file", str(subs)])
    out = capsys.readouterr().out
    assert "keeping it" in out and "0 imported, 5 already logged" in out and S.load(tmp_path).metric == "auc"


def test_cli_adopt_needs_slug_unless_detected_once(tmp_path):
    (tmp_path / "README.md").write_text("kaggle.com/c/aaa-bbb and kaggle.com/c/ccc-ddd")
    with pytest.raises(SystemExit, match="candidates aaa-bbb, ccc-ddd"):
        cli.main(["adopt", "--dir", str(tmp_path), "--no-submissions"])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="none detected"):
        cli.main(["adopt", "--dir", str(empty), "--no-submissions"])
    assert not (tmp_path / ".kaggle-gm").exists() and not (empty / ".kaggle-gm").exists()


# ----------------------------------------------------------------------------- #63 the log checked against Kaggle's list
def _row(when, name="a.csv", desc="0001_x | CV 0.8"):
    return {"fileName": name, "date": when.strftime("%Y-%m-%d %H:%M:%S"), "description": desc,
            "status": "SubmissionStatus.COMPLETE", "publicScore": "0.7", "privateScore": None}


def _log(tmp_path, *recs):
    S.CompetitionState(slug="demo").save(tmp_path)
    S.submissions_log(tmp_path).write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_import_matches_one_to_one_and_by_file_name(tmp_path):
    t = NOW - dt.timedelta(minutes=30)
    # the wrapper logged it 20 min after Kaggle's date, by file path and a message Kaggle shows differently
    _log(tmp_path, {"time_utc": (t + dt.timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "file": "subs/a.csv", "message": "x", "status": "submitted"})
    recs = A.parse_submissions(json.dumps([_row(t)]), "demo")
    assert A.import_submissions(tmp_path, recs) == (0, 1)
    # two submits of one file within an hour are two; one logged record absorbs only one of them
    recs = A.parse_submissions(json.dumps([_row(t), _row(t + dt.timedelta(minutes=5))]), "demo")
    assert A.import_submissions(tmp_path, recs) == (1, 1)
    assert A.import_submissions(tmp_path, recs) == (0, 2)
    assert len(S.submission_records(tmp_path)) == 2


def test_import_unknown_identity_absorbs_one_row(tmp_path):
    t = NOW - dt.timedelta(minutes=10)
    _log(tmp_path, {"time_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "file": "$FILE", "message": None, "status": "submitted"})
    recs = A.parse_submissions(json.dumps([_row(t, "b.csv", "m1"), _row(t, "c.csv", "m2")]), "demo")
    assert A.import_submissions(tmp_path, recs) == (1, 1)


def test_sync_adds_missing_submits_and_stamps_the_check(tmp_path, monkeypatch):
    _log(tmp_path)
    st = S.load(tmp_path)
    assert "local log only, not checked against Kaggle today" in S.submissions_line(tmp_path, st)
    rows = [_row(NOW - dt.timedelta(minutes=m), desc=f"000{m}_x") for m in (5, 15, 25)]
    monkeypatch.delenv("KGKIT_OFFLINE")
    monkeypatch.setattr(A, "fetch_history", lambda slug, timeout=120: (json.dumps(rows), None))
    assert A.sync(tmp_path, "demo") == (3, None)
    line = S.submissions_line(tmp_path, st)
    assert "Submissions logged today (UTC): 3/5" in line and "checked against Kaggle's list at" in line
    assert A.sync(tmp_path, "demo") == (0, None)  # again: nothing new


def test_sync_failure_keeps_the_local_only_note(tmp_path, monkeypatch):
    _log(tmp_path)
    monkeypatch.delenv("KGKIT_OFFLINE")
    monkeypatch.setattr(A, "fetch_history", lambda slug, timeout=120: (None, "no kaggle CLI on PATH"))
    assert A.sync(tmp_path, "demo") == (0, "no kaggle CLI on PATH")
    assert "local log only" in S.submissions_line(tmp_path, S.load(tmp_path))


def test_sync_is_off_with_kgkit_offline(tmp_path, monkeypatch):
    _log(tmp_path)
    monkeypatch.setattr(A, "fetch_history", lambda *a, **k: pytest.fail("asked Kaggle"))
    assert A.sync(tmp_path, "demo")[0] == 0 and not S.sync_stamp(tmp_path).exists()


def test_cli_status_syncs_unless_offline(tmp_path, capsys, monkeypatch):
    _log(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KGKIT_OFFLINE")
    monkeypatch.setattr(A, "fetch_history", lambda slug, timeout=120: (json.dumps([_row(NOW - dt.timedelta(minutes=5))]), None))
    cli.main(["status", "--offline"])
    assert "local log only" in capsys.readouterr().out
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "added 1 submission(s) from Kaggle's list" in out and "1/5; checked against Kaggle's list" in out
