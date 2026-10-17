"""Discussions from a (fake, offline) Meta Kaggle cache and a faked live Kaggle CLI, and the fresh-split recheck
end to end."""

import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from kgkit import cli
from kgkit import discussions as D
from kgkit.experiment import Ledger


def run(args, cwd, plugin_root, env_extra=None, check=True):
    env = dict(os.environ, PYTHONPATH=str(plugin_root), KGKIT_HOME=str(plugin_root), **(env_extra or {}))
    r = subprocess.run([sys.executable, "-m", "kgkit", *args], cwd=cwd, env=env, capture_output=True, text=True,
                       timeout=900)
    if check and r.returncode != 0:
        raise AssertionError(r.stdout[-3000:] + r.stderr[-3000:])
    return r


@pytest.fixture
def meta(tmp_path):
    d = tmp_path / "meta"
    d.mkdir()
    pd.DataFrame({
        "Id": [1, 2, 3, 4], "Slug": ["comp-a", "comp-b", "new-comp", "live-comp"], "Title": ["A", "B", "N", "L"],
        "ForumId": [100, 200, np.nan, np.nan],
        "DeadlineDate": ["01/01/2026 23:59:00"] * 3 + ["12/31/2099 23:59:00"],  # live-comp has not ended
        "TeamMergerDeadlineDate": ["12/25/2025 23:59:00"] * 4, "MaxTeamSize": [5] * 4,
        "BanTeamMergers": [False] * 4, "MaxDailySubmissions": [5] * 4,
        "EvaluationAlgorithmName": ["Roc Auc Score"] * 4, "HostSegmentTitle": ["Featured"] * 4,
    }).to_csv(d / "Competitions.csv", index=False)
    pd.DataFrame({
        "Id": [10, 11, 12, 13, 14], "ForumId": [100, 100, 100, 100, 200], "KernelId": [np.nan, np.nan, 999, np.nan, np.nan],
        "LastForumMessageId": 0, "FirstForumMessageId": 0,
        "CreationDate": ["01/05/2026 10:00:00", "01/02/2026 10:00:00", "01/03/2026 10:00:00", "01/04/2026 08:00:00",
                         "01/01/2026 00:00:00"],
        "LastCommentDate": "01/06/2026 00:00:00",
        "Title": ["1st place solution", "Possible leak in row order?", "", "CV vs LB gap discussion", "other comp"],
        "IsSticky": False, "TotalViews": [900, 500, 1, 300, 5], "Score": [120, 45, 3, 30, 9],
        "TotalMessages": [20, 12, 1, 8, 1], "TotalReplies": [19, 11, 0, 7, 0],
    }).to_csv(d / "ForumTopics.csv", index=False)
    pd.DataFrame({
        "Id": [1, 2, 3], "ForumTopicId": [11, 11, 13],
        "PostDate": ["01/02/2026 11:00:00", "01/02/2026 10:00:00", "01/04/2026 09:00:00"],
        "Message": ["<p>reply: confirmed, <b>shuffle</b> fixes it</p>", "<p>The id column leaks the target &amp; order</p>",
                    "x"],
        "Medal": [np.nan, 2, np.nan],
    }).to_csv(d / "ForumMessages.csv", index=False)
    return d


def test_index_top_search_solutions_read(meta, monkeypatch):
    monkeypatch.setenv("KGKIT_META_DIR", str(meta))
    t = D.index("comp-a")
    assert sorted(t["Id"]) == [10, 11, 13]  # notebook comment thread (KernelId) and other forums excluded
    assert t.loc[t["Id"] == 10, "Url"].iloc[0] == "https://www.kaggle.com/competitions/comp-a/discussion/10"
    assert list(D.top(t, 2, "votes")["Id"]) == [10, 11]
    assert list(D.top(t, 1, "recent")["Id"]) == [10]
    assert list(D.search(t, "leak|cv.*lb")["Id"]) == [11, 13]
    assert list(D.solutions(t)["Id"]) == [10]
    msgs = D.read_thread(11, meta)
    assert [m["Id"] for m in msgs] == [2, 1]  # chronological
    assert msgs[0]["text"] == "The id column leaks the target & order"
    assert "facts" not in D.facts(D.competition_row("comp-a", meta)) and "merger deadline" in D.facts(
        D.competition_row("comp-a", meta))
    with pytest.raises(SystemExit, match="no forum id .*not in the export"):
        D.index("new-comp")
    with pytest.raises(SystemExit, match="hides the forum id of competitions that have not ended.*--source live"):
        D.index("live-comp")
    with pytest.raises(SystemExit, match="not in Meta Kaggle"):
        D.index("missing")


def test_discussions_cli(meta, tmp_path, plugin_root):
    ws = tmp_path / "ws"
    ws.mkdir()
    run(["init", "comp-a", "--metric", "auc"], ws, plugin_root)
    env = {"KGKIT_META_DIR": str(meta)}  # files are fresh, so no download is attempted
    r = run(["discussions", "sync", "--source", "meta"], ws, plugin_root, env)
    assert "comp-a: 3 topics" in r.stdout and "merger deadline: 12/25/2025" in r.stdout
    assert (ws / "reports" / "discussions.csv").is_file()
    r = run(["discussions", "search", "leak", "--source", "meta"], ws, plugin_root, env)
    assert "Possible leak in row order?" in r.stdout and "1st place" not in r.stdout
    r = run(["discussions", "solutions", "--source", "meta"], ws, plugin_root, env)
    assert "1st place solution" in r.stdout
    r = run(["discussions", "top", "--sort", "replies", "-n", "1", "--source", "meta"], ws, plugin_root, env)
    assert "1st place solution" in r.stdout
    r = run(["discussions", "read", "11", "--source", "meta"], ws, plugin_root, env)
    assert "shuffle fixes it" in r.stdout and 'topic="11"' in r.stdout and 'medal="2"' in r.stdout
    assert len(re.findall(r"<untrusted-content-[0-9a-f]{8} ", r.stdout)) == r.stdout.count("</untrusted-content-") >= 1
    r = run(["discussions", "top", "--competition", "comp-b", "--source", "meta"], ws, plugin_root, env)
    assert "other comp" in r.stdout
    os.utime(meta / "ForumMessages.csv", (0, 0))  # stale cache + no kaggle CLI on PATH -> falls back, no crash
    (meta / "ForumMessages.csv").unlink()
    r = run(["discussions", "read", "11", "--source", "meta"], ws, plugin_root, env)
    assert "not cached" in r.stdout and "/discussion/11" in r.stdout


LIVE_TOPICS = [{"id": 100 + i, "title": f"topic {i}" + (" leak" if i == 3 else ""), "authorName": "",
                "commentCount": i, "votes": 50 - i, "postDate": f"2026-10-{i + 1:02d}T10:00:00.123000"}
               for i in range(25)]
LIVE_MESSAGES = [{"id": 1, "authorName": "", "postDate": "2026-10-01T10:00:00", "votes": 3,
                  "content": "<p>Opening post \U0001f680 &amp; data</p>"},
                 {"id": 2, "authorName": "", "postDate": "2026-10-02T10:00:00", "votes": 1, "content": "<p>reply</p>"}]


def fake_kaggle(calls, topics=LIVE_TOPICS, rc=0, err=""):
    """subprocess.run stand-in printing what Kaggle CLI 2.2.4 prints: a warning, the JSON, a page-token line."""
    def run_(args, **kw):
        calls.append(args)
        assert kw["env"]["PYTHONIOENCODING"] == "utf-8" and kw["encoding"] == "utf-8"
        if rc:
            return subprocess.CompletedProcess(args, rc, "", err)
        if args[1:3] == ["competitions", "topics"]:
            page = int(args[args.index("-p") + 1])
            rows = topics[(page - 1) * 20: page * 20]
            more = f"Next Page Token = {page + 1}\n" if page * 20 < len(topics) else ""
            out = ("Warning: --page-size and --search are not supported\n" + json.dumps(rows, indent=2) + "\n" + more
                   if rows else "No topics found\n")
            return subprocess.CompletedProcess(args, 0, out, "")
        if args[1:3] == ["competitions", "topic-messages"]:
            return subprocess.CompletedProcess(args, 0, json.dumps(LIVE_MESSAGES), "")
        raise AssertionError(f"unexpected kaggle call {args}")
    return run_


def test_live_index_and_thread(monkeypatch):
    calls = []
    monkeypatch.setattr(D.shutil, "which", lambda _: "kaggle")
    monkeypatch.setattr(D.subprocess, "run", fake_kaggle(calls))
    t = D.live_index("live-comp")
    assert len(t) == 25 and len(calls) == 2 and not t.attrs["truncated"]  # stops on the short second page
    assert calls[0][1:] == ["competitions", "topics", "list", "live-comp", "--format", "json", "-s", "top", "-p", "1"]
    assert list(D.top(t, 2, "votes")["Id"]) == [100, 101] and list(D.top(t, 1, "recent")["Id"]) == [124]
    assert list(D.search(t, "leak")["Id"]) == [103] and t["Url"].iloc[0].endswith("/live-comp/discussion/100")
    with pytest.raises(SystemExit, match="no view counts"):
        D.top(t, 5, "views")
    assert D.live_index("live-comp", max_pages=1).attrs["truncated"]
    msgs = D.live_thread("live-comp", 100)
    assert calls[-1][1:] == ["competitions", "topic-messages", "live-comp", "100", "--format", "json", "-n", "-1",
                             "-s", "old"]
    assert [m["text"] for m in msgs] == ["Opening post \U0001f680 & data", "reply"]
    monkeypatch.setattr(D.subprocess, "run", fake_kaggle(calls, topics=[]))
    assert D.live_index("live-comp").empty
    monkeypatch.setattr(D.subprocess, "run", fake_kaggle(calls, rc=2, err="argument command: invalid choice: 'topics'"))
    with pytest.raises(D.LiveUnavailable, match="2.2.4"):
        D.live_index("live-comp")
    monkeypatch.setattr(D.shutil, "which", lambda _: None)
    with pytest.raises(D.LiveUnavailable, match="not found"):
        D.live_thread("live-comp", 1)


def test_discussions_cli_live_and_fallback(meta, tmp_path, monkeypatch, capsys):
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    monkeypatch.setenv("KGKIT_META_DIR", str(meta))
    cli.main(["init", "live-comp", "--metric", "auc"])
    calls = []
    monkeypatch.setattr(D.shutil, "which", lambda _: "kaggle")
    monkeypatch.setattr(D.subprocess, "run", fake_kaggle(calls))
    capsys.readouterr()
    cli.main(["discussions", "sync"])
    out = capsys.readouterr().out
    assert "live-comp: 25 topics" in out and "(live, Kaggle CLI)" in out and 'source="kaggle-cli"' in out
    assert "merger deadline: 12/25/2025" in out  # facts from the already cached export, nothing downloaded
    assert pd.read_csv(ws / "reports" / "discussions.csv")["Source"].eq("live").all()
    cli.main(["discussions", "search", "leak"])
    out = capsys.readouterr().out
    assert "topic 3 leak" in out and 'source="kaggle-cli"' in out
    cli.main(["discussions", "read", "100"])
    out = capsys.readouterr().out
    assert "Opening post" in out and 'topic="100"' in out and "/live-comp/discussion/100" in out

    # no forum commands in this CLI: auto falls back to Meta Kaggle, which stops before the topic download
    monkeypatch.setattr(D.subprocess, "run", fake_kaggle(calls, rc=2, err="invalid choice: 'topics'"))
    fetched = []
    monkeypatch.setattr(D, "ensure", lambda files, **kw: fetched.extend(files) or meta)
    with pytest.raises(SystemExit, match="hides the forum id"):
        cli.main(["discussions", "sync"])
    assert "falling back to Meta Kaggle" in capsys.readouterr().out and fetched == ["Competitions.csv"]
    with pytest.raises(SystemExit, match="2.2.4"):
        cli.main(["discussions", "sync", "--source", "live"])
    fetched.clear()
    cli.main(["discussions", "sync", "--competition", "comp-a"])  # a finished competition still works from the export
    assert "comp-a: 3 topics" in capsys.readouterr().out and fetched == ["Competitions.csv", "ForumTopics.csv"]


# ----------------------------------------------------------------------------- recheck
def test_recheck_end_to_end(tmp_path, plugin_root, binary_df):
    ws = tmp_path
    run(["init", "demo", "--metric", "auc", "--target", "target", "--id-col", "id"], ws, plugin_root)
    binary_df.to_csv(ws / "data" / "train.csv", index=False)
    test = binary_df.drop(columns="target").iloc[:200].copy()
    test["id"] = np.arange(50_000, 50_200)
    test.to_csv(ws / "data" / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "target": 0.5}).to_csv(ws / "data" / "sample_submission.csv", index=False)
    run(["folds", "data/train.csv", "--target", "target", "--id-col", "id", "--out", "data/folds.csv"], ws, plugin_root)
    assert (ws / ".kaggle-gm" / "folds_spec.json").is_file()
    (ws / "src").mkdir(exist_ok=True)
    shutil.copy(plugin_root / "templates" / "train_gbdt.py", ws / "src" / "train_gbdt.py")
    py = f'"{sys.executable}"'
    env = dict(os.environ, PYTHONPATH=str(plugin_root), KGKIT_HOME=str(plugin_root))
    for name, model in (("hgb", "hgb"), ("lin", "linear")):  # old, current (linear wins on this linear-logit data)
        subprocess.run([sys.executable, "src/train_gbdt.py", "--model", model, "--name", name,
                        "--params", '{"max_iter": 100}' if model == "hgb" else "{}"], cwd=ws, env=env, check=True,
                       capture_output=True)
    led = Ledger(ws)
    frozen_hash = led.records()[-1]["folds_hash"]
    r = run(["recheck", "--seed", "11",
             "--run", f'{py} src/train_gbdt.py --model hgb --name hgb_rc --params "{{\\"max_iter\\": 100}}"',
             "--run", f"{py} src/train_gbdt.py --model linear --name lin_rc",
             "--exp", "0001", "0002", "--truth", "data/train.csv:target", "--boot", "100"], ws, plugin_root)
    assert (ws / "data" / "folds_recheck_s11.csv").is_file()
    assert "of the original gain survived" in r.stdout and "the gain holds" in r.stdout
    recs = led.records()
    rc = [x for x in recs if x.get("recheck")]
    assert [x["name"] for x in rc] == ["hgb_rc", "lin_rc"]
    assert all(x["folds_hash"] != frozen_hash for x in rc)  # hashed from the re-drawn split
    assert led.baseline()["id"] not in {x["id"] for x in rc}
    assert all(x["id"] not in {b["id"] for b in led.best(10)} for x in rc)
    assert (ws / "reports" / "recheck_s11.md").is_file()
    # same split twice is refused (deterministic recipe)
    spec = (ws / ".kaggle-gm" / "folds_spec.json").read_text(encoding="utf-8")
    assert '"seed": 42' in spec
    r = run(["recheck", "--seed", "42"], ws, plugin_root, check=False)
    assert r.returncode != 0 and "identical split" in (r.stdout + r.stderr)


def test_recheck_report_verdicts():
    from kgkit.recheck import report

    def rec(i, cv, folds):
        return {"id": i, "cv": cv, "fold_scores": folds}

    old, new = rec("o", 0.80, [0.80] * 5), rec("n", 0.82, [0.82] * 5)
    shrunk = report(rec("orc", 0.80, [0.80, 0.81, 0.79, 0.80, 0.80]), rec("nrc", 0.801, [0.80, 0.812, 0.791, 0.802, 0.80]),
                    True, (old, new))
    assert "did NOT survive" in shrunk or "shrank by more than half" in shrunk
    gone = report(rec("orc", 0.80, [0.80, 0.81, 0.79, 0.80, 0.80]), rec("nrc", 0.79, [0.79, 0.80, 0.78, 0.79, 0.79]),
                  True, (old, new))
    assert "did NOT survive" in gone
    none = report(rec("orc", 0.80, [0.80] * 5), rec("nrc", 0.81, [0.81] * 5), True, (new, old))
    assert "nothing to recheck" in none
