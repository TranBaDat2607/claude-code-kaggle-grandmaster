"""Discussions from a (fake, offline) Meta Kaggle cache, and the fresh-split recheck end to end."""

import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

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
        "Id": [1, 2, 3], "Slug": ["comp-a", "comp-b", "new-comp"], "Title": ["A", "B", "N"],
        "ForumId": [100, 200, np.nan], "DeadlineDate": ["01/01/2026 23:59:00"] * 3,
        "TeamMergerDeadlineDate": ["12/25/2025 23:59:00"] * 3, "MaxTeamSize": [5, 5, 5],
        "BanTeamMergers": [False] * 3, "MaxDailySubmissions": [5, 5, 5],
        "EvaluationAlgorithmName": ["Roc Auc Score"] * 3, "HostSegmentTitle": ["Featured"] * 3,
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
    with pytest.raises(SystemExit, match="no forum id"):
        D.index("new-comp")
    with pytest.raises(SystemExit, match="not in Meta Kaggle"):
        D.index("missing")


def test_discussions_cli(meta, tmp_path, plugin_root):
    ws = tmp_path / "ws"
    ws.mkdir()
    run(["init", "comp-a", "--metric", "auc"], ws, plugin_root)
    env = {"KGKIT_META_DIR": str(meta)}  # files are fresh, so no download is attempted
    r = run(["discussions", "sync"], ws, plugin_root, env)
    assert "comp-a: 3 topics" in r.stdout and "merger deadline: 12/25/2025" in r.stdout
    assert (ws / "reports" / "discussions.csv").is_file()
    r = run(["discussions", "search", "leak"], ws, plugin_root, env)
    assert "Possible leak in row order?" in r.stdout and "1st place" not in r.stdout
    r = run(["discussions", "solutions"], ws, plugin_root, env)
    assert "1st place solution" in r.stdout
    r = run(["discussions", "top", "--sort", "replies", "-n", "1"], ws, plugin_root, env)
    assert "1st place solution" in r.stdout
    r = run(["discussions", "read", "11"], ws, plugin_root, env)
    assert "shuffle fixes it" in r.stdout and "[medal 2]" in r.stdout
    r = run(["discussions", "top", "--competition", "comp-b"], ws, plugin_root, env)
    assert "other comp" in r.stdout
    os.utime(meta / "ForumMessages.csv", (0, 0))  # stale cache + no kaggle CLI on PATH -> falls back, no crash
    (meta / "ForumMessages.csv").unlink()
    r = run(["discussions", "read", "11"], ws, plugin_root, env)
    assert "not cached" in r.stdout and "/discussion/11" in r.stdout


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
