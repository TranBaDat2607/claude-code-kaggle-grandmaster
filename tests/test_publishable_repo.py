"""A competition repo that can go public when the competition ends: the exact code of each record (snapshot commits
under refs/kg/), its command and phase, `ledger phase` / `ledger journey`, decisions over a range, the layout
`kgkit init` sets up and the drift checks, and the hook notes that keep the tree readable while the work happens."""

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from kgkit import cli
from kgkit import layout as LY
from kgkit import snapshot as SN
from kgkit import state as S
from kgkit.experiment import Ledger

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="needs git")
SH = shutil.which("sh")


def git(root, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "core.autocrlf=false",
                           "-c", "commit.gpgsign=false", *args], cwd=root, check=True, capture_output=True,
                          text=True).stdout


def commit(root, msg="change"):
    git(root, "add", "-A")
    git(root, "commit", "-qm", msg)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A workspace in git with one commit: src/train.py, the state, an ignored data/."""
    root = tmp_path / "comp"
    root.mkdir()
    git(root, "init", "-q")
    S.CompetitionState(slug="demo", metric="auc", deadline="2099-01-01").save(root)
    (root / ".gitignore").write_text("data/\nwork/\n")
    (root / "src").mkdir()
    (root / "src" / "train.py").write_text("print('v1')\n")
    commit(root, "init")
    for k in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(k, "t" if "NAME" in k else "t@example.com")
    monkeypatch.delenv("KG_NO_SNAPSHOT", raising=False)
    monkeypatch.chdir(root)
    return root


def _status(root):
    return git(root, "status", "--porcelain", "--untracked-files=all"), git(root, "diff", "--cached", "--name-only")


# ----------------------------------------------------------------------------- snapshots
def test_a_clean_tree_needs_no_snapshot(repo):
    info = SN.git_info(repo, "0001_a")
    assert info == {"git_commit": git(repo, "rev-parse", "--short", "HEAD").strip(), "git_dirty": False}
    assert SN.local_refs(repo) == []


def test_ledger_changes_alone_leave_the_tree_clean(repo):
    led = Ledger(repo)
    rec = led.log("a", 0.8)
    assert rec["git_dirty"] is False and "git_snapshot" not in rec  # the ledger file itself is not code
    rec = led.log("b", 0.81)
    assert rec["git_dirty"] is False and "git_snapshot" not in rec


def test_a_snapshot_holds_the_changed_and_new_files_and_touches_nothing(repo):
    (repo / "src" / "train.py").write_text("print('v2')\n")
    (repo / "src" / "feats.py").write_text("X = 1\n")
    (repo / "src" / "big.bin").write_bytes(b"\0" * (SN.MAX_FILE_BYTES + 1))
    (repo / "data").mkdir()
    (repo / "data" / "train.csv").write_text("id,y\n1,0\n")  # ignored: never in a snapshot
    (repo / ".kaggle-gm" / "scratch.txt").write_text("x")  # kgkit's state: not code
    git(repo, "add", "src/feats.py")  # a staged file stays staged, the rest stays unstaged
    before = _status(repo)
    info = SN.git_info(repo, "0001_v2")
    assert _status(repo) == before
    assert info["git_dirty"] is True and info["git_snapshot_skipped"] == 1
    sha = info["git_snapshot"]
    assert git(repo, "rev-parse", "refs/kg/0001_v2").strip() == sha
    assert git(repo, "rev-parse", f"{sha}^").strip() == git(repo, "rev-parse", "HEAD").strip()
    files = set(git(repo, "ls-tree", "-r", "--name-only", sha).split())
    assert {"src/train.py", "src/feats.py"} <= files and "src/big.bin" not in files and "data/train.csv" not in files
    assert ".kaggle-gm/scratch.txt" not in files
    assert git(repo, "show", f"{sha}:src/train.py") == "print('v2')\n"
    assert SN.code_ref({"git_snapshot": sha, "git_commit": "abc", "git_dirty": True}) == (sha, True)
    assert SN.code_ref({"git_commit": "abc1234", "git_dirty": True}) == ("abc1234", False)
    assert SN.code_ref({"git_commit": "abc1234", "git_dirty": False}) == ("abc1234", True)


def test_runs_of_the_same_code_share_a_snapshot(repo):
    (repo / "src" / "train.py").write_text("print('v2')\n")
    a = SN.git_info(repo, "0001_a")["git_snapshot"]
    b = SN.git_info(repo, "0002_b")["git_snapshot"]
    assert a == b and len(SN.local_refs(repo)) == 2
    (repo / "src" / "train.py").write_text("print('v3')\n")
    assert SN.git_info(repo, "0003_c")["git_snapshot"] != a


def test_the_workspace_folder_scopes_the_snapshot(repo):
    """A workspace in a folder of a bigger repository: changes outside it are not its code."""
    ws = repo / "comps" / "demo"
    ws.mkdir(parents=True)
    (ws / "train.py").write_text("print(1)\n")
    commit(repo, "ws")
    (repo / "src" / "train.py").write_text("print('elsewhere')\n")
    assert SN.git_info(ws, "0001_a") == {"git_commit": git(repo, "rev-parse", "--short", "HEAD").strip(),
                                         "git_dirty": False}
    (ws / "train.py").write_text("print(2)\n")
    sha = SN.git_info(ws, "0001_a")["git_snapshot"]
    assert git(repo, "show", f"{sha}:comps/demo/train.py") == "print(2)\n"
    assert git(repo, "show", f"{sha}:src/train.py") == "print('v1')\n"  # the change outside is not in it


def test_no_repository_or_no_commit_gives_nones(tmp_path):
    assert SN.git_info(tmp_path, "0001_a") == {"git_commit": None, "git_dirty": None}
    git(tmp_path, "init", "-q")
    (tmp_path / "x.py").write_text("1\n")
    assert SN.git_info(tmp_path, "0001_a") == {"git_commit": None, "git_dirty": None}
    assert SN.snapshot(tmp_path, "0001_a") == (None, 0)


def test_ref_names_are_valid():
    assert SN.ref_name("0001_lgbm") == "refs/kg/0001_lgbm"
    assert SN.ref_name("0002 my run: v2?") == "refs/kg/0002-my-run-v2"
    assert SN.ref_name("build/x.lock") == "refs/kg/build/x-lock"
    assert SN.ref_name("..") == "refs/kg/x"


def test_kg_no_snapshot_turns_snapshots_off(repo, monkeypatch):
    monkeypatch.setenv("KG_NO_SNAPSHOT", "1")
    (repo / "src" / "train.py").write_text("print('v2')\n")
    rec = Ledger(repo).log("a", 0.8)
    assert rec["git_dirty"] is True and "git_snapshot" not in rec and SN.local_refs(repo) == []


def test_audit_repo_reads_the_snapshot_refs(repo, capsys):
    """An untracked copy of competition data outside data/ goes into a snapshot: the audit, which reads every ref,
    finds it there before the refs are pushed."""
    from kgkit import audit as AU

    (repo / "data").mkdir()
    rows = "".join(f"{i},{i * 7919 % 1000},feature text {i}\n" for i in range(400))
    (repo / "data" / "train.csv").write_text("id,y,text\n" + rows)
    (repo / "src" / "train.py").write_text("print('v2')\n")
    (repo / "src" / "cache.csv").write_text("id,y,text\n" + rows)
    sha = SN.git_info(repo, "0001_a")["git_snapshot"]
    assert "src/cache.csv" in git(repo, "ls-tree", "-r", "--name-only", sha)
    (repo / "src" / "cache.csv").unlink()  # gone from the tree, still in the snapshot
    res = AU.run(repo, online=False)
    assert any(f["path"] == "src/cache.csv" for f in res["findings"]), res["findings"]


# ----------------------------------------------------------------------------- command, phase, decisions
def _script(root, body):
    (root / "src" / "run.py").write_text(
        "import sys\nfrom kgkit.experiment import Ledger\n" + body, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(Path(cli.__file__).resolve().parents[1])}
    return subprocess.run([sys.executable, "src/run.py", "--lr", "3e-4", "--note", "two words"], cwd=root, env=env,
                          capture_output=True, text=True, timeout=120)


def test_a_record_keeps_its_command_and_phase(repo):
    st = S.load(repo)
    st.phase = "baselines"
    st.save(repo)
    r = _script(repo, "rec = Ledger().log('lgbm', 0.8)\nprint(rec['cmd']); print(rec['phase'])\n")
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ["python src/run.py --lr 3e-4 --note 'two words'", "baselines"]
    rec = Ledger(repo).log("by hand", 0.7)  # logged from pytest: its argv is not this workspace's script
    assert rec["phase"] == "baselines"


def test_kgkit_cli_records_have_no_command(repo, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(Path(cli.__file__).with_name("__main__.py")), "ledger", "add", "x", "0.8"])
    cli.main(["ledger", "add", "x", "0.8", "from a notebook"])
    rec = Ledger(repo).records()[-1]
    assert rec["cmd"] is None and rec["phase"] is None


def test_ledger_phase_names_lists_and_reassigns(repo, capsys):
    led = Ledger(repo)
    for name in ("a", "b", "c"):
        led.log(name, 0.8)
    cli.main(["ledger", "phase"])
    out = capsys.readouterr().out
    assert "(no phase)" in out and "3 record(s)" in out and "no current phase" in out
    cli.main(["ledger", "phase", "gbdt", "baselines"])
    assert "phase is now 'gbdt baselines'" in capsys.readouterr().out
    assert S.load(repo).phase == "gbdt baselines"
    assert led.log("d", 0.81)["phase"] == "gbdt baselines"
    cli.main(["ledger", "phase", "early", "--records", "1..2"])
    assert "put 2 record(s) in phase 'early': 0001_a, 0002_b" in capsys.readouterr().out
    assert [r.get("phase") for r in led.records()] == ["early", "early", None, "gbdt baselines"]
    assert [(p["name"], p["n"]) for p in led.phases()] == [("early", 2), ("", 1), ("gbdt baselines", 1)]
    with pytest.raises(SystemExit, match="--records needs the phase"):
        cli.main(["ledger", "phase", "--records", "1..2"])
    with pytest.raises(SystemExit, match="no record numbered 7..9"):
        cli.main(["ledger", "phase", "x", "--records", "7..9"])


def test_ids_take_lists_ranges_and_names(repo):
    led = Ledger(repo)
    for name in ("a", "b", "c", "d"):
        led.log(name, 0.8)
    assert led.ids("2") == ["0002_b"]
    assert led.ids("0003_c") == ["0003_c"]
    assert led.ids("4..2") == ["0002_b", "0003_c", "0004_d"]
    assert led.ids("1, 3..9, 1") == ["0001_a", "0003_c", "0004_d"]
    with pytest.raises(KeyError):
        led.ids("12")


def test_decide_a_range_at_once(repo, capsys):
    led = Ledger(repo)
    for name in ("base", "x", "y", "z"):
        led.log(name, 0.8)
    cli.main(["ledger", "decide", "1", "baseline", "first"])
    capsys.readouterr()
    cli.main(["ledger", "decide", "2..4", "discard", "lr sweep: no gain", "--parent", "1"])
    out = capsys.readouterr().out
    assert out.count(": discard") == 3 and "accepted baseline is now" not in out
    recs = led.records()
    assert [r["decision"] for r in recs] == ["baseline", "discard", "discard", "discard"]
    assert {r["decision_note"] for r in recs[1:]} == {"lr sweep: no gain"} and {r["parent"] for r in recs[1:]} == {"0001_base"}
    with pytest.raises(SystemExit):
        cli.main(["ledger", "decide", "8..9", "keep"])


# ----------------------------------------------------------------------------- layout
def test_init_makes_work_and_studies_and_ignores_generated_folders(tmp_path, capsys):
    cli.main(["init", "demo", "--dir", str(tmp_path), "--metric", "auc"])
    assert (tmp_path / "work").is_dir() and (tmp_path / "studies").is_dir()
    assert "work/" in (tmp_path / ".gitignore").read_text().splitlines()
    gi = (tmp_path / ".kaggle-gm" / ".gitignore").read_text()
    assert "remote/" in gi.splitlines() and "build/" in gi.splitlines()
    S.load(tmp_path).save(tmp_path)  # saving again adds nothing twice
    assert (tmp_path / ".kaggle-gm" / ".gitignore").read_text() == gi


def test_state_ignores_complete_an_existing_file(tmp_path):
    d = tmp_path / ".kaggle-gm"
    d.mkdir()
    (d / ".gitignore").write_text("remote/\n*.tmp")
    assert S.ensure_state_ignores(tmp_path) == ["build/"]
    assert (d / ".gitignore").read_text().splitlines()[:2] == ["remote/", "*.tmp"]
    assert S.ensure_state_ignores(tmp_path) == []


def test_variants_and_log_targets():
    assert LY.VARIANT_RE.match("train_v2") and LY.VARIANT_RE.match("train (2)") and LY.VARIANT_RE.match("train - Copy")
    assert not LY.VARIANT_RE.match("train") and not LY.VARIANT_RE.match("make_folds")
    assert LY.log_targets("python src/train.py > train.log 2>&1") == ["train.log"]
    assert LY.log_targets("python a.py 2>&1 | tee -a reports/run.out; python b.py >> ./logs/b.log") == [
        "reports/run.out", "logs/b.log"]
    assert LY.log_targets("python a.py > work/logs/a.log; python b.py > .kaggle-gm/x.log; c > /tmp/c.log") == []
    assert LY.log_targets("python a.py | Tee-Object -FilePath 'logs\\a.log'") == ["logs/a.log"]
    assert LY.log_targets("python a.py > C:\\temp\\a.log; python b.py > $env:TEMP/b.log") == []
    assert LY.log_targets("python a.py > out.csv") == []


def test_variant_of_needs_the_script_beside_it(tmp_path):
    (tmp_path / "train.py").write_text("")
    for name in ("train_v2.py", "train_new.py", "train-copy.py", "train (2).py", "train_3.py"):
        (tmp_path / name).write_text("")
        assert LY.variant_of(tmp_path / name) == tmp_path / "train.py", name
    (tmp_path / "infer_v2.py").write_text("")
    assert LY.variant_of(tmp_path / "infer_v2.py") is None and LY.variant_of(tmp_path / "train.py") is None


def test_drift_names_copies_crowding_logs_and_generated_files(repo):
    assert LY.drift(repo) == [] and LY.brief_line(repo) == ""
    src = repo / "src"
    (src / "train_v2.py").write_text("")
    for i in range(LY.SRC_SCRIPTS_MAX):
        (src / f"probe{chr(97 + i)}.py").write_text("")
    (repo / "run.log").write_text("x")
    (repo / ".kaggle-gm" / "build" / "a-train").mkdir(parents=True)
    (repo / ".kaggle-gm" / "build" / "a-train" / "a_train.py").write_text("x")
    git(repo, "add", "-f", "run.log", ".kaggle-gm/build")
    (repo / ".gitignore").write_text("data/\n")
    (repo / "work").mkdir()
    found = LY.drift(repo)
    assert len(found) == 5, found
    assert "train_v2.py (of train.py)" in found[0] and "src/ has 17 scripts" in found[1]
    assert "1 log file(s) tracked by git (e.g. run.log)" in found[2]
    assert "under .kaggle-gm/build tracked by git" in found[3] and "work/ is not ignored" in found[4]
    assert LY.brief_line(repo).startswith("Repo layout (it goes public when the competition ends, /kg-showcase): ")


def test_journey_lines_ask_for_a_phase_and_decisions():
    st = S.CompetitionState(slug="d")
    recs = [{"id": f"000{i}_x"} for i in range(1, 4)]
    lines = LY.journey_lines(st, recs)
    assert lines[0].startswith("No phase named") and "3 record(s) have no decision (oldest 0001_x)" in lines[1]
    st.phase = "stacking"
    recs[2]["phase"] = "stacking"
    for r in recs:
        r["decision"] = "keep"
    lines = LY.journey_lines(st, recs)
    assert len(lines) == 1 and lines[0].startswith('Phase: "stacking" (1 record(s) so far)')
    assert LY.journey_lines(S.CompetitionState(slug="d"), recs[:2]) == []


def test_status_prints_the_phase_and_the_drift(repo, capsys):
    led = Ledger(repo)
    for name in ("a", "b", "c"):
        led.log(name, 0.8)
    (repo / "src" / "train_v2.py").write_text("")
    cli.main(["status", "--offline"])
    out = capsys.readouterr().out
    assert "No phase named" in out and "3 record(s) have no decision" in out
    assert "Repo layout (it goes public" in out and "train_v2.py (of train.py)" in out


# ----------------------------------------------------------------------------- hooks
def _hook(plugin_root, script, payload):
    r = subprocess.run([SH, str(plugin_root / "hooks" / "run.sh"), script], input=json.dumps(payload),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    out = r.stdout.strip()
    return json.loads(out)["hookSpecificOutput"] if out else None


@pytest.mark.skipif(SH is None, reason="needs a POSIX sh")
def test_post_write_notes_a_copy_of_a_script_once(plugin_root, repo):
    sid = uuid.uuid4().hex
    (repo / "src" / "train_v2.py").write_text("print('copy')\n")
    payload = {"hook_event_name": "PostToolUse", "tool_name": "Write", "session_id": sid, "cwd": str(repo),
               "tool_input": {"file_path": str(repo / "src" / "train_v2.py"), "content": "print('copy')\n"}}
    out = _hook(plugin_root, "post_write.py", payload)
    assert out and "src/train_v2.py looks like a copy of train.py" in out["additionalContext"]
    assert _hook(plugin_root, "post_write.py", payload) is None  # once per file and session
    payload["tool_name"] = "Edit"
    payload["session_id"] = uuid.uuid4().hex
    assert _hook(plugin_root, "post_write.py", payload) is None  # an edit of an existing copy: no note


@pytest.mark.skipif(SH is None, reason="needs a POSIX sh")
def test_post_tool_notes_a_log_outside_work_once(plugin_root, repo):
    sid = uuid.uuid4().hex

    def run(cmd):
        return _hook(plugin_root, "post_tool.py", {"hook_event_name": "PostToolUse", "tool_name": "Bash",
                                                   "session_id": sid, "cwd": str(repo),
                                                   "tool_input": {"command": cmd}, "tool_response": {"stdout": ""}})

    assert run("python src/train.py > work/logs/a.log 2>&1") is None
    out = run("python src/train.py > reports/a.log 2>&1")
    assert out and "This command writes reports/a.log: logs go in work/" in out["additionalContext"]
    assert run("python src/train.py > b.log") is None  # once per session


@pytest.mark.skipif(SH is None, reason="needs a POSIX sh")
def test_session_brief_asks_for_a_phase_and_shows_the_drift(plugin_root, repo):
    led = Ledger(repo)
    for name in ("a", "b", "c"):
        led.log(name, 0.8)
    (repo / "src" / "train_v2.py").write_text("")
    ctx = _hook(plugin_root, "session_start.py", {"hook_event_name": "SessionStart", "cwd": str(repo)})[
        "additionalContext"]
    assert "No phase named" in ctx and "3 record(s) have no decision" in ctx
    assert "Repo layout (it goes public when the competition ends" in ctx
