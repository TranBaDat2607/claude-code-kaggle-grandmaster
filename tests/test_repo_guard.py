"""Never-public files: the .gitignore `kgkit init` writes, and the guard that checks a commit or a push.

The .gitignore names only what is never code (data, outputs, weights, logs, credentials, caches). The files that
/kg-showcase removes at cleanup (copies of scripts, probes, scratch notebooks) stay tracked: experiment snapshots
(``refs/kg/*``) leave ignored files out, and the journey links to the code that ran."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from kgkit import cli
from kgkit import repoguard as RG

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

NEVER_CODE = ["data/train.csv", "data/ext/kout-a/model.pt", "artifacts/oof.npy", "subs/sub_001.csv",
              "src/feat.pkl", "models/fold0.safetensors", "work/probe.py", "train.log", "src/run.log",
              "kaggle.json", ".env", "wandb/run-1/x", "src/__pycache__/a.pyc", ".claude/settings.local.json",
              "comp.zip", "features/train.parquet"]
KEPT = ["src/train.py", "src/train_v2.py", "notebooks/scratch.ipynb", "probe_lr.py", "studies/s1/README.md",
        "reports/journal.md", ".env.example", ".kaggle-gm/ledger.jsonl", "kernels/k1/kernel-metadata.json"]


def git(root, *args, check=True):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false",
                           *args], cwd=root, check=check, capture_output=True, text=True)


def ignored(root, paths):
    r = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"], cwd=root,
                       input=("\n".join(paths) + "\n").encode(), capture_output=True)  # bytes: no CRLF on Windows
    return set(r.stdout.decode().split())


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "comp"
    root.mkdir()
    git(root, "init", "-q")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "no-global-gitconfig"))  # the repo's .gitignore must do it
    monkeypatch.chdir(root)
    return root


def _init(root, *extra):
    cli.main(["init", "knee", "--metric", "auc", "--dir", str(root), *extra])


def test_init_ignores_what_is_never_code_and_keeps_cleanup_files_tracked(repo):
    _init(repo)
    assert ignored(repo, NEVER_CODE) == set(NEVER_CODE)
    assert ignored(repo, KEPT) == set()  # the journey and the snapshots need them until /kg-showcase removes them
    text = (repo / ".gitignore").read_text(encoding="utf-8")
    assert "# Competition data" in text and "# Credentials" in text


def test_init_completes_an_existing_gitignore_once(repo, capsys):
    (repo / ".gitignore").write_bytes(b"data/train.csv\r\n# mine\r\n*.log\r\nnode_modules/")
    _init(repo)
    out = capsys.readouterr().out
    raw = (repo / ".gitignore").read_bytes()
    assert raw.startswith(b"data/train.csv\r\n# mine\r\n*.log\r\nnode_modules/\r\n")  # theirs first, CRLF kept
    assert b"\n" not in raw.replace(b"\r\n", b"")
    assert raw.count(b"*.log") == 1  # already there: not added again
    assert ignored(repo, NEVER_CODE) == set(NEVER_CODE) and ignored(repo, KEPT) == set()
    assert "to .gitignore" in out
    _init(repo, "--force")
    assert (repo / ".gitignore").read_bytes() == raw


def test_ensure_gitignore_reports_what_it_added(repo):
    (repo / ".gitignore").write_text("/data/\n/work/\n*.log\n")
    added = RG.ensure_gitignore(repo)
    assert "/data/" not in added and "*.log" not in added and "kaggle.json" in added
    assert RG.ensure_gitignore(repo) == []


# ------------------------------------------------------------------------------------------------ the guard
TRAIN = "id,text,label\n" + "".join(f"{i},report {i} shows a tear,{i % 2}\n" for i in range(50))
SAMPLE = "id,label\n" + "".join(f"t{i},0.5\n" for i in range(10))
KEY = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def ws(repo):
    """A workspace after `kgkit init`, with the competition's files in data/ and one clean commit."""
    _init(repo)
    (repo / "data" / "train.csv").write_text(TRAIN)
    (repo / "data" / "sample_submission.csv").write_text(SAMPLE)
    (repo / "src" / "train.py").write_text("print('train')\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "init")
    return repo


def _paths(lines):
    return {x.split(": ", 1)[0] for x in lines}


def test_guard_refuses_credentials_and_competition_data(ws):
    (ws / "kaggle.json").write_text('{"username": "me", "key": "%s"}' % KEY)
    (ws / "src" / "util.py").write_text(f"import os\nos.environ['KAGGLE_KEY'] = '{KEY}'\n")
    (ws / "src" / "copy.csv").write_text(TRAIN)  # an exact copy under another name
    (ws / "src" / "train.py").write_text("print('train v2')\n")
    git(ws, "add", "-f", "kaggle.json", "data/train.csv", "src/util.py", "src/copy.csv", "src/train.py")
    refused, asked = RG.check(ws, RG.staged(ws))
    assert _paths(refused) == {"kaggle.json", "data/train.csv", "src/util.py", "src/copy.csv"}
    why = dict(x.split(": ", 1) for x in refused)
    assert "credentials" in why["kaggle.json"]
    assert "Kaggle API key" in why["src/util.py"] and "line 2" in why["src/util.py"]
    assert "an exact copy of data/train.csv" in why["src/copy.csv"] and "competition" in why["data/train.csv"]
    assert KEY not in "\n".join(refused)  # never the secret itself
    assert asked == []


def test_guard_asks_about_other_never_public_files_and_passes_cleanup_files(ws):
    for rel, text in {"artifacts/oof.npy": "x", "models/fold0.pt": "x", "train.log": "epoch 1",
                      "data/ext/kout-a/notes.txt": "mine", "submission.csv": SAMPLE.replace("0.5", "0.7"),
                      ".kaggle-gm/remote/k/out.txt": "x"}.items():
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text(text)
    for rel in ("src/train_v2.py", "notebooks/scratch.ipynb", "probe_lr.py", ".env.example"):
        (ws / rel).write_text("KAGGLE_KEY=\n")
    git(ws, "add", "-A")
    git(ws, "add", "-f", "artifacts/oof.npy", "models/fold0.pt", "train.log", "data/ext/kout-a/notes.txt",
        ".kaggle-gm/remote/k/out.txt")
    refused, asked = RG.check(ws, RG.staged(ws))
    assert refused == []
    assert _paths(asked) == {"artifacts/oof.npy", "models/fold0.pt", "train.log", "data/ext/kout-a/notes.txt",
                             "submission.csv", ".kaggle-gm/remote/k/out.txt"}
    why = dict(x.split(": ", 1) for x in asked)
    assert "sample_submission.csv" in why["submission.csv"] and "*.pt" in why["models/fold0.pt"]


def test_guard_sees_what_git_add_would_stage(ws):
    (ws / "src" / "copy.csv").write_text(TRAIN)
    (ws / "src" / "ok.py").write_text("x = 1\n")
    items = RG.would_add(ws, ["-A"])
    assert {i.path for i in items} == {"src/copy.csv", "src/ok.py"}
    assert _paths(RG.check(ws, items)[0]) == {"src/copy.csv"}
    assert {i.path for i in RG.would_add(ws, ["-f", "data/train.csv"])} == {"data/train.csv"}
    assert RG.would_add(ws, ["data/train.csv"]) == []  # ignored, and git would refuse it


def test_a_push_checks_every_file_its_commits_send(ws, tmp_path):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    git(ws, "remote", "add", "origin", str(remote))
    git(ws, "push", "-q", "origin", "HEAD:refs/heads/main")
    assert RG.check(ws, RG.pushed(ws, ["HEAD"], "origin")) == ([], [])  # all of it is on the remote already
    (ws / "src" / "tok.txt").write_text(f'{{"key": "{KEY}"}}\n')
    git(ws, "add", "src/tok.txt")
    git(ws, "commit", "-qm", "oops", "--no-verify")  # the pre-commit hook would refuse it
    git(ws, "rm", "-q", "src/tok.txt")
    git(ws, "commit", "-qm", "remove it")  # gone from the tip, still in a commit the push sends
    refused, _ = RG.check(ws, RG.pushed(ws, ["HEAD"], "origin"))
    assert _paths(refused) == {"src/tok.txt"}


def test_repo_guard_check_exits_1_on_a_refusal_and_0_with_warnings(ws, capsys):
    (ws / "train.log").write_text("epoch 1\n")
    git(ws, "add", "-f", "train.log")
    cli.main(["repo-guard", "check", "--staged"])
    out = capsys.readouterr()
    assert "train.log" in out.err and "never-public" in out.err
    (ws / "kaggle.json").write_text("{}")
    git(ws, "add", "-f", "kaggle.json")
    with pytest.raises(SystemExit) as e:
        cli.main(["repo-guard", "check", "--staged"])
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert "kaggle.json" in err and "git rm --cached kaggle.json" in err


# ------------------------------------------------------------------------------- the PreToolUse hook (Claude's git)
def _hook(plugin_root, cmd, cwd):
    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("needs a POSIX sh")
    payload = {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": str(cwd)}
    r = subprocess.run([sh, str(plugin_root / "hooks" / "run.sh"), "pre_tool.py"], input=json.dumps(payload),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    out = r.stdout.strip()
    return json.loads(out)["hookSpecificOutput"] if out else None


def test_hook_refuses_a_commit_that_would_send_a_token_or_the_data(ws, plugin_root):
    (ws / "src" / "cfg.py").write_text(f'KAGGLE_KEY = "{KEY}"\n')
    out = _hook(plugin_root, 'git add src/cfg.py && git commit -m "config"', ws)
    assert out["permissionDecision"] == "deny"
    assert "src/cfg.py" in out["permissionDecisionReason"] and KEY not in out["permissionDecisionReason"]
    out = _hook(plugin_root, "cd src && git add -f ../data/train.csv", ws)  # the cd is followed
    assert out["permissionDecision"] == "deny" and "data/train.csv" in out["permissionDecisionReason"]
    (ws / "data" / "train.csv").rename(ws / "data" / "kept.csv")
    git(ws, "add", "-f", "data/kept.csv")  # staged earlier, outside Claude Code
    out = _hook(plugin_root, "git commit -qm wip", ws)
    assert out["permissionDecision"] == "deny" and "data/kept.csv" in out["permissionDecisionReason"]


def test_hook_asks_before_other_never_public_files_and_says_nothing_otherwise(ws, plugin_root):
    (ws / "models").mkdir()
    (ws / "models" / "best.pt").write_text("w")
    out = _hook(plugin_root, "git add -f models/best.pt", ws)
    assert out["permissionDecision"] == "ask" and "models/best.pt" in out["permissionDecisionReason"]
    (ws / "src" / "train_v2.py").write_text("print(2)\n")  # a cleanup file: tracked until /kg-showcase
    assert _hook(plugin_root, "git add -A && git commit -m v2", ws) is None
    assert _hook(plugin_root, 'git log --oneline -3 && echo "git add -f data/train.csv"', ws) is None


def test_hook_checks_what_a_push_sends(ws, plugin_root, tmp_path):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    git(ws, "remote", "add", "origin", str(remote))
    git(ws, "push", "-q", "origin", "HEAD:refs/heads/main")
    (ws / "notes.md").write_text(f"token: {{'key': '{KEY}'}}\n")
    git(ws, "add", "notes.md")
    git(ws, "commit", "-qm", "notes", "--no-verify")
    for cmd in ("git push", "git push origin HEAD:main", "git push -u origin HEAD"):
        out = _hook(plugin_root, cmd, ws)
        assert out["permissionDecision"] == "deny" and "notes.md" in out["permissionDecisionReason"], cmd
    git(ws, "reset", "-q", "--hard", "HEAD~1")
    assert _hook(plugin_root, "git push origin HEAD:main", ws) is None


# ------------------------------------------------------------------------- the git hooks (the user's own commits)
def _hooks(root):
    return root / ".git" / "hooks"


def test_init_installs_git_hooks_that_refuse_data_and_warn_about_outputs(ws):
    for name in ("pre-commit", "pre-push"):
        f = _hooks(ws) / name
        assert RG.HOOK_MARK in f.read_text() and (os.name == "nt" or os.access(f, os.X_OK))
    git(ws, "add", "-f", "data/train.csv")
    r = git(ws, "commit", "-qm", "data", check=False)
    assert r.returncode != 0 and "data/train.csv" in r.stderr and "--no-verify" in r.stderr
    git(ws, "reset", "-q")
    (ws / "train.log").write_text("epoch 1\n")
    git(ws, "add", "-f", "train.log")
    r = git(ws, "commit", "-qm", "log", check=False)
    assert r.returncode == 0 and "train.log" in r.stderr  # a warning: the user chose to commit it


def test_the_pre_push_hook_refuses_a_token_in_any_commit_it_sends(ws, tmp_path):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    git(ws, "remote", "add", "origin", str(remote))
    assert git(ws, "push", "-q", "origin", "HEAD:refs/heads/main", check=False).returncode == 0
    (ws / "notes.md").write_text(f'"key": "{KEY}"\n')
    git(ws, "add", "notes.md")
    git(ws, "commit", "-qm", "notes", "--no-verify")
    r = git(ws, "push", "-q", "origin", "HEAD:refs/heads/main", check=False)
    assert r.returncode != 0 and "notes.md" in r.stderr and KEY not in r.stderr


def test_install_leaves_other_hooks_alone(repo, capsys):
    (_hooks(repo)).mkdir(parents=True, exist_ok=True)
    (_hooks(repo) / "pre-commit").write_text("#!/bin/sh\nnpx lint-staged\n")
    _init(repo)
    out = capsys.readouterr().out
    assert (_hooks(repo) / "pre-commit").read_text() == "#!/bin/sh\nnpx lint-staged\n"
    assert "pre-commit" in out and "kgkit repo-guard check --staged" in out
    assert RG.HOOK_MARK in (_hooks(repo) / "pre-push").read_text()
    git(repo, "config", "core.hooksPath", ".githooks")
    assert any("core.hooksPath" in x for x in RG.install_hooks(repo))
    assert not (repo / ".githooks").exists()


def test_each_session_points_the_hooks_at_the_current_plugin(ws, plugin_root):
    f = _hooks(ws) / "pre-commit"
    f.write_text(f.read_text().replace(RG._sh_path(RG.KGKIT_HOME), "/old/plugin/1.0.0"))
    assert RG.refresh_hooks(ws) == ["pre-commit"]
    assert "/old/plugin" not in f.read_text() and RG.refresh_hooks(ws) == []
    other = _hooks(ws) / "pre-push"
    other.write_text("#!/bin/sh\nexit 0\n")
    assert RG.refresh_hooks(ws) == [] and other.read_text() == "#!/bin/sh\nexit 0\n"  # not ours: never touched


def test_a_hook_that_cannot_find_kgkit_lets_the_commit_through_and_says_so(ws, monkeypatch):
    f = _hooks(ws) / "pre-commit"
    f.write_text(f.read_text().replace(RG._sh_path(RG.KGKIT_HOME), "/no/such/plugin"))
    monkeypatch.setenv("PATH", os.pathsep.join(p for p in os.environ["PATH"].split(os.pathsep)
                                               if not (Path(p) / "kgkit").exists()))
    git(ws, "add", "-f", "data/train.csv")
    r = git(ws, "commit", "-qm", "data", check=False)
    assert r.returncode == 0 and "repo guard did not run" in r.stderr


def test_an_exact_copy_is_found_when_git_stores_it_with_lf(ws):
    """Git for Windows has core.autocrlf on: a CRLF copy of a CRLF competition file is stored with LF endings, so
    its blob is neither the file's bytes nor its size."""
    git(ws, "config", "core.autocrlf", "true")
    crlf = TRAIN.replace("\n", "\r\n").encode()
    (ws / "data" / "train.csv").write_bytes(crlf)
    (ws / "src" / "copy.csv").write_bytes(crlf)
    git(ws, "add", "src/copy.csv")
    refused, _ = RG.check(ws, RG.staged(ws))
    assert _paths(refused) == {"src/copy.csv"} and "an exact copy of data/train.csv" in refused[0]
    git(ws, "reset", "-q")
    assert _paths(RG.check(ws, RG.would_add(ws, ["src/copy.csv"]))[0]) == {"src/copy.csv"}
