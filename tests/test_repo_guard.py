"""Never-public files: the .gitignore `kgkit init` writes, and the guard that checks a commit or a push.

The .gitignore names only what is never code (data, outputs, weights, logs, credentials, caches). The files that
/kg-showcase removes at cleanup (copies of scripts, probes, scratch notebooks) stay tracked: experiment snapshots
(``refs/kg/*``) leave ignored files out, and the journey links to the code that ran."""
import shutil
import subprocess

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
    r = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"], cwd=root, input="\n".join(paths) + "\n",
                       capture_output=True, text=True)
    return set(r.stdout.split())


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

