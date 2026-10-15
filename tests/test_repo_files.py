"""Repository hygiene the plugin directory scanner relies on."""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (REPO / ".git").exists(), reason="needs a git checkout"
)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def test_no_gitattributes():
    # The directory scanner rejects attributes that rewrite files at checkout (eol, filter, ident, ...).
    assert [f for f in _git("ls-files").splitlines() if f.endswith(".gitattributes")] == []


def test_committed_text_files_use_lf():
    # Without .gitattributes nothing normalises line endings, so CRLF must never be committed
    # (hooks/run.sh in particular breaks under sh with CRLF).
    crlf = [line.split("\t", 1)[1] for line in _git("ls-files", "--eol").splitlines() if line.split()[0] != "i/lf"
            and line.split()[0] not in ("i/-text", "i/none")]
    assert crlf == []
