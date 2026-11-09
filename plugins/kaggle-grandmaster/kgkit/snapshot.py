"""The exact code behind every experiment: a snapshot commit under ``refs/kg/`` when the working tree has changes.

An experiment is usually run from code that is not committed yet, so the commit a record names (``git_commit``)
holds the code *before* the change it tested. When the workspace's code differs from that commit, :func:`git_info`
writes a snapshot of it as a commit object whose parent is ``HEAD`` and keeps it under ``refs/kg/<name>``. Neither the
working tree nor the staging area is touched: the snapshot is built in a temporary index (``read-tree HEAD``, then
the changed and the new files, then ``write-tree`` and ``commit-tree``).

What goes in: every change under the workspace folder that git does not ignore, except ``.kaggle-gm/`` (the ledger
changes with every record, it is not code). Untracked files larger than ``MAX_FILE_BYTES``, and all of them when
there are more than ``MAX_UNTRACKED``, are left out (``git_snapshot_skipped`` counts them).

The refs stay local. ``git push`` does not send ``refs/kg/*``; `/kg-showcase` pushes them, after `kgkit audit-repo`
(which reads every ref) has checked them for competition data, so that the journey's code links open the code that
ran. Standard library only (``kgkit.gpu`` imports it); every function returns quietly when git fails.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

REF_PREFIX = "refs/kg/"
MAX_FILE_BYTES = 1_000_000
MAX_UNTRACKED = 2000
EXCLUDE = ":(exclude).kaggle-gm"
TIMEOUT = 20


def _run(root: Path, *args: str, env: dict | None = None, data: bytes | None = None,
         timeout: float = TIMEOUT) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=str(root), input=data,
                              capture_output=True, timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError):
        return None


def _out(root: Path, *args: str, env: dict | None = None, data: bytes | None = None) -> str | None:
    r = _run(root, *args, env=env, data=data)
    if r is None or r.returncode != 0:
        return None
    return r.stdout.decode("utf-8", "replace").strip()


def ref_name(name: str) -> str:
    """``refs/kg/<name>`` with the characters git does not allow in a ref replaced."""
    parts = [re.sub(r"[^A-Za-z0-9._-]+", "-", p).strip(".-") or "x" for p in str(name).split("/")]
    return REF_PREFIX + "/".join(p[:-5] + "-lock" if p.endswith(".lock") else p for p in parts)


def changed(root: Path) -> list[str] | None:
    """Paths under ``root`` (outside .kaggle-gm/) that differ from HEAD or are new and not ignored; None when git
    cannot tell (no repository, no commit yet)."""
    r = _run(root, "status", "--porcelain", "-z", "--untracked-files=all", "--", ".", EXCLUDE)
    if r is None or r.returncode != 0:
        return None
    return [e[3:] for e in r.stdout.decode("utf-8", "replace").split("\0") if len(e) > 3]


def _new_files(root: Path) -> list[str] | None:
    """Files not in HEAD: untracked and not ignored, or staged as new (``git add`` without a commit), as paths
    relative to ``root``."""
    out = _run(root, "ls-files", "--others", "--exclude-standard", "-z", "--", ".", EXCLUDE)
    if out is None or out.returncode != 0:
        return None
    files = [p for p in out.stdout.decode("utf-8", "replace").split("\0") if p]
    staged = _run(root, "diff", "--cached", "--name-only", "--relative", "--diff-filter=A", "-z", "--", ".", EXCLUDE)
    if staged is not None and staged.returncode == 0:
        files += [p for p in staged.stdout.decode("utf-8", "replace").split("\0") if p and (root / p).is_file()]
    return list(dict.fromkeys(files))


def _untracked(root: Path) -> tuple[list[str], int]:
    """(new files to put in the snapshot, how many were left out for size or number)."""
    files = _new_files(root)
    if files is None:
        return [], 0
    if len(files) > MAX_UNTRACKED:
        return [], len(files)
    keep = []
    for p in files:
        try:
            if (root / p).stat().st_size <= MAX_FILE_BYTES:
                keep.append(p)
        except OSError:
            pass
    return keep, len(files) - len(keep)


def _identity_env(root: Path, env: dict) -> dict:
    """The user's git identity when one is set, else a neutral one: commit-tree refuses to run without any."""
    if _out(root, "var", "GIT_COMMITTER_IDENT", env=env) and _out(root, "var", "GIT_AUTHOR_IDENT", env=env):
        return env
    return {**env, "GIT_AUTHOR_NAME": "kgkit", "GIT_AUTHOR_EMAIL": "kgkit@localhost",
            "GIT_COMMITTER_NAME": "kgkit", "GIT_COMMITTER_EMAIL": "kgkit@localhost"}


def _same_snapshot(root: Path, tree: str, head: str) -> str | None:
    """The newest snapshot commit when it holds exactly ``tree`` on top of ``head``: runs that differ only in their
    arguments share one commit, so a diff link between them appears only when the code changed."""
    out = _out(root, "for-each-ref", "--sort=-creatordate", "--count=1", "--format=%(objectname) %(tree) %(parent)",
               REF_PREFIX)
    parts = (out or "").split()
    return parts[0] if len(parts) == 3 and parts[1] == tree and parts[2] == head else None


def snapshot(root: str | Path, name: str, message: str = "") -> tuple[str | None, int]:
    """(full sha of a snapshot commit of the workspace's code kept under ``refs/kg/<name>``, files left out); (None,
    n) when there is nothing to snapshot or git failed. The working tree and the index are never touched."""
    root = Path(root)
    head = _out(root, "rev-parse", "--verify", "-q", "HEAD")
    if not head:
        return None, 0
    with tempfile.TemporaryDirectory(prefix="kgkit-snap-") as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        if _out(root, "read-tree", head, env=env) is None:
            return None, 0
        if _out(root, "add", "-u", "--", ".", EXCLUDE, env=env) is None:
            return None, 0
        new, skipped = _untracked(root)
        if new:
            data = b"\0".join(p.encode("utf-8") for p in new)
            if _out(root, "add", "--pathspec-from-file=-", "--pathspec-file-nul", env=env, data=data) is None:
                for i in range(0, len(new), 100):  # git < 2.25 has no --pathspec-from-file
                    if _out(root, "add", "--", *new[i:i + 100], env=env) is None:
                        return None, skipped
        tree = _out(root, "write-tree", env=env)
        if not tree or tree == _out(root, "rev-parse", f"{head}^{{tree}}"):
            return None, skipped
        sha = _same_snapshot(root, tree, head)  # the code did not change since the last snapshot: the same commit
        if sha is None:
            msg = message or f"kgkit snapshot {name}"
            # never signed: a user's commit.gpgSign would ask for a passphrase on every logged run
            sha = _out(root, "commit-tree", "--no-gpg-sign", tree, "-p", head, "-m", msg, env=_identity_env(root, env))
        if not sha or _out(root, "update-ref", ref_name(name), sha) is None:
            return None, skipped
        return sha, skipped


def git_info(root: str | Path, name: str | None = None) -> dict:
    """``git_commit`` (HEAD, short), ``git_dirty`` (the workspace's code differs from it) and, when it does and
    ``name`` is given, ``git_snapshot``: the full sha of a commit holding the code as it is now (:func:`snapshot`).
    Every value is None when git cannot tell. Never raises."""
    root = Path(root)
    try:
        commit = _out(root, "rev-parse", "--short", "HEAD")
        if not commit:
            return {"git_commit": None, "git_dirty": None}
        diff = changed(root)
        info: dict = {"git_commit": commit, "git_dirty": bool(diff) if diff is not None else None}
        if diff and name:
            sha, skipped = snapshot(root, name)
            if sha:
                info["git_snapshot"] = sha
            if skipped:
                info["git_snapshot_skipped"] = skipped
        return info
    except Exception:  # a record is never lost to a git problem
        return {"git_commit": None, "git_dirty": None}


def code_ref(rec: dict) -> tuple[str | None, bool]:
    """(the commit holding the code a record ran from, exact?): its snapshot, else its commit, which is exact only
    when the tree was clean."""
    snap = rec.get("git_snapshot")
    if snap:
        return str(snap), True
    commit = rec.get("git_commit")
    return (str(commit), rec.get("git_dirty") is False) if commit else (None, False)


def local_refs(root: str | Path) -> list[str]:
    """The snapshot refs in the repository (``refs/kg/...``)."""
    out = _out(Path(root), "for-each-ref", "--format=%(refname)", REF_PREFIX)
    return [line for line in (out or "").splitlines() if line]
