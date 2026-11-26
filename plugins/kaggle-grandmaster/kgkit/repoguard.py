"""Files a competition repository never publishes: what its .gitignore names, and the check on a commit or a push.

A competition's repository goes public when the competition ends (/kg-showcase), and Kaggle's rules forbid
publishing the competition's data. Two layers keep the rest out of GitHub:

- **.gitignore** (``kgkit init``, ``kgkit repo-guard install``) names what is never code: the data, outputs and
  model weights, logs and scratch, credentials, tool caches. It names nothing that /kg-showcase removes at cleanup
  (copies of scripts such as ``train_v2.py``, probes outside ``work/``, scratch notebooks): those stay tracked until
  then, because an experiment's snapshot (``refs/kg/*``, :mod:`kgkit.snapshot`) leaves ignored files out and the
  journey links to the code that ran. Claude reads ignored files (Read, Glob and Bash do; only the Grep tool skips
  them), so ignoring never-code files hides nothing it needs.
- **The guard** checks what a ``git add``, ``git commit`` or ``git push`` would send, for what .gitignore cannot
  stop: ``git add -f``, a token pasted into code, a copy of a competition file under ``src/``. It runs in the
  PreToolUse hook for Claude's own git commands and as the repository's git hooks (pre-commit, pre-push) for the
  user's. Credentials and competition data are refused; other never-public files (submissions, predictions,
  weights, logs, kernel outputs) are asked about in Claude Code and warned about by the git hooks.

Standard library only (the hooks import it, and the git hooks run it); nothing here raises.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# (comment, [(pattern, a path the pattern ignores)]): the probe paths let `git check-ignore` tell whether a
# repository's own .gitignore already covers a pattern, whatever it is spelled like.
GITIGNORE = (
    ("# Competition data (copies and subsets too): Kaggle's rules forbid publishing it", [
        ("/data/", "data/kgkit-probe.csv"),
        ("*.zip", "kgkit-probe.zip"),
    ]),
    ("# Outputs: models, predictions, submissions", [
        ("/artifacts/", "artifacts/kgkit-probe.csv"),
        ("/subs/*.csv", "subs/kgkit-probe.csv"),
        ("*.npy", "src/kgkit-probe.npy"), ("*.npz", "src/kgkit-probe.npz"), ("*.pkl", "src/kgkit-probe.pkl"),
        ("*.joblib", "src/kgkit-probe.joblib"),
        ("*.parquet", "src/kgkit-probe.parquet"), ("*.feather", "src/kgkit-probe.feather"),
        ("*.pt", "src/kgkit-probe.pt"), ("*.pth", "src/kgkit-probe.pth"), ("*.ckpt", "src/kgkit-probe.ckpt"),
        ("*.safetensors", "src/kgkit-probe.safetensors"), ("*.onnx", "src/kgkit-probe.onnx"),
        ("*.h5", "src/kgkit-probe.h5"),
        ("wandb/", "src/wandb/x"), ("lightning_logs/", "src/lightning_logs/x"),
        ("catboost_info/", "src/catboost_info/x"),
        ("mlruns/", "src/mlruns/x"),
    ]),
    ("# Scratch: probes, logs, downloads", [
        ("/work/", "work/kgkit-probe.py"),
        ("*.log", "src/kgkit-probe.log"),
    ]),
    ("# Credentials", [
        ("kaggle.json", "src/kaggle.json"),
        ("access_token", "src/access_token"),
        (".env", "src/.env"), (".env.*", "src/.env.kgkit-probe"), ("!.env.example", None),
        ("*.pem", "src/kgkit-probe.pem"),
        ("/.claude/settings.local.json", ".claude/settings.local.json"),
    ]),
    ("# Caches", [
        ("__pycache__/", "src/__pycache__/a.pyc"),
        (".ipynb_checkpoints/", "src/.ipynb_checkpoints/a.ipynb"),
        (".pytest_cache/", ".pytest_cache/x"),
    ]),
)


def _git(root: Path, *args: str, data: str | None = None, timeout: float = 20) -> subprocess.CompletedProcess | None:
    """git's result with text stdout/stderr. ``data`` goes to stdin as UTF-8 bytes: in text mode Python on Windows
    would write each newline as CRLF, and `--stdin` readers take the CR as part of the path."""
    try:
        r = subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=str(root),
                           input=None if data is None else data.encode("utf-8"), capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    r.stdout = r.stdout.decode("utf-8", "replace").replace("\r\n", "\n")
    r.stderr = r.stderr.decode("utf-8", "replace")
    return r


def _covered(root: Path, text: str) -> set[str]:
    """The patterns of :data:`GITIGNORE` that ``root`` already ignores: by git's own rules (any spelling, a global
    ignore file too) in a repository, else by the lines of ``text``."""
    lines = {ln.strip() for ln in text.splitlines()}
    bare = {x.strip("/") for x in lines}
    have = {p for _, group in GITIGNORE for p, _ in group if p in lines or p.strip("/") in bare}
    probes = {probe: p for _, group in GITIGNORE for p, probe in group if probe}
    r = _git(root, "check-ignore", "--no-index", "--stdin", data="\n".join(probes) + "\n")
    if r is not None and r.returncode in (0, 1):
        have |= {probes[x] for x in r.stdout.splitlines() if x in probes}
    return have


def ensure_gitignore(root: str | Path) -> list[str]:
    """Add to ``root/.gitignore`` each pattern of :data:`GITIGNORE` it does not cover yet, under its group's
    comment, keeping the file's own lines and line endings. Returns the patterns added. Never raises."""
    root = Path(root)
    gi = root / ".gitignore"
    try:
        raw = gi.read_bytes() if gi.is_file() else b""
        text = raw.decode("utf-8", "replace")
        have = _covered(root, text)
        lines = {ln.strip() for ln in text.splitlines()}
        blocks, added = [], []
        for comment, group in GITIGNORE:
            new = [p for p, probe in group if (p not in have if probe else p not in lines)]
            if not [p for p in new if not p.startswith("!")]:
                continue  # a negation only goes with a pattern it qualifies
            blocks.append([comment, *new])
            added += [p for p in new if not p.startswith("!")]
        if not blocks:
            return []
        nl = "\r\n" if b"\r\n" in raw else "\n"
        lead = nl if raw and not raw.endswith(b"\n") else ""
        if raw:
            lead += nl  # a blank line between the repository's own lines and kgkit's
        body = (nl + nl).join(nl.join(b) for b in blocks) + nl
        with open(gi, "a", encoding="utf-8", newline="") as f:
            f.write(lead + ("" if raw else f"# Written by kgkit init: never-public files{nl}{nl}") + body)
        return added
    except OSError:
        return []


# ---------------------------------------------------------------------------------------------------- the guard
REFUSE_GROUPS = ("# Competition data", "# Credentials")  # the rest of GITIGNORE is asked about
DATA_DIRS = ("data", "input")  # as kgkit.audit.DATA_DIRS: where a workspace keeps the competition's files
NOT_THE_COMPETITION = ("ext",)  # data/ext/: collected kernel outputs, public libraries (kgkit.audit, #150)
STATE_GENERATED = (".kaggle-gm/remote/", ".kaggle-gm/build/", ".kaggle-gm/local/")  # kgkit.state.STATE_IGNORES
BIG = 50 * 1024 * 1024  # GitHub warns above 50 MB and refuses a file above 100 MB
READ_MAX = 1_000_000  # bytes read from a file for its credentials scan
SCAN_MAX = 20_000  # files read per check: a first push of a long history is only partly scanned
SCAN_BYTES = 256 * 1024 * 1024  # and at most this much of them, read by one `git cat-file --batch`
SECRET_NAMES = (".env", ".env.*", "*.env", "kaggle.json", "access_token", "access_token.txt", "credentials.json",
                ".netrc", ".pypirc", "id_rsa", "id_ed25519", "id_ecdsa", "*.pem", "*.p12", "*.pfx")  # kgkit.safety
NOT_SECRET = (".env.example", ".env.sample", ".env.template")
# (what it is, pattern): values, never the names alone (`KAGGLE_KEY=` with nothing after it is a template)
SECRET_VALUES = (
    ("Kaggle API key", re.compile(r"""(?:["']key["']\s*:|KAGGLE_KEY["']?\]?\s*[=:,])\s*["']?[0-9a-f]{32}\b""",
                                  re.IGNORECASE)),
    ("Kaggle API token", re.compile(r"\bKGAT_[A-Za-z0-9]{20,}")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40,})")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{34}\b")),
    ("Weights & Biases key", re.compile(r"WANDB_API_KEY[\"']?\]?\s*[=:,]\s*[\"']?[0-9a-f]{40}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("API key", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{32,}")),
    ("private key", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
)


@dataclass
class Item:
    """A file a git command would send: its path, its blob id when git has one, its size, and how to read it."""
    path: str
    size: int
    oid: str | None = None
    worktree: Path | None = None

    def read(self, root: Path, n: int = READ_MAX) -> bytes:
        try:
            if self.worktree is not None:
                with open(self.worktree, "rb") as f:
                    return f.read(n)
            r = subprocess.run(["git", "cat-file", "blob", self.oid], cwd=str(root), capture_output=True, timeout=20)
            return r.stdout[:n] if r.returncode == 0 else b""
        except (OSError, subprocess.SubprocessError, TypeError):
            return b""


def _prefetch(root: Path, items: list[Item]) -> dict[str, bytes]:
    """The contents of the git blobs among ``items`` that the credentials scan reads, by one ``git cat-file
    --batch`` (a process per file would take minutes on a first push of a long history)."""
    want, total = [], 0
    for it in items:
        if it.oid and it.size <= READ_MAX and len(want) < SCAN_MAX and total + it.size <= SCAN_BYTES:
            want.append(it.oid)
            total += it.size
    want = list(dict.fromkeys(want))
    if not want:
        return {}
    try:
        r = subprocess.run(["git", "cat-file", "--batch"], cwd=str(root), input=("\n".join(want) + "\n").encode(),
                           capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return {}
    out, data, i = {}, r.stdout, 0
    while i < len(data):
        nl = data.find(b"\n", i)
        if nl < 0:
            break
        head = data[i:nl].split()
        if len(head) != 3:  # "<oid> missing"
            i = nl + 1
            continue
        size = int(head[2])
        out[head[0].decode()] = data[nl + 1:nl + 1 + size]
        i = nl + 1 + size + 1
    return out


def _z(r: subprocess.CompletedProcess | None) -> list[str]:
    return [x for x in r.stdout.split("\0") if x] if r is not None and r.returncode == 0 else []


def _sizes(root: Path, oids: list[str]) -> dict[str, int]:
    r = _git(root, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)", data="\n".join(oids) + "\n",
             timeout=60)
    out = {}
    for line in (r.stdout.splitlines() if r is not None and r.returncode == 0 else []):
        parts = line.split()
        if len(parts) == 3 and parts[1] == "blob":
            out[parts[0]] = int(parts[2])
    return out


def _from_index(root: Path, entries: list[tuple[str, str]]) -> list[Item]:
    """Items of (path, blob id) pairs; ids that are not blobs (submodules) are left out."""
    sizes = _sizes(root, sorted({o for _, o in entries}))
    return [Item(p, sizes[o], oid=o) for p, o in entries if o in sizes]


def staged(root: str | Path) -> list[Item]:
    """What a ``git commit`` would record: the files added or changed in the index (deletions send nothing)."""
    root = Path(root)
    changed = _z(_git(root, "diff", "--cached", "--name-only", "-z", "--no-renames", "--diff-filter=ACMRT"))
    if not changed:
        return []
    r = _git(root, "ls-files", "-s", "-z", "--", *[f":(literal){p}" for p in changed])
    entries = []
    for rec in _z(r):
        meta, _, path = rec.partition("\t")
        parts = meta.split()
        if len(parts) >= 2:
            entries.append((path, parts[1]))
    return _from_index(root, entries)


def _worktree(root: Path, paths: list[str]) -> list[Item]:
    out = []
    for p in dict.fromkeys(paths):
        f = root / p
        try:
            if f.is_file():
                out.append(Item(p, f.stat().st_size, worktree=f))
        except OSError:
            continue
    return out


def would_add(root: str | Path, args: list[str], cwd: str | Path | None = None) -> list[Item]:
    """What ``git add <args>`` run in ``cwd`` (default ``root``, the top folder) would stage, from ``git add
    --dry-run`` (so ignore rules and ``-f`` apply as git applies them), read from the working tree."""
    root = Path(root)
    args = [a for a in args if a not in ("-n", "--dry-run", "-v", "--verbose", "-q", "--quiet")]
    r = _git(Path(cwd or root), "add", "--dry-run", *args, timeout=60)
    if r is None or r.returncode != 0:
        return []
    paths = []
    for line in r.stdout.splitlines():
        if line.startswith("add '") and line.endswith("'"):
            paths.append(line[5:-1])
    return _worktree(root, paths)


def modified(root: str | Path, paths: list[str] | None = None) -> list[Item]:
    """What ``git commit -a`` (or ``git commit <paths>``) adds to the staged files: tracked files changed in the
    working tree."""
    root = Path(root)
    spec = ["--", *paths] if paths else []
    return _worktree(root, _z(_git(root, "diff", "--name-only", "-z", "--no-renames", "--diff-filter=ACMRT", *spec)))


def pushed(root: str | Path, tips: list[str], remote: str | None = None) -> list[Item]:
    """Every file version that pushing ``tips`` would send: the blobs of their commits that no ref of ``remote``
    (all remotes when None) already has. A file removed by a later commit is in the push too."""
    root = Path(root)
    tips = [t for t in tips if t and (t.startswith("--") or _resolves(root, t))]
    if not tips:
        return []  # nothing git could push
    have = f"--remotes={remote}" if remote else "--remotes"
    r = _git(root, "rev-list", "--objects", *tips, "--not", have, timeout=120)
    entries = []
    for line in (r.stdout.splitlines() if r is not None and r.returncode == 0 else []):
        oid, _, path = line.partition(" ")
        if path:
            entries.append((path, oid))
    return _from_index(root, entries)


def _resolves(root: Path, ref: str) -> bool:
    r = _git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", timeout=10)
    return r is not None and r.returncode == 0


def _match(pattern: str, path: str) -> bool:
    """Whether a .gitignore ``pattern`` of :data:`GITIGNORE` (``/dir/``, ``dir/``, ``*.ext``, ``name``,
    ``/dir/*.csv``) names ``path``, a file."""
    anchored, is_dir, p = pattern.startswith("/"), pattern.endswith("/"), pattern.strip("/")
    parts = path.split("/")
    if is_dir:
        dirs = parts[:-1]
        if "/" in p or anchored:
            return path.startswith(p + "/")
        return any(fnmatch.fnmatchcase(d, p) for d in dirs)
    if "/" in p or anchored:
        return fnmatch.fnmatchcase(path, p)
    return fnmatch.fnmatchcase(parts[-1], p)


def _secret_name(name: str) -> bool:
    low = name.lower()
    return low not in NOT_SECRET and any(fnmatch.fnmatchcase(low, p) for p in SECRET_NAMES)


def _secret_value(data: bytes) -> tuple[str, int] | None:
    """(what, line number) of the first credential value in ``data``; None when there is none."""
    if b"\0" in data[:8000]:
        return None  # binary
    text = data.decode("utf-8", "replace")
    for what, rx in SECRET_VALUES:
        m = rx.search(text)
        if m:
            return what, text.count("\n", 0, m.start()) + 1
    return None


class _Competition:
    """The competition's files on this machine (the files of data/ and input/ outside ext/), by size, for the
    exact-copy check. Their blob ids are computed only for a size some item has."""

    def __init__(self, root: Path):
        self.root, self.by_size, self.lf_size, self.sample_header = root, {}, {}, None
        for d in DATA_DIRS:
            base = root / d
            if not base.is_dir():
                continue
            stack = [base]
            while stack and sum(len(v) for v in self.by_size.values()) < 5000:
                here = stack.pop()
                try:
                    entries = list(os.scandir(here))
                except OSError:
                    continue
                for e in entries:
                    rel = Path(e.path).relative_to(root).as_posix()
                    try:
                        if e.is_dir(follow_symlinks=False):
                            if rel.split("/")[1] not in NOT_THE_COMPETITION:
                                stack.append(Path(e.path))
                        elif e.is_file():
                            size = e.stat().st_size
                            self.by_size.setdefault(size, []).append(rel)
                            if size <= 64 * 2**20:  # its size with CRLF turned into LF, as autocrlf stores it
                                with open(e.path, "rb") as f:
                                    crlf = f.read().count(b"\r\n")
                                if crlf:
                                    self.lf_size.setdefault(size - crlf, []).append(rel)
                            if e.name.lower().startswith("sample_submission") and self.sample_header is None:
                                self.sample_header = (rel, _header(Path(e.path).read_bytes()[:65536]))
                    except (OSError, IndexError):
                        continue
        self._oids: dict[str, str] = {}

    def _hashes(self, rel: str) -> set[str]:
        """A file's blob ids as it is and as `git add` would store it (core.autocrlf, the default of Git for
        Windows, turns CRLF into LF: the bytes in git then differ from the file's)."""
        out = set()
        for flags in (["--no-filters"], ["--path", rel]):
            r = _git(self.root, "hash-object", *flags, "--", rel, timeout=120)
            if r is not None and r.returncode == 0 and r.stdout.strip():
                out.add(r.stdout.strip())
        return out

    def copy_of(self, item: Item) -> str | None:
        # a blob's size is after the filters, so a CRLF file stored as LF is smaller than the file: by size, both
        candidates = list(dict.fromkeys(self.by_size.get(item.size, []) + self.lf_size.get(item.size, [])))
        mine = None
        for rel in candidates:
            if rel == item.path:
                continue
            if rel not in self._oids:
                self._oids[rel] = self._hashes(rel)
            if mine is None:
                mine = {item.oid} if item.oid else (self._hashes(item.path) if item.worktree is not None else set())
            if mine & self._oids[rel]:
                return rel
        return None


def _header(data: bytes) -> str | None:
    line = data.split(b"\n", 1)[0].strip().decode("utf-8", "replace").lstrip("﻿")
    return line.lower() if "," in line else None


def _data_file(path: str) -> str | None:
    """The data folder a path is in ("data"), or None; ext/ under it is not the competition's."""
    parts = path.split("/")
    if len(parts) > 1 and parts[0] in DATA_DIRS:
        return None if len(parts) > 2 and parts[1] in NOT_THE_COMPETITION else parts[0]
    return None


def check(root: str | Path, items: list[Item]) -> tuple[list[str], list[str]]:
    """(refused, asked): one ``"<path>: <why>"`` per file of ``items`` that must not go to GitHub (credentials,
    competition data) and per file that should not (other never-public files). Never the secret's value."""
    root = Path(root)
    refused, asked = [], []
    comp = None
    seen = 0
    blobs = _prefetch(root, items)
    for it in items:
        path = it.path
        name = path.rsplit("/", 1)[-1]
        if _secret_name(name):
            refused.append(f"{path}: a credentials file")
            continue
        d = _data_file(path)
        if d:
            refused.append(f"{path}: in {d}/, where the competition's files are (Kaggle's rules forbid publishing "
                           "them, copies and subsets too)")
            continue
        if it.oid in blobs or (it.worktree is not None and seen < SCAN_MAX and it.size <= READ_MAX):
            seen += 1
            hit = _secret_value(blobs[it.oid] if it.oid in blobs else it.read(root))
            if hit:
                refused.append(f"{path}: holds what looks like a {hit[0]} (line {hit[1]})")
                continue
        comp = comp or _Competition(root)
        src = comp.copy_of(it)
        if src:
            refused.append(f"{path}: an exact copy of {src}, a competition file")
            continue
        why = _never_public(path)
        if why is None and comp.sample_header and path.lower().endswith(".csv") and it.size < BIG:
            rel, head = comp.sample_header
            if head and _header(it.read(root, 65536)) == head:
                why = f"a submission file (the columns of {rel})"
        if why is None and path.split("/")[0] in DATA_DIRS:
            why = "a file in the data folders that is not the competition's (a kernel output or a public library)"
        if why is None and it.size > BIG:
            why = f"{it.size / 2**20:.0f} MB: GitHub warns above 50 MB and refuses a file above 100 MB"
        if why:
            asked.append(f"{path}: {why}")
    return refused, asked


def _never_public(path: str) -> str | None:
    for prefix in STATE_GENERATED:
        if path.startswith(prefix):
            return f"generated by kgkit in {prefix} (remote-run downloads, built kernels)"
    for comment, group in GITIGNORE:
        if comment.startswith(REFUSE_GROUPS):
            continue
        for p, _ in group:
            if not p.startswith("!") and _match(p, path):
                kind = comment.lstrip("# ").split(":")[0].lower()
                return f"{kind} (kgkit's .gitignore names `{p}`)"
    return None


def fix_steps(refused: list[str], asked: list[str], staged_files: bool = True) -> list[str]:
    """What to do about a check's findings, as lines."""
    paths = [x.split(": ", 1)[0] for x in refused + asked]
    shown = " ".join(_quote(p) for p in paths[:6]) + (" ..." if len(paths) > 6 else "")
    out = []
    if staged_files:
        out.append(f"unstage: `git rm --cached {shown}` (the files stay on disk)")
    else:
        out.append("the files are in commits that are not on the remote yet: drop them from those commits "
                   "(`git reset --soft <commit>` and commit again, or `git rebase -i`) before pushing")
    out.append("keep them out: `kgkit repo-guard install` completes .gitignore")
    return out


def _quote(p: str) -> str:
    return p if all(c.isalnum() or c in "/._-" for c in p) else "'" + p.replace("'", "'\\''") + "'"


def report(refused: list[str], asked: list[str], staged_files: bool = True, limit: int = 12) -> str:
    """The guard's message: what was found, and what to do."""
    def listed(found):
        more = [f"  - ... {len(found) - limit} more"] if len(found) > limit else []
        return [f"  - {x}" for x in found[:limit]] + more

    out = []
    if refused:
        out.append("kaggle-grandmaster: these must never be public (credentials, the competition's data):")
        out += listed(refused)
    if asked:
        out.append("kaggle-grandmaster: never-public files (outputs, logs, kernel outputs) that .gitignore keeps out:")
        out += listed(asked)
    out += [f"  {x}" for x in fix_steps(refused, asked, staged_files)]
    return "\n".join(out)


def toplevel(start: str | Path = ".") -> Path | None:
    """The repository's top folder, None outside one."""
    r = _git(Path(start), "rev-parse", "--show-toplevel", timeout=10)
    return Path(r.stdout.strip()) if r is not None and r.returncode == 0 and r.stdout.strip() else None


# What Claude's git commands would send (hooks/pre_tool.py): the arguments of `git add`, `git commit`, `git push`.
COMMIT_VALUE_OPTS = {"-m", "--message", "-F", "--file", "-C", "--reuse-message", "-c", "--reedit-message", "-t",
                     "--template", "--author", "--date", "--cleanup", "--fixup", "--squash", "--trailer",
                     "--pathspec-from-file"}
PUSH_VALUE_OPTS = {"--repo", "-o", "--push-option", "--receive-pack", "--exec", "--force-with-lease"}


def commit_items(root: Path, args: list[str], cwd: Path | None = None) -> list[Item]:
    """What ``git commit <args>`` would record: the staged files, plus the changed tracked files with ``-a``, or the
    paths it names."""
    every, paths, i = False, [], 0
    while i < len(args):
        a = args[i]
        if a == "--":
            paths += args[i + 1:]
            break
        if a in COMMIT_VALUE_OPTS:
            i += 2
            continue
        if a in ("-a", "--all") or (a.startswith("-") and not a.startswith("--") and "a" in a[1:].split("m")[0]
                                    and all(c.isalpha() for c in a[1:])):
            every = True  # -a, -am "msg", -qa
        elif not a.startswith("-"):
            paths.append(a)
        i += 1
    out = staged(root)
    if every:
        out += modified(root)
    elif paths:
        base = Path(cwd or root)
        rel = [os.path.relpath(base / p, root).replace(os.sep, "/") for p in paths]
        out += modified(root, rel)
    return out


def push_items(root: Path, args: list[str]) -> list[Item]:
    """What ``git push <args>`` would send (see :func:`pushed`)."""
    pos, tips, i = [], [], 0
    while i < len(args):
        a = args[i]
        if a in PUSH_VALUE_OPTS:
            i += 2
            continue
        if a in ("--all", "--branches", "--mirror"):
            tips.append("--branches" if a != "--mirror" else "--all")
        elif a == "--tags":
            tips.append("--tags")
        elif not a.startswith("-"):
            pos.append(a)
        i += 1
    remote = pos[0] if pos else None
    for spec in pos[1:]:
        src = spec.lstrip("+").split(":", 1)[0]
        if not src:
            continue  # `:branch` deletes a remote branch: nothing is sent
        tips.append(f"--glob={src if src.startswith('refs/') else 'refs/heads/' + src}" if "*" in src else src)
    if remote is not None:
        r = _git(root, "remote", "get-url", remote, timeout=10)
        remote = remote if r is not None and r.returncode == 0 else None  # a URL: whatever no remote has
    return pushed(root, tips or ["HEAD"], remote)


# --------------------------------------------------------------------------------------------- the git hooks
HOOK_MARK = "# kaggle-grandmaster repo guard"
HOOKS = ("pre-commit", "pre-push")
KGKIT_HOME = Path(__file__).resolve().parents[1]  # the folder that holds this kgkit (the plugin, or a vendored copy)


def _sh_path(p) -> str:
    """A path as a single-quoted sh word (forward slashes: Git for Windows runs hooks with its own sh)."""
    return "'" + Path(p).as_posix().replace("'", "'\\''") + "'"


def hook_script(name: str, python: str | None = None, home: Path | None = None) -> str:
    """The git hook ``name``: runs this module with the Python and the kgkit folder of the session that wrote it
    (each Claude Code session rewrites them: :func:`refresh_hooks`), else a `kgkit` on PATH; it lets the commit
    or push through, saying so, when it finds neither (a hook never stops the user for its own failure)."""
    py, home = _sh_path(python or sys.executable), _sh_path(home or KGKIT_HOME)
    verb = "commit" if name == "pre-commit" else "push"
    return f"""#!/bin/sh
{HOOK_MARK} ({name}): written by `kgkit init` or `kgkit repo-guard install`, and kept up to date
# by each Claude Code session in this repository. Refuses a {verb} that would send credentials or the
# competition's data, and warns about other never-public files. `git {verb} --no-verify` skips it once;
# delete this file to turn it off.
KGKIT_PYTHON={py}
KGKIT_HOME={home}
if [ -f "$KGKIT_HOME/kgkit/repoguard.py" ] && [ -x "$KGKIT_PYTHON" ]; then
  PYTHONPATH="$KGKIT_HOME" exec "$KGKIT_PYTHON" -m kgkit.repoguard hook {name} "$@"
fi
if command -v kgkit >/dev/null 2>&1; then
  exec kgkit repo-guard hook {name} "$@"
fi
echo "kaggle-grandmaster: the repo guard did not run ($KGKIT_HOME has no kgkit):" \\
  "open Claude Code in this repository to update the hook" >&2
exit 0
"""


def _hooks_dir(root: Path) -> tuple[Path | None, str | None]:
    """(the repository's hooks folder, why not) for writing kgkit's hooks."""
    r = _git(root, "config", "--get", "core.hooksPath", timeout=10)
    if r is not None and r.returncode == 0 and r.stdout.strip():
        return None, (f"core.hooksPath is set ({r.stdout.strip()}): git runs the hooks there, which kgkit does not "
                      "write; add `kgkit repo-guard check --staged` (pre-commit) and `kgkit repo-guard check --push` "
                      "(pre-push) to them")
    r = _git(root, "rev-parse", "--git-path", "hooks", timeout=10)
    if r is None or r.returncode != 0 or not r.stdout.strip():
        return None, "not a git repository: no hooks installed (run `kgkit repo-guard install` after `git init`)"
    d = Path(r.stdout.strip())
    return (d if d.is_absolute() else Path(root) / d), None


def _write_hook(f: Path, text: str) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "w", encoding="utf-8", newline="\n") as out:
        out.write(text)
    try:
        f.chmod(0o755)
    except OSError:
        pass


def install_hooks(root: str | Path) -> list[str]:
    """Write kgkit's pre-commit and pre-push hooks into the repository of ``root`` (its own hooks are rewritten, any
    other hook is left alone). Returns one line per hook not written, with why. Never raises."""
    root = Path(root)
    d, why = _hooks_dir(root)
    if d is None:
        return [why]
    notes = []
    for name in HOOKS:
        f = d / name
        try:
            if f.is_file() and HOOK_MARK not in f.read_text(encoding="utf-8", errors="replace"):
                cmd = "check --staged" if name == "pre-commit" else "check --push"
                notes.append(f"{name}: the repository has its own {name} hook, left as it is; add "
                             f"`kgkit repo-guard {cmd}` to it")
                continue
            _write_hook(f, hook_script(name))
        except OSError as e:
            notes.append(f"{name}: not written ({e})")
    return notes


def refresh_hooks(root: str | Path) -> list[str]:
    """Rewrite kgkit's hooks of ``root``'s repository whose Python or kgkit folder is not this session's (a plugin
    update moves the plugin). Hooks that are not kgkit's, and missing ones, are left alone. Returns the hooks
    rewritten. Never raises; fast (a session start runs it)."""
    try:
        git_dir = Path(root) / ".git"
        d = git_dir / "hooks" if git_dir.is_dir() else _hooks_dir(Path(root))[0]
        out = []
        for name in HOOKS:
            f = d / name if d is not None else None
            if f is None or not f.is_file():
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
            if HOOK_MARK in text and text != hook_script(name):
                _write_hook(f, hook_script(name))
                out.append(name)
        return out
    except OSError:
        return []


def _hook_main(argv: list[str]) -> int:
    """``python -m kgkit.repoguard hook pre-commit|pre-push [<remote> <url>]``, as git runs it (pre-push reads its
    refs from stdin). 1 refuses; warnings alone go to stderr and let it through."""
    name = argv[0] if argv else ""
    root = toplevel(".")
    if root is None or name not in HOOKS:
        return 0
    if name == "pre-commit":
        items, staged_files = staged(root), True
    else:
        remote = argv[1] if len(argv) > 1 else None
        tips = []
        for line in sys.stdin.read().splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[1].strip("0"):  # an all-zero id deletes a remote ref: nothing is sent
                tips.append(parts[1])
        if not tips:
            return 0
        items, staged_files = pushed(root, tips, remote), False
    refused, asked = check(root, items)
    if not (refused or asked):
        return 0
    sys.stderr.write(report(refused, asked, staged_files) + "\n")
    if refused:
        verb = "commit" if name == "pre-commit" else "push"
        sys.stderr.write(f"kaggle-grandmaster: {verb} refused. `git {verb} --no-verify` skips this check (only if "
                         "you are sure: a public repository cannot take back what it published)\n")
        return 1
    return 0


if __name__ == "__main__":
    try:
        code = _hook_main(sys.argv[2:]) if sys.argv[1:2] == ["hook"] else 0
    except Exception:  # a hook never stops the user for its own failure
        code = 0
    sys.exit(code)
