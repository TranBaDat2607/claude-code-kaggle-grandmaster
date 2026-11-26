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
    try:
        return subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=str(root), input=data,
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


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
        self.root, self.by_size, self.sample_header = root, {}, None
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
                            self.by_size.setdefault(e.stat().st_size, []).append(rel)
                            if e.name.lower().startswith("sample_submission") and self.sample_header is None:
                                self.sample_header = (rel, _header(Path(e.path).read_bytes()[:65536]))
                    except (OSError, IndexError):
                        continue
        self._oids: dict[str, str] = {}

    def copy_of(self, item: Item) -> str | None:
        for rel in self.by_size.get(item.size, []):
            if rel == item.path:
                continue
            if rel not in self._oids:
                r = _git(self.root, "hash-object", "--no-filters", "--", rel, timeout=120)
                self._oids[rel] = r.stdout.strip() if r is not None and r.returncode == 0 else ""
            oid = item.oid
            if oid is None and item.worktree is not None:
                r = _git(self.root, "hash-object", "--no-filters", "--", item.path, timeout=120)
                oid = r.stdout.strip() if r is not None and r.returncode == 0 else None
            if oid and oid == self._oids[rel]:
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
        if seen < SCAN_MAX and it.size <= 4 * READ_MAX:
            seen += 1
            hit = _secret_value(it.read(root))
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
    out = []
    if refused:
        out.append("kaggle-grandmaster: these must never be public (credentials, the competition's data):")
        out += [f"  - {x}" for x in refused[:limit]] + ([f"  - ... {len(refused) - limit} more"] if len(refused) > limit else [])
    if asked:
        out.append("kaggle-grandmaster: never-public files (outputs, logs, kernel outputs) that .gitignore keeps out:")
        out += [f"  - {x}" for x in asked[:limit]] + ([f"  - ... {len(asked) - limit} more"] if len(asked) > limit else [])
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
