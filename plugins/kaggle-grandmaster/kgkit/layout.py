"""Is the workspace still a repository someone could read? Signs of drift from the layout `kgkit init` sets up.

A competition's repository goes public when the competition ends (/kg-showcase): the final pipeline in ``solution/``,
how the ideas developed in ``journey/``. That works when the tree stayed tidy while the work happened, so these
checks run in the session brief and ``kgkit status``, and the hooks catch the two habits that make most of the mess
at the moment they happen:

- a variant written as a copy of a script (``train_v2.py`` beside ``train.py``) instead of a flag of that script;
- a log written next to the code or the reports instead of in ``work/`` (ignored by git).

Nothing here blocks. Standard library only (hooks import it); every check returns quietly when git fails.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

SRC_SCRIPTS_MAX = 15
UNDECIDED_MIN = 3
WORK = "work"
GENERATED = (".kaggle-gm/remote", ".kaggle-gm/build")
# train_v2, train-new, train_copy, train (2), "train - Copy": a copy of a script, not a script of its own
VARIANT_RE = re.compile(r"^(?P<base>.+?)(?:[_-](?:v\d+|new|copy|old|fixed|final|alt|tmp|bak|backup|\d)"
                        r"|\s*\(\d+\)|\s+-\s+copy|\s+copy)$", re.IGNORECASE)
# `> x.log`, `>> x.log`, `2>&1 > x.log`, `tee [-a] x.log`, PowerShell `Out-File [-FilePath] x.log`, `Tee-Object x.log`
LOG_RE = re.compile(r"(?:\d?>>?|\btee(?:\s+-a)?|\bOut-File(?:\s+-FilePath)?|\bTee-Object(?:\s+-FilePath)?)\s*"
                    r"(['\"]?)(?P<path>[^\s'\";|&<>]+\.(?:log|out))\1", re.IGNORECASE)


def _git(root: Path, *args: str, timeout: float = 5) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=str(root), capture_output=True,
                              timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def _tracked(root: Path, *pathspec: str) -> list[str] | None:
    r = _git(root, "ls-files", "-z", "--", *pathspec)
    if r is None or r.returncode != 0:
        return None
    return [p for p in r.stdout.decode("utf-8", "replace").split("\0") if p]


def variant_of(path: str | Path) -> Path | None:
    """The script a file looks like a copy of (``train_v2.py`` -> ``train.py``), when that script is beside it."""
    path = Path(path)
    m = VARIANT_RE.match(path.stem)
    if not m or not m.group("base").strip():
        return None
    base = path.with_name(m.group("base").strip() + path.suffix)
    try:
        return base if base != path and base.is_file() else None
    except OSError:
        return None


def log_targets(command: str) -> list[str]:
    """Log files a shell command writes to with a relative path outside ``work/`` (and outside .kaggle-gm/, whose logs
    kgkit keeps)."""
    out = []
    for m in LOG_RE.finditer(command or ""):
        p = m.group("path").replace("\\", "/")
        if p.startswith(("/", "~", "$")) or re.match(r"^[A-Za-z]:/", p):
            continue  # an absolute or temp path: outside the repository
        p = p[2:] if p.startswith("./") else p
        if p.split("/")[0] in (WORK, ".kaggle-gm", "..") or p in out:
            continue
        out.append(p)
    return out


def drift(root: str | Path) -> list[str]:
    """One sentence per sign of drift, with what to do about it; [] for a tidy workspace."""
    root = Path(root)
    out = []
    src = root / "src"
    try:
        scripts = sorted(p for p in src.glob("*.py")) if src.is_dir() else []
    except OSError:
        scripts = []
    copies = [(p, variant_of(p)) for p in scripts]
    copies = [(p, b) for p, b in copies if b is not None]
    if copies:
        shown = ", ".join(f"{p.name} (of {b.name})" for p, b in copies[:4]) + (" ..." if len(copies) > 4 else "")
        out.append(f"copies of scripts in src/: {shown}. A variant belongs behind a flag of the script it varies, "
                   "so its record's command says what changed")
    if len(scripts) > SRC_SCRIPTS_MAX:
        out.append(f"src/ has {len(scripts)} scripts at its top level. One-off probes go in work/ (ignored by git); an "
                   "analysis a decision rests on goes in studies/<name>/ with a README; group the pipeline by stage")
    logs = _tracked(root, "*.log", f":(exclude){WORK}", ":(exclude).kaggle-gm")
    if logs:
        out.append(f"{len(logs)} log file(s) tracked by git (e.g. {logs[0]}). Logs go in work/; the numbers that "
                   "matter go in the ledger or a study's README (`git rm --cached <file>` untracks one)")
    gen = _tracked(root, *GENERATED)
    if gen:
        dirs = sorted({"/".join(p.split("/")[:2]) for p in gen})
        out.append(f"{len(gen)} generated file(s) under {' and '.join(dirs)} tracked by git (remote-run downloads, "
                   f"built kernels): `git rm -r --cached {' '.join(dirs)}` untracks them, .kaggle-gm/.gitignore "
                   "keeps them out")
    if (root / WORK).is_dir():
        r = _git(root, "check-ignore", "-q", f"{WORK}/")
        if r is not None and r.returncode == 1:
            out.append("work/ is not ignored by git: add `/work/` to .gitignore")
    return out


def brief_line(root: str | Path) -> str:
    """The drift as one line of the session brief; '' when there is none."""
    try:
        found = drift(root)
    except Exception:  # the brief must survive anything here
        return ""
    if not found:
        return ""
    return ("Repo layout (it goes public when the competition ends, /kg-showcase): " + "; ".join(found) + ".")


def journey_lines(st, recs: list[dict]) -> list[str]:
    """The phase the work is in and the records with no decision: the journey of the finished competition is told
    from both (`kgkit ledger journey`), so the brief asks for them while the work happens rather than having them
    reconstructed at the end."""
    out = []
    phase = getattr(st, "phase", "") or ""
    if phase:
        n = sum(1 for r in recs if r.get("phase") == phase)
        out.append(f"Phase: \"{phase}\" ({n} record(s) so far). When the work changes direction (a new model family, "
                   "new data, ensembling, the final week), name the new phase: `kgkit ledger phase \"<name>\"`.")
    elif len(recs) >= UNDECIDED_MIN:
        out.append("No phase named: `kgkit ledger phase \"<what the work is about now>\"` groups the records that "
                   "follow, so the competition's journey can be told phase by phase when it ends.")
    undecided = [r for r in recs if not r.get("decision") and not r.get("recheck")]
    if len(undecided) >= UNDECIDED_MIN:
        out.append(f"{len(undecided)} record(s) have no decision (oldest {undecided[0]['id']}): decide each, or a sweep "
                   "at once (`kgkit ledger decide 12..20 discard \"<why>\"`), so the lineage shows how each idea "
                   "fared.")
    return out
