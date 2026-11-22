"""Allow rules for the kgkit calls a competition session makes all day (#123).

A kg-* command's ``allowed-tools`` pre-approves its kgkit calls only until the user's next message, and never in the
subagents it starts. So `/kg-start` after a question, `/kg-grind` after "continue", and every agent asked before each
`kgkit status`. Rules in the workspace's ``.claude/settings.local.json`` hold for the whole session, every later
session and subagents; that file is personal and git-ignored by Claude Code, so nothing here reaches the public repo.

``RULES`` are the local bookkeeping and read-only calls. Left out on purpose: whatever spends quota, deletes or
publishes (``gpu push`` / ``repush`` / ``cancel``, ``vendor``), runs a command it is given (``bench``, ``recheck``,
``sweep run``: approving them would approve that command), and ``folds`` (a frozen split is never redrawn without asking). Kaggle
CLI submits, pushes and uploads are not kgkit calls: the hooks and the user still gate them.

Standard library only.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

# subcommand -> also allowed with no arguments (Bash(kgkit status) next to Bash(kgkit status *))
SUBCOMMANDS: dict[str, bool] = {
    "status": True, "ledger": False, "exp": False, "backlog": False, "final": True, "standing": True, "config": True,
    "eda": False, "adv": False, "validate": False, "metrics": True, "score": False, "speedcheck": False,
    "blend": False, "features": False, "pages": True, "discussions": False, "kernels top": True,
    "kernels pull": False, "datasets": False, "audit-repo": True, "whoami": True, "update-check": True,
    "gpu quota": True, "gpu plan": True, "gpu status": True, "gpu log": True, "gpu wait": False,
    "gpu collect": False, "gpu build": False, "gpu release": False, "grind": False, "screen": False,
    "sweep status": True, "sweep decide": False,
}
NEVER = ("gpu push", "gpu repush", "gpu cancel", "vendor", "bench", "recheck", "sweep run", "folds", "init", "adopt")
SETTINGS = Path(".claude") / "settings.local.json"


def rules() -> list[str]:
    out = []
    for sub, bare in SUBCOMMANDS.items():
        if bare:
            out.append(f"Bash(kgkit {sub})")
        out.append(f"Bash(kgkit {sub} *)")
    return out


def _allowed(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    allow = ((data or {}).get("permissions") or {}).get("allow") if isinstance(data, dict) else None
    return [r for r in allow if isinstance(r, str)] if isinstance(allow, list) else []


def present(root: Path) -> set[str]:
    """The kgkit rules already allowed in this workspace's settings files or the user's."""
    files = [root / ".claude" / "settings.local.json", root / ".claude" / "settings.json",
             Path.home() / ".claude" / "settings.json"]
    return {r for f in files for r in _allowed(f)} & set(rules())


def missing(root: Path) -> list[str]:
    have = present(root)
    return [r for r in rules() if r not in have]


def ensure_ignored(root: Path) -> bool:
    """Make git ignore ``.claude/settings.local.json`` in ``root`` (Claude Code does when it creates the file, but
    kgkit may create it first; the repo goes public later). True when a line was added to .gitignore."""
    rel = SETTINGS.as_posix()
    try:
        r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=root, capture_output=True, timeout=20)
        if r.returncode == 0:
            return False  # ignored already
        if r.returncode not in (0, 1):  # 128: not a git repository, or git failed: the file is not committed anyway
            return False
    except (OSError, subprocess.SubprocessError):
        return False
    gi = root / ".gitignore"
    raw = gi.read_bytes() if gi.is_file() else b""
    nl = "\r\n" if b"\r\n" in raw else "\n"
    lead = nl if raw and not raw.endswith(b"\n") else ""
    with open(gi, "a", encoding="utf-8", newline="") as f:
        f.write(f"{lead}# Claude Code settings of this machine only (kgkit permissions){nl}/{rel}{nl}")
    return True


def write(root: Path) -> tuple[Path, list[str]]:
    """Merge ``rules()`` into ``<root>/.claude/settings.local.json``: every other setting and rule is kept, nothing is
    added twice. Returns (the file, the rules added). Raises ValueError on a file that is not a JSON object."""
    path = root / SETTINGS
    data: dict = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
        except ValueError as e:
            raise ValueError(f"{path} is not valid JSON ({e}): fix it first, nothing was changed") from e
        if not isinstance(data, dict):
            raise ValueError(f"{path} is not a JSON object: nothing was changed")
    perms = data.setdefault("permissions", {})
    if not isinstance(perms, dict) or not isinstance(perms.setdefault("allow", []), list):
        raise ValueError(f"{path}: permissions.allow is not a list: nothing was changed")
    added = [r for r in rules() if r not in perms["allow"]]
    if added:
        perms["allow"].extend(added)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    ensure_ignored(root)
    return path, added


def status_line(root: Path) -> str:
    """One line for `kgkit status` when the workspace lacks the rules, '' otherwise."""
    gone = missing(root)
    if not gone:
        return ""
    return (f"Permissions: {len(gone)} of {len(rules())} kgkit allow rules are not set, so kgkit calls after your next "
            "message (and in subagents) ask each time. `kgkit permissions` lists them; `kgkit permissions --write` "
            "adds them to .claude/settings.local.json (ask the user first).")
