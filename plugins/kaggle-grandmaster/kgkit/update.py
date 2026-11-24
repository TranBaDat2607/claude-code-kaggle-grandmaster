"""`kgkit update-check`: is a newer kaggle-grandmaster released, and how does this session get it (/kg-update).

It compares the running version with the latest GitHub release of the plugin, finds this session's install in
`claude plugin list --json` (the one whose folder kgkit runs from, or, after an update, the same plugin's install for
this project in a newer version folder), and says what to do: the exact `claude plugin update` command (which
`/kg-update` runs: a command of the plugin's hooks module that runs `python -m kgkit.update --apply`, see `main`), then
/reload-plugins, which loads the new version into the running
session and keeps the conversation. After the update it says the new version waits for that reload and which folder
kgkit runs from after it (the paths earlier in the conversation are the old version's), and lists the copies made by
`kgkit vendor` in the workspace that are older.

The only outside request is GitHub's public list of the plugin's releases, with a generic client name. Standard
library only: it must run from a vendored copy, and never fails on a network or CLI error (it says what it could
not read).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from . import __version__

REPO = "TranBaDat2607/claude-code-kaggle-grandmaster"
PLUGIN = "kaggle-grandmaster"
PLUGIN_ROOT = Path(__file__).resolve().parent.parent
RELEASES_URL = f"https://api.github.com/repos/{REPO}/releases?per_page=30"
USER_AGENT = "kgkit-update-check"
MAX_CHANGES = 10
UPDATE_TIMEOUT = 300  # seconds `claude plugin update` may take (it fetches the marketplace)
TAIL_LINES = 8
SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "site-packages", "__pycache__", ".kaggle-gm"}


def parse_version(text: str | None) -> tuple[int, ...] | None:
    """(1, 24, 0) from '1.24.0', 'v1.24.0' or a release tag 'kaggle-grandmaster--v1.24.0'; None when there is none."""
    m = re.search(r"v?(\d+(?:\.\d+)+)$", str(text or "").strip())
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def _fmt(v: tuple[int, ...]) -> str:
    return ".".join(str(x) for x in v)


def notes(release: dict) -> list[str]:
    """The merged PRs a release lists (GitHub's generated notes): their titles."""
    out = []
    for line in str(release.get("body") or "").splitlines():
        line = line.strip()
        if line.startswith(("* ", "- ")):
            out.append(re.sub(r"\s+by @\S+\s+in\s+\S+$", "", line[2:]).strip())
    return out


def _releases_json() -> list | None:
    """GitHub's releases of the plugin, newest first; None when they cannot be read (offline, rate limit)."""
    req = urllib.request.Request(RELEASES_URL, headers={"User-Agent": USER_AGENT,
                                                        "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            rows = json.loads(r.read().decode("utf-8"))
    except Exception:  # any failure: the check says it could not read them
        return None
    return rows if isinstance(rows, list) else None


def _plugin_list() -> list | None:
    """`claude plugin list --json`; None without the claude CLI or when it fails."""
    exe = shutil.which("claude")
    if exe is None:
        return None
    try:
        r = subprocess.run([exe, "plugin", "list", "--json"], capture_output=True, encoding="utf-8",
                           errors="replace", timeout=60)
        rows = json.loads(r.stdout) if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return rows if isinstance(rows, list) else None


def _same(a: str | Path | None, b: str | Path | None) -> bool:
    if not a or not b:
        return False
    return os.path.normcase(os.path.realpath(str(a))) == os.path.normcase(os.path.realpath(str(b)))


def _inside(path: Path, folder: str | None) -> bool:
    if not folder:
        return False
    p, f = (os.path.normcase(os.path.realpath(str(x))) for x in (path, folder))
    return p == f or p.startswith(f.rstrip(os.sep) + os.sep)


def _this_install(installs: list, root: Path, cwd: Path) -> dict | None:
    """The install kgkit runs from; after an update, the same plugin's install (in a sibling version folder) for this
    project, or the user-wide one."""
    mine = [i for i in installs if isinstance(i, dict) and str(i.get("id", "")).split("@")[0] == PLUGIN]
    exact = next((i for i in mine if _same(i.get("installPath"), root)), None)
    if exact is not None:
        return exact
    siblings = [i for i in mine if i.get("installPath") and _same(Path(i["installPath"]).parent, root.parent)]
    here = [i for i in siblings if i.get("scope") in ("project", "local") and _inside(cwd, i.get("projectPath"))]
    return (here or [i for i in siblings if i.get("scope") == "user"] or [None])[0]


def _vendored(workspace: Path | None, root: Path) -> list[tuple[Path, str]]:
    """kgkit copies (`kgkit vendor`) in the workspace: (folder, version), up to two levels down (src/kgkit,
    third_party/kgkit)."""
    if workspace is None or not workspace.is_dir():
        return []
    out = []
    for pattern in ("*/kgkit/__init__.py", "*/*/kgkit/__init__.py"):
        for init in workspace.glob(pattern):
            rel = init.relative_to(workspace).parts
            if any(p in SKIP_DIRS or p.startswith(".") for p in rel[:-2]) or _same(init.parent, root / "kgkit"):
                continue
            try:
                m = re.search(r'^__version__\s*=\s*["\']([^"\']+)', init.read_text(encoding="utf-8"), re.M)
            except OSError:
                continue
            if m:
                out.append((init.parent, m.group(1)))
    return sorted(out, key=lambda x: str(x[0]))


def check(root: str | Path | None = None, cwd: str | Path | None = None, running: str | None = None,
          workspace: str | Path | None = "auto") -> dict:
    """What `report` tells: the running and latest versions, this install and its kind (marketplace, dev, vendored,
    unknown, no-cli), the update command, a version installed but not loaded yet, the changes in between, and the
    older vendored copies."""
    root = Path(root or PLUGIN_ROOT)
    cwd = Path(cwd or os.getcwd())
    running = running or __version__
    run_v = parse_version(running) or (0,)
    releases = sorted(((v, r) for r in (_releases_json() or []) if isinstance(r, dict)
                       for v in [parse_version(r.get("tag_name"))] if v is not None), key=lambda x: x[0], reverse=True)
    latest_v = releases[0][0] if releases else None
    installs = _plugin_list()
    mine = _this_install(installs, root, cwd) if installs is not None else None
    if mine is not None:
        kind = "marketplace"
    elif not (root / ".claude-plugin").is_dir() and (root / "kgkit").is_dir():
        kind = "vendored"
    elif any((p / ".git").exists() for p in (root, *root.parents)):
        kind = "dev"
    else:
        kind = "no-cli" if installs is None else "unknown"
    mine_v = parse_version(mine.get("version")) if mine else None
    pending = mine if mine_v and mine_v > run_v else None
    command = run_from = args = None
    if mine is not None and pending is None and (latest_v is None or latest_v > run_v):
        args = ["plugin", "update", str(mine["id"]), "--scope", str(mine.get("scope") or "user")]
        command = "claude " + " ".join(args)
        if mine.get("scope") in ("project", "local") and not _inside(cwd, mine.get("projectPath")):
            run_from = mine.get("projectPath")
    target = mine_v if pending else latest_v
    changes = [(_fmt(v), notes(r)) for v, r in releases if run_v < v <= (target or run_v)][:MAX_CHANGES]
    if workspace == "auto":
        from . import state as S

        workspace = S.find_root(cwd) or cwd
    floor = target or run_v
    vendored = [(p, v) for p, v in _vendored(Path(workspace) if workspace else None, root)
                if (parse_version(v) or (0,)) < floor]
    return {"running": running, "latest": _fmt(latest_v) if latest_v else None, "kind": kind, "install": mine,
            "pending": pending, "command": command, "args": args, "run_from": run_from, "changes": changes,
            "root": str(root), "workspace": str(workspace) if workspace else None, "vendored": vendored}


def report(info: dict) -> str:
    """What to do, in words: the lines `kgkit update-check` prints (ASCII only)."""
    running, latest, kind = info["running"], info["latest"], info["kind"]
    lines = [f"kaggle-grandmaster: this session runs {running}"
             + (f"; the latest release is {latest}" if latest else "") + f" (kgkit at {info['root']})"]
    pending, mine = info["pending"], info["install"]
    new_root = pending.get("installPath") if pending else None
    if kind == "no-cli":
        lines.append("The claude CLI was not found (or `claude plugin list --json` failed), so this install is "
                     "unknown: update from Claude Code's /plugin menu, then type /reload-plugins.")
    elif kind == "vendored":
        lines.append("This kgkit is a copy made by `kgkit vendor`: update the plugin itself (/kg-update in Claude "
                     "Code), then refresh the copy with the plugin's kgkit (`kgkit vendor <folder>`).")
    elif kind == "dev":
        lines.append(f"This is a development checkout ({info['root']}): `git pull` there updates it; a restart or "
                     "/reload-plugins loads it.")
    elif kind == "unknown":
        lines.append(f"No installed plugin points to {info['root']} (loaded with --plugin-dir?): replace that folder "
                     "with the latest release, or install it from the marketplace.")
    elif pending:
        where = "for this project" if pending.get("scope") in ("project", "local") else "for your user"
        lines.append(f"{pending.get('version')} is installed {where}; this session still runs {running}. "
                     "Next: type /reload-plugins to load it (the conversation stays).")
        lines.append(f'After the reload, run kgkit as `PYTHONPATH="{new_root}" python -m kgkit <command>`: the plugin '
                     f"paths earlier in this conversation are {running}'s.")
    elif info["command"] is None:
        lines.append(f"kaggle-grandmaster is up to date ({running}).")
    else:
        lines.append(f"Update: {running} -> {latest}" if latest else
                     "Update: kgkit could not read the latest release from GitHub (offline, or its rate limit); the "
                     "command below installs whatever is latest anyway.")
        lines.append(f"Run: {info['command']}"
                     + (f"   (from {info['run_from']}: it is a {mine.get('scope')} install of that project)"
                        if info["run_from"] else ""))
        lines.append("Then type /reload-plugins: the session loads the new version and the conversation stays.")
    if info["changes"] and kind == "marketplace":
        lines.append("What changed:")
        for v, titles in info["changes"]:
            lines.append(f"  {v}: " + ("; ".join(titles) if titles else "(no notes)"))
    if info["vendored"]:
        ws = Path(info["workspace"] or ".")
        rels = [p.relative_to(ws).as_posix() if _inside(p, str(ws)) else p.as_posix() for p, _ in info["vendored"]]
        lines.append("Older kgkit copies in this workspace (`kgkit vendor`): "
                     + ", ".join(f"{r} {v}" for r, (_, v) in zip(rels, info["vendored"])) + ".")
        if new_root:
            lines += [f'  refresh after the reload: PYTHONPATH="{new_root}" python -m kgkit vendor '
                      f"{Path(r).parent.as_posix()}" for r in rels]
        else:
            lines.append("  refresh them after the update: update-check then prints the commands.")
    return "\n".join(lines).encode("ascii", "replace").decode("ascii")


def apply(info: dict) -> tuple[bool, str]:
    """Runs the `claude plugin update` that `check` found (from the project folder it names): whether it succeeded,
    and the last lines it printed."""
    exe = shutil.which("claude")
    if exe is None or not info.get("args"):
        return False, "the claude CLI was not found"
    try:
        r = subprocess.run([exe, *info["args"]], capture_output=True, encoding="utf-8", errors="replace",
                           timeout=UPDATE_TIMEOUT, cwd=info.get("run_from") or None)
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)
    lines = [ln.rstrip() for ln in f"{r.stdout}\n{r.stderr}".splitlines() if ln.strip()]
    tail = "\n".join("  " + ln for ln in lines[-TAIL_LINES:])
    return r.returncode == 0, (tail or f"  (exit code {r.returncode}, nothing printed)")


def run(do_apply: bool) -> dict:
    """What /kg-update shows: the check, and with `do_apply` the update it then runs and the check after it.
    `{"text", "updated"}`; the text is ASCII."""
    info = check()
    if not (do_apply and info["command"]):
        text = report(info)
        if info["command"]:
            text += "\n/kg-update runs that command for you."
        return {"text": text, "updated": False}
    ok, out = apply(info)
    if not ok:
        text = report(info) + f"\nThe update failed ({info['command']}):\n{out}"
        return {"text": text.encode("ascii", "replace").decode("ascii"), "updated": False}
    return {"text": f"Updated: ran {info['command']}\n" + report(check()), "updated": True}


def main(argv: list[str] | None = None) -> int:
    """`python -m kgkit.update [--apply]`, what the /kg-update command of the plugin's hooks module runs: one JSON line,
    `run`'s answer. Without --apply it only checks."""
    args = sys.argv[1:] if argv is None else argv
    print(json.dumps(run("--apply" in args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
