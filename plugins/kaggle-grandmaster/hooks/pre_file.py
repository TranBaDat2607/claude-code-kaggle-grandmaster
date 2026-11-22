"""PreToolUse guard for the file tools (Read, Grep, Glob, Edit, Write, NotebookEdit).

Denies a call whose path names the sealed vault's folder (``~/.kaggle-gm/vault/``, ``kgkit.vault``): its labels must
never reach the agent, and its check log must not be edited. ``kgkit vault check`` reads them itself and answers PASS
or FAIL only. Grep's search pattern is text, not a path, and is not looked at. Fails open: any internal error exits 0
with no output.
"""

from __future__ import annotations

import sys

try:
    import _common as C
except Exception:  # pragma: no cover - fail open
    sys.exit(0)

PATH_KEYS = ("file_path", "path", "notebook_path")


def paths(payload: dict) -> list[str]:
    ti = payload.get("tool_input") or {}
    out = [str(ti[k]) for k in PATH_KEYS if isinstance(ti.get(k), str)]
    if payload.get("tool_name") == "Glob" and isinstance(ti.get("pattern"), str):
        out.append(ti["pattern"])  # a glob is a path pattern; Grep's pattern is the text searched for
    return out


def main() -> None:
    payload = C.read_input()
    if any(C.V.names_vault(p) for p in paths(payload)):
        C.emit("PreToolUse", permissionDecision="deny",
               permissionDecisionReason="kaggle-grandmaster: " + C.V.DENY_REASON)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - a hook never fails closed
        pass
    sys.exit(0)
