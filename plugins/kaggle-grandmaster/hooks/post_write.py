"""PostToolUse (Write|Edit): when model code is written in a competition workspace, put the compute defaults in front
of Claude.

A skill loads when the model decides to load it; this hook makes the compute-efficiency guidance arrive whenever a
training or inference script is written, by the main session or by a subagent (which may have no Skill tool, so
the note names the reference file's path). It fires for a ``.py`` file inside a workspace that trains or runs a model
(``kgkit.speedcheck.kind``):

- once per session and domain: the domain's defaults and the reference to read;
- each time the file has a slow pattern not reported yet in this session: the pattern, its line and the fix
  (``kgkit.speedcheck.check``, ``warn`` level only).

State is kept in the temp dir, keyed by session_id. Standard library only; exits 0 on any error and never blocks.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

try:
    import _common as C
    from kgkit import speedcheck as SC
except Exception:  # pragma: no cover - fail open
    sys.exit(0)

MAX_BYTES = 1_000_000


def _state_file(payload: dict) -> Path | None:
    sid = re.sub(r"[^A-Za-z0-9_-]", "", str(payload.get("session_id") or ""))[:64]
    return Path(tempfile.gettempdir()) / "kaggle-gm-router" / f"{sid}.compute.json" if sid else None


def _seen(path: Path | None) -> set:
    if path is None:
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return set()


def _remember(path: Path | None, seen: set) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(seen)), encoding="utf-8")
    except OSError:
        pass


def note_for(path: Path, text: str, rel: str, seen: set, task_domain: str | None = None) -> tuple:
    """(note, keys to remember) for a written file; ('', set()) when there is nothing new to say."""
    k = SC.kind(text)
    if k is None:
        return "", set()
    dom = SC.domain(text) or task_domain
    keys, parts = set(), []
    if dom in SC.DOMAINS and f"domain:{dom}" not in seen:
        ref, defaults = SC.DOMAINS[dom]
        ref_path = (C.PLUGIN_ROOT / "skills" / "compute-efficiency" / "references" / ref).as_posix()
        verb = "trains" if k == "train" else "runs"
        parts.append(f"{rel} {verb} a {dom} model. Apply the compute defaults before running it (skill "
                     f"kaggle-grandmaster:compute-efficiency; reference: {ref_path}): {defaults}. Measure before a "
                     f"long run: `--bench` in the plugin's templates, or `python -m kgkit bench --run \"<command>\"`.")
        keys.add(f"domain:{dom}")
    new = [f for f in SC.check(text) if f.level == "warn" and f"{rel}:{f.code}" not in seen]
    if new:
        found = "; ".join(f"line {f.line}: {f.message}" for f in new[:5])
        parts.append(f"Slow patterns in {rel}: {found}. Fix them, or say in a comment why the script differs.")
        keys |= {f"{rel}:{f.code}" for f in new}
    return " ".join(parts), keys


def main() -> None:
    payload = C.read_input()
    ti = payload.get("tool_input") or {}
    name = ti.get("file_path") or ti.get("path") or ""
    if not str(name).lower().endswith(".py"):
        return
    root = C.workspace(payload)
    if root is None:
        return
    path = Path(name)
    if not path.is_absolute():
        path = Path(payload.get("cwd") or ".") / path
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return  # a file outside the workspace
    try:
        if path.stat().st_size > MAX_BYTES:
            return
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ti.get("content") or ""
    if not text:
        return
    state_file = _state_file(payload)
    seen = _seen(state_file)
    try:
        st = C.S.load(root)
        task = (st.task or "").strip().lower() if st else ""
    except Exception:
        task = ""
    task_domain = {"cv": "vision", "nlp": "text", "llm": "llm", "tabular": "tabular", "timeseries": "tabular",
                   "audio": "audio"}.get(task)
    note, keys = note_for(path, text, rel, seen, task_domain)
    if not note:
        return
    _remember(state_file, seen | keys)
    C.emit("PostToolUse", additionalContext=f"[kaggle-grandmaster] {note}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
