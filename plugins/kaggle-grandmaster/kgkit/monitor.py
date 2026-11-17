"""Adds the hardware monitor (kgkit/kaggle_monitor_cell.py) to a hand-written kernel's code before it is pushed.

The kaggle-monitor pane draws a running kernel's CPU, RAM, disk and GPUs from the `[kgmon] {...}` lines its log
holds; Kaggle's API reports no session resources. Kernels built by `kgkit gpu build` print them already (the runner).
For any other kernel, `gpu push` and the push hook (hooks/_kgmon.py, for a `kaggle kernels push` Claude runs) put
the cell into the code file kernel-metadata.json names: as the first cell of a notebook, or after a script's shebang,
encoding line, docstring and `from __future__` imports, between BEGIN/END markers. The cell is filled in with the
GPUs the pushed machine shape promises, so a session that lands on a machine with fewer stops at once (#113); a block
an earlier push added is brought up to date (the shape may have changed). A code file that mentions kgmon anywhere
else (the runner, a pasted cell, a `# kgmon: off` comment) is left alone.

Standard library only: the push hook imports it.
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from pathlib import Path

from . import gpu as G

CELL = Path(__file__).resolve().parent / "kaggle_monitor_cell.py"
BEGIN = "# >>> kaggle-grandmaster hardware monitor: [kgmon] lines for /kaggle-monitor (`# kgmon: off` opts out) >>>"
END = "# <<< kaggle-grandmaster hardware monitor <<<"
SKIP_MARK = re.compile(r"kgmon", re.IGNORECASE)
EXPECT_RE = re.compile(r"^KGMON_EXPECT_GPUS = .*$", re.MULTILINE)
CODING_RE = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+")
FUTURE_RE = re.compile(r"^[ \t]*from[ \t]+__future__[ \t]+import\b", re.MULTILINE)


def cell_lines(accelerator: str | None = None) -> list[str] | None:
    """The monitor cell between its markers, as lines without line ends, expecting the GPUs ``accelerator``
    promises; None when the cell file is missing."""
    try:
        body = CELL.read_text(encoding="utf-8").replace("\r\n", "\n")  # CRLF in a Windows checkout
    except OSError:
        return None
    body = EXPECT_RE.sub(f"KGMON_EXPECT_GPUS = {G.expected_gpus(accelerator)}", body, count=1)
    return [BEGIN, *body.rstrip("\n").split("\n"), END]


def _script_insert_at(text: str) -> int | None:
    """Index of the line the cell goes before in a Python script: after the shebang, encoding line, module docstring
    and `from __future__` imports. None when the script cannot be parsed and has a __future__ import (the cell
    would have to go after it, and where that ends cannot be told)."""
    lines = text.split("\n")
    try:
        body = ast.parse(text).body
    except (SyntaxError, ValueError):
        body = None
    end = 0
    if body:
        i = 0
        if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            end, i = body[0].end_lineno or 0, 1
        while i < len(body) and isinstance(body[i], ast.ImportFrom) and body[i].module == "__future__":
            end, i = body[i].end_lineno or end, i + 1
    elif FUTURE_RE.search(text):
        return None
    if end == 0:
        if lines and lines[0].startswith("#!"):
            end = 1
        if end < len(lines) and end < 2 and CODING_RE.match(lines[end]):
            end += 1
    return end


def _with_script_cell(raw: str, cell: list[str]) -> str | None:
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.replace("\r\n", "\n").split("\n")
    if BEGIN in lines and END in lines[lines.index(BEGIN):]:  # an earlier push's block: bring it up to date
        a = lines.index(BEGIN)
        b = lines.index(END, a)
        return nl.join(lines[:a] + cell + lines[b + 1:])
    at = _script_insert_at("\n".join(lines))
    if at is None:
        return None
    return nl.join(lines[:at] + cell + [""] + lines[at:])


def _with_notebook_cell(raw: str, cell: list[str]) -> str | None:
    try:
        nb = json.loads(raw)
    except ValueError:
        return None
    cells = nb.get("cells") if isinstance(nb, dict) else None
    if not isinstance(cells, list):
        return None
    source = [ln + "\n" for ln in cell[:-1]] + [cell[-1]]
    old = next((c for c in cells if isinstance(c, dict) and "".join(c.get("source") or []).startswith(BEGIN)), None)
    if old is not None:  # an earlier push's cell: bring it up to date
        old["source"] = source
    else:
        new = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}
        if any(isinstance(c, dict) and "id" in c for c in cells):
            new["id"] = "kgmon-" + uuid.uuid4().hex[:8]
        cells.insert(0, new)
    m = re.match(r"\{\r?\n( +)\"", raw)
    if m:
        out = json.dumps(nb, indent=len(m.group(1)), ensure_ascii=False)
    else:
        compact = re.search(r'"\s*:\s', raw[:2000]) is None  # Kaggle's own .ipynb: {"metadata":{...},"cells":...}
        out = json.dumps(nb, ensure_ascii=False, separators=(",", ":") if compact else None)
    return out + ("\n" if raw.endswith("\n") else "")


def add_monitor(folder: Path, meta: dict, accelerator: str | None = None) -> str | None:
    """Put the monitor cell into the code file of the kernel in ``folder``, expecting the GPUs ``accelerator``
    (default: the metadata's machine shape) promises. A note about what was done or why not; None when there is
    nothing to say (already up to date, opted out, or no Python code)."""
    name = meta.get("code_file")
    lang = str(meta.get("language") or "python").lower()
    if not isinstance(name, str) or lang != "python":
        return None
    path = Path(folder) / name
    suffix = path.suffix.lower()
    if suffix not in (".py", ".ipynb"):
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    bom = data.startswith(b"\xef\xbb\xbf")
    try:
        raw = data[3 if bom else 0:].decode("utf-8")
    except UnicodeDecodeError:
        return f"no hardware monitor added to {path}: it is not UTF-8."
    had = BEGIN in raw
    if SKIP_MARK.search(raw) and not had:
        return None
    acc = accelerator or G.kernel_meta_accelerator(meta)
    cell = cell_lines(acc)
    if cell is None:
        return None
    new = _with_notebook_cell(raw, cell) if suffix == ".ipynb" else _with_script_cell(raw, cell)
    if new is None:
        why = ("its JSON could not be read" if suffix == ".ipynb"
               else "it does not parse here and has a `from __future__` import the cell must follow")
        return f"no hardware monitor added to {path}: {why}. Paste kgkit/kaggle_monitor_cell.py in by hand."
    if new == raw:
        return None
    out = (b"\xef\xbb\xbf" if bom else b"") + new.encode("utf-8")
    if len(out) >= G.SOURCE_MAX_BYTES > len(data):
        return f"no hardware monitor added to {path}: with it the source would reach 1 MB, which Kaggle refuses."
    path.write_bytes(out)
    want = G.expected_gpus(acc)
    check = (f" It stops the session at once if the machine has fewer than the {want} GPU{'s' if want > 1 else ''} "
             f"{acc} promises (KG_BAD_MACHINE in the log)." if want else "")
    if had:
        return f"updated the hardware monitor in {path} for {acc}, so the file changed on disk.{check}"
    where = "as its first cell" if suffix == ".ipynb" else "at the top"
    return (f"added the hardware monitor to {path} {where}, between `kaggle-grandmaster hardware monitor` markers, "
            "so the file changed on disk. The session log gets a [kgmon] line every 30 s (CPU, RAM, disk, GPUs) "
            f"for /kaggle-monitor.{check} Keep the block; to push without it, delete it and add a `# kgmon: off` "
            "comment.")
