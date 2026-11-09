"""Adds the hardware monitor (templates/kaggle_monitor_cell.py) to a kernel's code before `kaggle kernels push`.

The kaggle-monitor pane draws a running kernel's CPU, RAM, disk and GPUs from the `[kgmon] {...}` lines its log
holds; Kaggle's API reports no session resources. Kernels built by `kgkit gpu build` print them already. For any
other kernel Claude pushes, `add_monitors` puts the cell into the code file kernel-metadata.json names: as the first
cell of a notebook, or after a script's shebang, encoding line, docstring and `from __future__` imports. A code file
that mentions kgmon anywhere (the runner, the cell, an earlier insert, a `# kgmon: off` comment) is left alone.
Standard library only; the caller fails open.
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from pathlib import Path

import _common as C

CELL = C.PLUGIN_ROOT / "templates" / "kaggle_monitor_cell.py"
BEGIN = "# >>> kaggle-grandmaster hardware monitor: [kgmon] lines for /kaggle-monitor (`# kgmon: off` opts out) >>>"
END = "# <<< kaggle-grandmaster hardware monitor <<<"
SKIP_MARK = re.compile(r"kgmon", re.IGNORECASE)
CODING_RE = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+")
FUTURE_RE = re.compile(r"^[ \t]*from[ \t]+__future__[ \t]+import\b", re.MULTILINE)
SHELL_CMDS = {"bash", "sh", "zsh", "dash", "ksh", "pwsh", "powershell", "powershell.exe", "pwsh.exe", "cmd",
              "cmd.exe"}
# Programs that run the command after them: `nohup kaggle ...`, `conda run -n kg kaggle ...`.
RUNNERS = {"env", "nohup", "timeout", "sudo", "exec", "nice", "stdbuf", "xargs", "uv", "poetry", "pipx", "conda",
           "mamba", "micromamba", "run"}


def is_call(command: str, at: int) -> bool:
    """Whether the `kaggle ...` that starts at offset ``at`` of ``command`` runs as a program, rather than being
    text: an argument of git, gh, echo or grep, or a quoted string. A `bash -c "..."` script counts as a call."""
    text = C._shell_text(command)
    _, quote = C._shell_words(text[:at])
    close = text.find(text[quote], at) if quote is not None else -1
    if close > 0 and re.fullmatch(r"kaggle(?:\.exe)?", text[at:close], re.IGNORECASE):
        return _runs_here(text[:quote])  # a quoted program path: & "C:/Program Files/.../kaggle.exe" kernels push
    start = at
    while start > 0 and not text[start - 1].isspace() and text[start - 1] not in "&|;()'\"`":
        start -= 1  # the start of the word: `/opt/conda/bin/kaggle`
    return _runs_here(text[:start])


def _runs_here(head: str) -> bool:
    words, quote = C._shell_words(head)
    if quote is not None:  # in a quoted argument: a call only in a shell's -c / -Command script
        outer = C._command_words(words)
        flags = {w.lower() for w in outer[1:]}
        if not (outer and Path(outer[0]).name.lower() in SHELL_CMDS and flags & {"-c", "-command", "/c", "-lc"}):
            return False
        return _runs_here(head[quote + 1:])
    cmd = C._command_words(words)
    return not cmd or Path(cmd[0]).name.lower() in RUNNERS  # its words before kaggle are the runner's options


def _cell_lines() -> list[str] | None:
    """The monitor cell between its markers, as lines without line ends; None when the template is missing."""
    try:
        body = CELL.read_text(encoding="utf-8")
    except OSError:
        return None
    return [BEGIN, *body.replace("\r\n", "\n").rstrip("\n").split("\n"), END]  # CRLF in a Windows checkout


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
    text = raw.replace("\r\n", "\n")
    at = _script_insert_at(text)
    if at is None:
        return None
    lines = text.split("\n")
    new = lines[:at] + cell + [""] + lines[at:]
    return nl.join(new)


def _with_notebook_cell(raw: str, cell: list[str]) -> str | None:
    try:
        nb = json.loads(raw)
    except ValueError:
        return None
    cells = nb.get("cells") if isinstance(nb, dict) else None
    if not isinstance(cells, list):
        return None
    new = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
           "source": [ln + "\n" for ln in cell[:-1]] + [cell[-1]]}
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


def add_monitor(folder: Path, meta: dict) -> str | None:
    """Put the monitor cell into the code file of the kernel in ``folder``. A note for Claude about what was done
    or why not; None when there is nothing to say (the code already reports, or the push has no Python code)."""
    name = meta.get("code_file")
    lang = str(meta.get("language") or "python").lower()
    if not isinstance(name, str) or lang != "python":
        return None
    path = folder / name
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
    if SKIP_MARK.search(raw):
        return None
    cell = _cell_lines()
    if cell is None:
        return None
    new = _with_notebook_cell(raw, cell) if suffix == ".ipynb" else _with_script_cell(raw, cell)
    if new is None:
        why = ("its JSON could not be read" if suffix == ".ipynb"
               else "it does not parse here and has a `from __future__` import the cell must follow")
        return f"no hardware monitor added to {path}: {why}. Paste templates/kaggle_monitor_cell.py in by hand."
    out = (b"\xef\xbb\xbf" if bom else b"") + new.encode("utf-8")
    if len(out) >= C.G.SOURCE_MAX_BYTES > len(data):
        return (f"no hardware monitor added to {path}: with it the source would reach 1 MB, which Kaggle refuses.")
    path.write_bytes(out)
    where = "as its first cell" if suffix == ".ipynb" else "at the top"
    return (f"added the hardware monitor to {path} {where}, between `kaggle-grandmaster hardware monitor` markers, "
            "so the file changed on disk. The session log gets a [kgmon] line every 30 s (CPU, RAM, disk, GPUs) "
            "for /kaggle-monitor. Keep the block; to push without it, delete it and add a `# kgmon: off` comment.")


def add_monitors(payload: dict, command: str) -> list[str]:
    """Notes for Claude after adding the monitor to the code of each `kaggle kernels push` in ``command``."""
    notes = []
    for push in C.parse_pushes(C._text_heredocs_blanked(command)):
        if not is_call(command, push.get("at", 0)):
            continue
        target = C.push_target(payload, push)
        if target is None and C.path_at(payload, push.get("path"), push.get("at", 0)) is None:
            notes.append("no hardware monitor checked for a `kaggle kernels push` whose folder the hook cannot tell "
                         "(a `cd` with a variable, in a loop or in a pipe): push with an absolute -p path, or paste "
                         "templates/kaggle_monitor_cell.py into its code by hand.")
            continue
        note = add_monitor(*target) if target is not None else None  # no kernel-metadata.json: Kaggle refuses it
        if note:
            notes.append(note)
    return notes
