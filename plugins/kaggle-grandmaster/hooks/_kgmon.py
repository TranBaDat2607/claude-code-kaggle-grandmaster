"""Adds the hardware monitor (kgkit/kaggle_monitor_cell.py) to a kernel's code before a `kaggle kernels push` Claude
runs: finds each real push in the command and its folder, and leaves the file editing to `kgkit.monitor.add_monitor`
(which `kgkit gpu push` uses too), with the accelerator the push asks for. Standard library only; the caller fails
open.
"""

from __future__ import annotations

import re
from pathlib import Path

import _common as C
from kgkit import monitor as MON  # noqa: E402  (stdlib-only module)

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
                         "kgkit/kaggle_monitor_cell.py into its code by hand.")
            continue
        # no kernel-metadata.json: Kaggle refuses the push
        note = MON.add_monitor(*target, push.get("accelerator")) if target is not None else None
        if note:
            notes.append(note)
    return notes
