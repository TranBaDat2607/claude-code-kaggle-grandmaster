"""Shared helpers for kaggle-grandmaster hooks. Standard library only; every entry point fails open."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from kgkit import adopt as A  # noqa: E402  (stdlib-only module)
from kgkit import backlog as B  # noqa: E402  (stdlib-only module)
from kgkit import evalset as EV  # noqa: E402  (stdlib-only module)
from kgkit import gpu as G  # noqa: E402  (stdlib-only module)
from kgkit import safety as SF  # noqa: E402  (stdlib-only module)
from kgkit import state as S  # noqa: E402  (stdlib-only module)
from kgkit.state import (server_remaining_today, submission_records, submissions_log,  # noqa: E402,F401
                         submissions_today, today_utc)

SUBMIT_RE = re.compile(r"kaggle(?:\.exe)?\s+(?:competitions|c)\s+submit\b(.*)", re.IGNORECASE | re.DOTALL)
PUSH_RE = re.compile(r"kaggle(?:\.exe)?\s+kernels\s+(?:push|update)\b(.*)", re.IGNORECASE | re.DOTALL)
# Commands that upload a whole folder (Kaggle CLI 2.2.4; `d`, `m`, `i`/`v` and `v` are its aliases).
UPLOAD_RE = re.compile(r"kaggle(?:\.exe)?\s+(?:(?:datasets|d)\s+(?:create|version)"
                       r"|(?:models|m)\s+(?:instances|i|variations|v)\s+(?:(?:versions|v)\s+)?create)\b(.*)",
                       re.IGNORECASE | re.DOTALL)


def read_input() -> dict:
    try:
        return json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, ValueError):
        return {}


def emit(event: str, **fields) -> None:
    sys.stdout.write(json.dumps({"hookSpecificOutput": {"hookEventName": event, **fields}}))
    sys.stdout.flush()


def command_of(payload: dict) -> str:
    ti = payload.get("tool_input") or {}
    return ti.get("command") or ""


def workspace(payload: dict) -> Path | None:
    cwd = payload.get("cwd") or os.getcwd()
    try:
        return S.find_root(cwd)
    except OSError:
        return None


def _tokens(rest: str) -> list[str]:
    rest = _shell_text(rest)  # Windows paths: POSIX lexing would treat \ as an escape
    rest = rest[:_line_end(rest)]  # a newline ends the command too: the next line's flags are not its own
    try:
        lex = shlex.shlex(rest, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        toks = list(lex)
    except ValueError:  # an unclosed quote, e.g. the end of `bash -c "cd A && kaggle kernels push -p ."`
        toks = [t.strip("'\"") for t in rest.split()]
    # stop at the next shell separator (quote-aware: separators inside quotes stay in their token)
    for i, t in enumerate(toks):
        if t in {"&&", "||", ";", "|", "&", ";;"}:
            return toks[:i]
    return toks


def _line_end(text: str) -> int:
    """Offset of the first newline or comment in `text` outside quotes: the end of the command on its first line."""
    quote, prev = None, " "
    for i, c in enumerate(text):
        if quote:
            quote = None if c == quote else quote
        elif c in "'\"":
            quote = c
        elif c == "\n" or (c == "#" and prev in " \t"):
            return i
        prev = c
    return len(text)


def parse_pushes(command: str) -> list[dict]:
    """Every `kaggle kernels push` in a (possibly compound) command, `--help` calls left out. ``at`` is where
    each one starts, so its folder is read after the `cd` steps before it (`path_at`)."""
    out, pos = [], 0
    while True:
        m = PUSH_RE.search(command, pos)
        if not m:
            return out
        pos = m.start() + 1
        push = _parse_push_args(m.group(1))
        if push is not None:
            push["at"] = m.start()
            out.append(push)


def parse_push(command: str) -> dict | None:
    """Folder / timeout / accelerator of the first real `kaggle kernels push` in a command."""
    pushes = parse_pushes(command)
    return pushes[0] if pushes else None


def _parse_push_args(rest: str) -> dict | None:
    toks = _tokens(rest)
    if "-h" in toks or "--help" in toks:
        return None
    out = {"path": None, "timeout": None, "accelerator": None}
    flags = {"-p": "path", "--path": "path", "-t": "timeout", "--timeout": "timeout", "--accelerator": "accelerator"}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in flags and i + 1 < len(toks):
            out[flags[t]] = toks[i + 1]
            i += 2
            continue
        if "=" in t and t.split("=", 1)[0] in flags:
            k, v = t.split("=", 1)
            out[flags[k]] = v
        i += 1
    return out


def parse_upload(command: str) -> dict | None:
    """Folder / dir-mode / ignore patterns of a dataset or model upload command."""
    m = UPLOAD_RE.search(command)
    if not m:
        return None
    toks = _tokens(m.group(1))
    out = {"path": None, "dir_mode": "skip", "ignore": [], "at": m.start()}
    i = 0
    while i < len(toks):
        t, nxt = toks[i], (toks[i + 1] if i + 1 < len(toks) else None)
        k, _, v = t.partition("=")
        if k in ("-p", "--path", "-r", "--dir-mode", "--ignore-patterns"):
            if not v:
                v, i = nxt, i + 1
            if v is not None:
                if k in ("-p", "--path"):
                    out["path"] = v
                elif k == "--ignore-patterns":
                    out["ignore"].append(v)
                else:
                    out["dir_mode"] = v
        i += 1
    return out


# A relative path in a compound command is relative to wherever the `cd` steps before it left the shell, not to
# the hook's `cwd` (where the command starts for PreToolUse, where it ended for PostToolUse). Commands that move
# the shell:
CD_CMDS = {"cd", "chdir", "pushd", "set-location", "sl", "push-location"}
POP_CMDS = {"popd", "pop-location"}
SHELL_OPS = ("&&", "||", ";;", "|&", ";", "|", "&", "\n", "(", ")")
# Words before a command name. A `cd` after an opener sits in a branch, a loop or a function: it may not run.
OPENERS = {"if", "while", "until", "for", "case", "select", "function", "foreach", "switch", "try"}
CLOSERS = {"fi", "done", "esac"}
PREFIXES = OPENERS | CLOSERS | {"then", "elif", "else", "do", "{", "}", "!", "time", "builtin", "command"}
ASSIGN_RE = re.compile(r"[A-Za-z_]\w*=")
REDIRECT_RE = re.compile(r"(?:\d*|&)(?:>>?|<)&?")
MSYS_DRIVE_RE = re.compile(r"/[A-Za-z](?=/|$)")
DYNAMIC = set("$`*?[{")  # variables, substitutions and globs: not followed
# A directory state is (base, ups, names): `ups` levels above `base`, then down `names`. Base None is the
# directory the command started in; a state of None is a directory that cannot be told.
START = (None, 0, ())


def _shell_text(command: str) -> str:
    """The command with line continuations blanked and backslashes as slashes (Windows paths), offsets kept."""
    for cont in ("\\\r\n", "\\\n", "`\r\n", "`\n"):
        command = command.replace(cont, " " * len(cont))
    return command.replace("\\", "/")


def _shell_words(text: str) -> tuple[list[tuple[bool, str]], int | None]:
    """Quote-aware words and operators of a command, as (is_operator, text), and the offset of a quote still
    open at the end (None when every quote closes)."""
    out, word, i = [], None, 0
    while i < len(text):
        c = text[i]
        if c in "'\"":
            j = text.find(c, i + 1)
            if j < 0:
                return out + ([(False, word)] if word is not None else []), i
            word, i = (word or "") + text[i + 1:j], j + 1
            continue
        op = next((o for o in SHELL_OPS if text.startswith(o, i)), None) if c in "&|;\n()" else None
        if op == "&" and (text.startswith("&>", i) or (word or "").endswith((">", "<"))):
            op = None  # a redirection: &>file, 2>&1
        if c == "#" and word is None:  # a comment, up to the end of the line
            j = text.find("\n", i)
            i = len(text) if j < 0 else j
        elif op or c in " \t\r":
            if word is not None:
                out.append((False, word))
            if op:
                out.append((True, op))
            word, i = None, i + (len(op) if op else 1)
        else:
            word, i = (word or "") + c, i + 1
    return out + ([(False, word)] if word is not None else []), None


def _cd_target(args: list[str]) -> str | None:
    """The directory a cd / pushd / Set-Location call names; None unless it names exactly one."""
    names, i = [], 0
    while i < len(args):
        a = args[i]
        if a.lower() in ("-path", "-literalpath") and i + 1 < len(args):
            names.append(args[i + 1])
            i += 2
            continue
        if REDIRECT_RE.match(a):  # >/dev/null, 2>&1, or a bare > before its target
            i += 2 if REDIRECT_RE.fullmatch(a) else 1
            continue
        if a.startswith("-") and a != "-":
            if a.lower() not in ("-l", "-p", "-e", "-@", "-passthru", "--"):
                return None
        else:
            names.append(a)
        i += 1
    return names[0] if len(names) == 1 else None


def _join(state: tuple | None, path: str) -> tuple | None:
    """`state` moved to `path`, lexically; None when `path` cannot be followed without running the shell."""
    if state is None or not path or DYNAMIC & set(path) or (path[0] == "~" and path[:2] not in ("~", "~/")):
        return None
    if path[0] == "~":
        state, path = (Path.home(), 0, ()), path[2:]
    elif os.name == "nt" and MSYS_DRIVE_RE.match(path):
        path = f"{path[1]}:/{path[3:]}"  # Git Bash's /c/x is C:/x
    p = Path(path)
    if p.is_absolute():
        state, path = (Path(p.anchor), 0, ()), path[len(p.anchor):]
    elif p.drive or path.startswith("/"):
        return None  # drive-relative (C:x), or a Git Bash path such as /tmp with no fixed Windows path
    base, ups, names = state
    names = list(names)
    for name in path.split("/"):
        if name == "..":
            if names:
                names.pop()
            else:
                ups += 1
        elif name not in ("", "."):
            names.append(name)
    return base, ups, tuple(names)


def _walk(words: list[tuple[bool, str]], state: tuple | None) -> tuple | None:
    """The directory state after the `cd` steps among `words` (from `_shell_words`). A step that may not run
    (in a branch or loop, after ||, in a pipe or a background job) leaves it unknown. Steps in a subshell
    end with it, as in bash."""
    old, pushed, scopes, depth = None, [], [], 0
    cmd, before = [], None
    for is_op, text in words + [(True, "")]:
        if not is_op:
            cmd.append(text)
            continue
        while cmd and (cmd[0].lower() in PREFIXES or ASSIGN_RE.match(cmd[0])):
            w = cmd.pop(0).lower()
            depth = depth + 1 if w in OPENERS else max(depth - (w in CLOSERS), 0)
        name = cmd[0].lower() if cmd else ""
        if name in CD_CMDS or name in POP_CMDS:
            if depth or before in ("||", "|", "|&") or text in ("|", "|&", "&"):
                new = None
            elif name in POP_CMDS:
                new = pushed.pop() if pushed else None
            else:
                target = _cd_target(cmd[1:])
                if target == "-":
                    new = old
                else:
                    if name in ("pushd", "push-location"):
                        pushed.append(state)
                    new = _join(state, target) if target else None
            old, state = state, new
        if text == "(":
            scopes.append((state, old, list(pushed), depth))
        elif text == ")" and scopes:
            state, old, pushed, depth = scopes.pop()
        cmd, before = [], text
    return state


def _cd_state(text: str, state: tuple | None = START) -> tuple | None:
    """The directory state after `text`: a command, or the part of one before an offset. When that offset is in
    a quoted string (`bash -c "cd A && kaggle ..."`), the string's text runs as a command too."""
    words, quote = _shell_words(text)
    state = _walk(words, state)
    return state if quote is None else _cd_state(text[quote + 1:], state)


def _start_after(cwd: Path, final: tuple | None) -> tuple[Path | None, int]:
    """(dir, ups): the directory a command started in is `ups` levels below `dir`, found by undoing its `cd`
    steps (`final`) from the `cwd` it ended in; (None, 0) when they do not lead to that `cwd`."""
    if final is not None and final[0] is None:
        _, ups, names = final
        keep = len(cwd.parts) - len(names)
        if keep >= 0 and [os.path.normcase(n) for n in cwd.parts[keep:]] == [os.path.normcase(n) for n in names]:
            return Path(*cwd.parts[:keep]), ups
    if os.environ.get("CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR", "").strip().lower() in ("1", "true", "yes", "on"):
        return cwd, 0  # Claude Code moved back to the directory every command starts in
    return None, 0


def _place(state: tuple, start: Path | None, known: int = 0) -> Path | None:
    """The path of a directory state, given the start directory (or its ancestor `known` levels up)."""
    base, ups, names = state
    if base is None:
        if start is None or ups < known:
            return None
        base, ups = start, ups - known
    for _ in range(ups):
        base = base.parent
    return base.joinpath(*names)


def path_at(payload: dict, path: str | None, at: int, ran: bool = False) -> Path | None:
    """`path` (default: the current directory) as the payload's command uses it at offset `at`: a relative path
    is read from the directory the `cd` steps before `at` lead to. ``ran`` marks a PostToolUse payload, whose
    `cwd` is where the whole command left the shell: the start is found by undoing the command's `cd` steps.
    None when the directory cannot be told, e.g. after a `cd` with a variable, in a loop or in a pipe."""
    text = _shell_text(command_of(payload))
    state = _join(_cd_state(text[:at]), path or ".")
    if state is None:
        return None
    cwd = Path(payload.get("cwd") or ".")
    return _place(state, *_start_after(cwd, _cd_state(text))) if ran else _place(state, cwd)


def push_target(payload: dict, push: dict, ran: bool = False) -> tuple[Path, dict] | None:
    """(kernel folder, kernel-metadata.json) of a parsed push, or None when the folder cannot be told or its
    metadata cannot be read. ``ran``: a PostToolUse payload (see `path_at`)."""
    folder = path_at(payload, push.get("path"), push.get("at", 0), ran)
    if folder is None:
        return None
    try:
        meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return folder, meta


def push_resource(push: dict, meta: dict) -> str | None:
    """'GPU' or 'TPU' when the push spends that quota, None for a CPU kernel."""
    acc = (push.get("accelerator") or "").strip()
    if acc:
        res = G.resource_of(acc)
    else:
        res = G.resource_of(G.kernel_meta_accelerator(meta))
    return None if res == "CPU" else res


def push_uses_gpu(push: dict, meta: dict) -> bool:
    return push_resource(push, meta) == "GPU"


# A heredoc can be only text: what cat or tee save or print (an issue body, notes) and what git or gh read (a commit
# message or PR body, also as "$(cat <<'EOF' ... EOF)" in their arguments). The `kaggle ...` lines it quotes are not
# calls. Its body is still read as commands when it can run: piped on, saved to a script the command then runs, or
# unquoted with $(...) or `...` in it.
HEREDOC_RE = re.compile(r"(?<!<)<<-?[ \t]*(['\"]?)([A-Za-z_]\w*)\1")
TEXT_READERS = {"cat", "tee", "git", "gh"}
MESSAGE_CMDS = {"git", "gh"}
RUNS_SCRIPT = (r"(?:(?:^|[\s;&|(])(?:(?:ba|z|da|k)?sh|source|\.|python[\d.]*|py|pwsh|powershell(?:\.exe)?)\s+"
               r"(?:-\S+\s+)*|\./)['\"]?(?:[^\s'\"]*/)?")  # `bash x.sh`, `python x.py`, `. x.sh`, `./x.sh`


def _text_heredocs_blanked(command: str) -> str:
    """`command` with the bodies of its text-only heredocs (see HEREDOC_RE) blanked, offsets kept."""
    # Lines are found in the command itself: _shell_text would join a body line ending in ` (Markdown) to the next.
    text, out, pos = _shell_text(command), list(command), 0
    for m in HEREDOC_RE.finditer(command):
        eol = command.find("\n", m.end())
        if m.start() < pos or eol < 0:
            continue  # in a body blanked already, or no body
        end = re.compile(rf"^[ \t]*{re.escape(m.group(2))}[ \t]*\r?$", re.MULTILINE).search(command, eol + 1)
        if end is None or (not m.group(1) and re.search(r"\$\(|`", command[eol + 1:end.start()])):
            continue
        if _heredoc_is_text(text[command.rfind("\n", 0, m.start()) + 1:m.start()], text[m.end():eol],
                            text[end.start():]):
            out[eol + 1:end.start()] = [c if c == "\n" else " " for c in command[eol + 1:end.start()]]
            pos = end.start()
    return "".join(out)


def _heredoc_is_text(head: str, tail: str, rest: str) -> bool:
    """Whether a heredoc's body is only text, from the command text before its marker on that line (`head`), the
    rest of that line (`tail`) and the command after the body (`rest`)."""
    words, quote = _shell_words(head)
    outer = None
    if quote is not None:  # in a quoted argument: `git commit -m "$(cat <<'EOF'`
        outer, (words, quote) = words, _shell_words(head[quote + 1:])
        if quote is not None:
            return False
    after, _ = _shell_words(tail)
    if next((t for is_op, t in after if is_op), "") in ("|", "|&"):
        return False  # piped on to another command
    cut = max((i for i, (is_op, _) in enumerate(words) if is_op), default=-1)
    cmd = _command_words(words[cut + 1:])
    if not cmd or cmd[0].lower() not in TEXT_READERS:
        return False
    if cut > 0 and words[cut][1] == "(" and words[cut - 1][1].endswith("$"):  # $(cat <<'EOF' ...)
        owner = _command_words(outer if outer is not None else words[:cut - 1])
        return bool(owner) and owner[0].lower() in MESSAGE_CMDS
    if outer is not None:
        return False
    for is_op, t in after:
        if is_op:
            break
        cmd.append(t)
    return not any(re.search(RUNS_SCRIPT + re.escape(f.rsplit("/", 1)[-1]) + r"(?![\w.-])", rest, re.IGNORECASE)
                   for f in _written_files(cmd))


def _command_words(words: list[tuple[bool, str]]) -> list[str]:
    """The words of the last command among `words` (from `_shell_words`), without prefixes such as `then` or X=1."""
    cmd = []
    for is_op, text in words:
        cmd = [] if is_op else cmd + [text]
    while cmd and (cmd[0].lower() in PREFIXES or ASSIGN_RE.match(cmd[0])):
        cmd.pop(0)
    return cmd


def _written_files(cmd: list[str]) -> list[str]:
    """The files a cat or tee command writes: its > and >> targets, and tee's arguments."""
    out, i = [], 1
    while i < len(cmd):
        m = re.fullmatch(r"\d?>>?(.*)", cmd[i])
        if m:
            target = m.group(1) or (cmd[i + 1] if i + 1 < len(cmd) else "")
            i += 1 if m.group(1) else 2
        else:
            target = cmd[i] if cmd[0].lower() == "tee" and not cmd[i].startswith("-") else ""
            i += 1
        if target and not target.startswith(("&", "/dev/")):
            out.append(target)
    return out


SUBMIT_FLAGS = {"-f": "file", "--file": "file", "-m": "message", "--message": "message", "-k": "kernel",
                "--kernel": "kernel", "-v": "version", "--version": "version", "-c": "competition",
                "--competition": "competition"}


def parse_submits(command: str) -> list[dict]:
    """Every `kaggle competitions submit` in a (possibly compound) command, in order: competition / file / message /
    kernel / version / quiet. `--help` calls and text in heredocs (an issue body quoting a submit) are left out.
    ``at`` is where each one starts, so its -f file is read after the `cd` steps before it (`path_at`)."""
    text, out, pos = _text_heredocs_blanked(command), [], 0
    while True:
        m = SUBMIT_RE.search(text, pos)
        if not m:
            return out
        pos = m.start() + 1
        sub = _parse_submit_args(m.group(1))
        if sub is not None:
            sub["at"] = m.start()
            out.append(sub)


def parse_submit(command: str) -> dict | None:
    """The first real `kaggle competitions submit` in a command."""
    subs = parse_submits(command)
    return subs[0] if subs else None


def _parse_submit_args(rest: str) -> dict | None:
    toks = _tokens(rest)
    if "-h" in toks or "--help" in toks:
        return None
    out = {"competition": None, "file": None, "message": None, "kernel": None, "version": None,
           "quiet": "-q" in toks or "--quiet" in toks}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in SUBMIT_FLAGS and i + 1 < len(toks):
            out[SUBMIT_FLAGS[t]] = toks[i + 1]
            i += 2
            continue
        if "=" in t and t.split("=", 1)[0] in SUBMIT_FLAGS:
            k, v = t.split("=", 1)
            out[SUBMIT_FLAGS[k]] = v
        elif not t.startswith("-") and out["competition"] is None:
            out["competition"] = t
        i += 1
    return out


def submit_label(sub: dict) -> str:
    """A parsed submit as messages name it: its notebook and version, or its file."""
    if sub.get("kernel"):
        return f"{sub['kernel']} v{sub['version']}" if sub.get("version") else sub["kernel"]
    return sub.get("file") or "no -f file"


def find_sample(root: Path) -> Path | None:
    for pat in ("data/sample_submission.csv", "data/*/sample_submission.csv", "data/sample_submission*.csv",
                "data/*sample*.csv", "sample_submission.csv"):
        hits = sorted(root.glob(pat))
        if hits:
            return hits[0]
    return None


def _read_csv(path: Path, max_rows: int = 5_000_000):
    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        header = next(r, [])
        ids, blanks, n = [], 0, 0
        for row in r:
            n += 1
            if row:
                ids.append(row[0])
            if any(c.strip() == "" or c.strip().lower() == "nan" for c in row) or len(row) != len(header):
                blanks += 1
            if n >= max_rows:
                break
    return header, ids, blanks, n


def validate_csv(sub: Path, sample: Path) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    try:
        h, ids, blanks, n = _read_csv(sub)
        rh, rids, _, rn = _read_csv(sample)
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        return [f"could not read CSV: {e}"], []
    if [c.strip() for c in h] != [c.strip() for c in rh]:
        if any(c == "" or c.startswith("Unnamed") for c in h):
            errors.append("header has an empty/'Unnamed' first column — the DataFrame index was written (use index=False)")
        errors.append(f"header {h} != sample header {rh}")
    if n != rn:
        errors.append(f"{n} rows but sample_submission has {rn}")
    if blanks:
        errors.append(f"{blanks} rows with empty/NaN values or wrong column count")
    if len(set(ids)) != len(ids):
        errors.append(f"{len(ids) - len(set(ids))} duplicated ids")
    if set(ids) != set(rids):
        errors.append(f"id set differs from sample ({len(set(rids) - set(ids))} missing, {len(set(ids) - set(rids))} unexpected)")
    elif ids != rids:
        warnings.append("ids are in a different order than sample_submission")
    return errors, warnings


def file_sha256(path: Path, max_bytes: int = 2 * 1024 ** 3) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def previous_submission(root: Path, path: Path, records: list[dict] | None = None) -> dict | None:
    """The latest logged submission of a byte-identical file that Kaggle did not refuse. The file is hashed
    only when a logged submission has the same size (or none recorded), so a new file costs one stat."""
    recs = [r for r in (submission_records(root) if records is None else records)
            if r.get("sha256") and r.get("status") != "failed"]
    try:
        size = path.stat().st_size
    except OSError:
        return None
    recs = [r for r in recs if r.get("size") in (None, size)]
    if not recs:
        return None
    sha = file_sha256(path)
    if not sha:
        return None
    for r in reversed(recs):
        if r["sha256"] == sha:
            return r
    return None


def ledger_records(root: Path) -> list[dict]:
    p = root / S.STATE_DIR / "ledger.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


EXP_ID_RE = re.compile(r"\b(\d{4}_[a-z0-9-]+)\b")
