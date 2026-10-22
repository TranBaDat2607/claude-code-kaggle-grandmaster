"""PreToolUse guard for Bash/PowerShell.

1. Blocks commands that would print Kaggle credentials: a file reader run on kaggle.json, ~/.kaggle/access_token
   or credentials.json; Python that loads one and prints anything but the username field; KAGGLE_KEY /
   KAGGLE_API_TOKEN; `kaggle auth print-access-token`; and Kaggle CLI calls with VERBOSE / VERBOSE_OUTPUT on
   for that call (the CLI then prints request headers, bearer token included). Text that only mentions the
   file (a heredoc saved with cat/tee, a commit message) passes; `python -m kgkit whoami` gives the username.
2. Blocks a dataset or model upload whose folder holds credential-like files or links to files outside it.
3. Before each `kaggle competitions submit -f FILE` in the command: validates FILE against sample_submission
   (deny with reasons when invalid), and asks for confirmation when the same file was already submitted, when
   the command's submits would take today's local count past the daily limit, or when Kaggle reported fewer
   submissions left today for the team than the command makes. A heredoc that is only text (an issue body
   that cat saves, a commit message) is not read for submits.
4. Before each GPU or TPU `kaggle kernels push` in the command: asks for confirmation when the cached
   weekly quota of that accelerator (minus kernels still running) is exhausted or smaller than the -t cap.
Relative paths (-p, -f) are read from where the command's own `cd` steps put the shell at that point.
Valid submissions produce no output, so the normal permission flow is unchanged.
"""

from __future__ import annotations

import re
import sys

try:
    import _common as C
except Exception:  # pragma: no cover - fail open
    sys.exit(0)

CRED_FILE = r"(?:kaggle\.json|\.kaggle[/\\](?:access_token(?:\.txt)?|credentials\.json))"
CRED_ENV = r"\$?(?:env:)?KAGGLE_(?:KEY|API_TOKEN)\b"
# A program name only counts where a command starts: prose that merely mentions the file (a commit
# message, an issue body) is not a read.
CMD_POS = r"(?:^|[;&|(`'\"\n{]|\$\()\s*(?:sudo\s+|command\s+)?"
READERS = (r"(?:cat|type|more|less|head|tail|bat|get-content|gc|nl|strings|xxd|od|hexdump|base64|jq|grep|egrep|"
           r"findstr|select-string|sed|awk|code|notepad|python\S*\s+-m\s+json\.tool)")
SECRET_READ = re.compile(
    rf"(?:{CMD_POS}{READERS}\b[^|;&\n]*{CRED_FILE})"
    rf"|(?:{CMD_POS}(?:echo|printf|printenv|print|write-output|write-host)\b[^|;&\n]*{CRED_ENV})"
    r"|(?:\benv\b\s*\|\s*(?:grep|findstr|select-string)\b[^|;&\n]*KAGGLE)",
    re.IGNORECASE | re.MULTILINE,
)
# Python that loads a credential file and prints: allowed only when it picks the username field alone.
CRED_REF = re.compile(rf"{CRED_FILE}|{CRED_ENV}", re.IGNORECASE)
PY_PRINT = re.compile(r"\b(?:print|pprint|sys\.stdout\.write|json\.dumps?)\s*\(", re.IGNORECASE)
USERNAME_ONLY = re.compile(r"""\[\s*['"]username['"]\s*\]|\.get\(\s*['"]username['"]""")
SECRET_FIELD = re.compile(r"""['"](?:key|token|access_token|api_token)['"]|\.read\(\)|KAGGLE_(?:KEY|API_TOKEN)""",
                          re.IGNORECASE)
HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1")
TEXT_SINK = re.compile(r"^\s*(?:cat|tee)\b")


def strip_text_heredocs(cmd: str) -> str:
    """Drop the bodies of heredocs that `cat`/`tee` write to a file: they are text being saved (an issue
    body, notes), not commands. A heredoc fed to an interpreter is kept."""
    lines, out, i = cmd.split("\n"), [], 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        m = HEREDOC.search(line)
        if not m or not TEXT_SINK.match(line):
            continue
        redirect = ">" in line[:m.start()] or ">" in line[m.end():] or re.match(r"\s*tee\s+[^-\s<]", line)
        if not redirect or "|" in line[m.end():]:
            continue
        while i < len(lines) and lines[i].strip() != m.group(2):
            i += 1
    return "\n".join(out)


def python_prints_secret(cmd: str) -> bool:
    if not CRED_REF.search(cmd) or not PY_PRINT.search(cmd):
        return False
    return not (USERNAME_ONLY.search(cmd) and not SECRET_FIELD.search(cmd))
PRINT_TOKEN = re.compile(r"\bkaggle(?:\.exe)?\s+auth\s+print-access-token\b", re.IGNORECASE)
# kaggle 2.2.4 / kagglesdk: VERBOSE or VERBOSE_OUTPUT = 1|true|yes prints every request's headers unmasked.
# kgkit drops both from the environment of its own Kaggle CLI calls (kgkit.safety.kaggle_env), so only a
# direct `kaggle ...` call or the kaggle Python package is at risk.
VERBOSE_ON = re.compile(r"\bVERBOSE(?:_OUTPUT)?[\"']?\s*=\s*[\"']?(?:1|true|yes)\b", re.IGNORECASE)
# Set for the rest of the command (bash export, cmd set/setx, PowerShell $env:) rather than one program.
VERBOSE_EXPORT = re.compile(r"(?:^|\s)(?:export|set|setx)\s+[^;&|\n]*\bVERBOSE|\$env:VERBOSE", re.IGNORECASE)
KAGGLE_CALL = re.compile(r"\bkaggle(?:\.exe)?\s+[a-z]|\b(?:import|from)\s+kaggle", re.IGNORECASE)
SEGMENT_SPLIT = re.compile(r"&&|\|\||[;|\n]")


def verbose_kaggle_call(cmd: str) -> bool:
    """A Kaggle CLI call that runs with VERBOSE on: set inline on that program, or exported earlier."""
    exported = False
    for seg in SEGMENT_SPLIT.split(cmd):
        on = VERBOSE_ON.search(seg) is not None
        if on and VERBOSE_EXPORT.search(seg):
            exported = True
        if KAGGLE_CALL.search(seg) and (on or exported):
            return True
    return False


def secret_reason(cmd: str) -> str | None:
    text = strip_text_heredocs(cmd)
    if SECRET_READ.search(text) or python_prints_secret(text):
        return ("this command would print Kaggle API credentials. Never display or copy the key; check auth with "
                "`kaggle competitions list`, and get the username alone with `python -m kgkit whoami`.")
    if PRINT_TOKEN.search(cmd):
        return "`kaggle auth print-access-token` prints a live access token. Never display it."
    if verbose_kaggle_call(cmd):
        return ("with VERBOSE / VERBOSE_OUTPUT on, the Kaggle CLI prints every request's headers, including the "
                "bearer token. Run it without them; read the error body instead (snippet in skill `kaggle-cli`).")
    return None


def upload_guard(payload: dict, up: dict) -> bool:
    """Deny a dataset/model upload that would send credential files; True when it emitted a decision."""
    folder = C.path_at(payload, up.get("path"), up.get("at", 0))
    if folder is None or not folder.is_dir():
        return False
    found = C.SF.scan_upload(folder, recursive=str(up.get("dir_mode")).lower() in ("zip", "tar"),
                             ignore=up.get("ignore"))
    if not found:
        return False
    shown = [C.SF.clean(f) for f in found[:20]] + ([f"... and {len(found) - 20} more"] if len(found) > 20 else [])
    C.emit("PreToolUse", permissionDecision="deny",
           permissionDecisionReason="kaggle-grandmaster: this upload would send credential-like files or links to "
                                    "files outside the folder (the Kaggle CLI uploads everything in it except "
                                    ".git, .cache and .huggingface):\n- " + "\n- ".join(shown) +
                                    "\nMove them out of the folder or add --ignore-patterns. If they really are "
                                    "meant to be published, the user can run the command themselves.")
    return True


def gpu_push_guard(payload: dict, push: dict) -> bool:
    """Ask before a GPU/TPU kernel push when the cached weekly quota cannot cover it; True when it emitted a
    decision."""
    target = C.push_target(payload, push)
    root = C.workspace(payload)
    res = C.push_resource(push, target[1]) if target is not None else None
    if root is None or res is None:
        return False
    q = C.G.cached_quota(root, res)
    if q is None:
        sys.stderr.write(f"kaggle-grandmaster: {res} quota unknown - run `python -m kgkit gpu quota` before pushing.\n")
        return False
    avail = C.G.available_hours(root, q, res)
    try:
        cap = float(push["timeout"]) / 3600 if push.get("timeout") else None
    except ValueError:
        cap = None
    when = f" (quota checked {q.get('fetched_at')}, resets {q.get('refresh_at')})"
    kernel = target[1].get("id") or target[0].name
    if avail <= 0.05:
        C.emit("PreToolUse", permissionDecision="ask",
               permissionDecisionReason=f"kaggle-grandmaster: the weekly Kaggle {res} quota looks exhausted: "
                                        f"{avail:.2f}h plannable after running kernels{when}. Push {kernel} anyway? "
                                        "(`python -m kgkit gpu quota` re-checks runs that already finished)")
        return True
    if cap is not None and cap > avail:
        C.emit("PreToolUse", permissionDecision="ask",
               permissionDecisionReason=f"kaggle-grandmaster: {kernel} is capped at {cap:.2f}h but only "
                                        f"{avail:.2f}h of {res} quota is plannable{when}. Push anyway?")
        return True
    if cap is None:
        sys.stderr.write(f"kaggle-grandmaster: {res} push without -t: a hung session can burn up to "
                         f"{C.G.session_limit(res):g}h of quota. Prefer `python -m kgkit gpu push <dir> [--hours H]` "
                         "(quota check + cap + run log).\n")
    return False


def submit_checks(payload: dict, root, records: list[dict], sub: dict) -> tuple[list, list, list]:
    """(problems, notes, asks) for one parsed submit: its -f file against sample_submission and the log."""
    problems, notes, asks = [], [], []
    if sub.get("file") and not sub.get("kernel"):
        f = C.path_at(payload, sub["file"], sub.get("at", 0))
        if f is None:
            notes.append("the -f file was not checked: the hook cannot tell which folder it is in (a `cd` with a "
                         "variable, in a loop or in a pipe)")
        elif not f.is_file():
            problems.append(f"submission file not found: {f}")
        elif root is not None and (prev := C.previous_submission(root, f, records)):
            asks.append(f"this exact file was already submitted at {prev.get('time_utc')} "
                        f"(message {prev.get('message')!r}); Kaggle would score it the same and it uses a daily slot")
        if f is not None and f.is_file() and f.suffix.lower() == ".csv" and root is not None:
            sample = C.find_sample(root)
            if sample is not None:
                errs, warns = C.validate_csv(f, sample)
                problems += errs
                notes += warns
    if not sub.get("message"):
        notes.append("no -m message: include the ledger experiment id and CV, e.g. -m \"0012_lgbm-te | CV 0.8123\"")
    return problems, notes, asks


def main() -> None:
    payload = C.read_input()
    cmd = C.command_of(payload)
    if not cmd:
        return

    reason = secret_reason(cmd)
    if reason:
        C.emit("PreToolUse", permissionDecision="deny", permissionDecisionReason="kaggle-grandmaster: " + reason)
        return

    # One command can chain an upload, a push and a submit: check each until one emits a decision.
    up = C.parse_upload(cmd)
    if up is not None and upload_guard(payload, up):
        return

    for push in C.parse_pushes(cmd):
        if gpu_push_guard(payload, push):
            return

    subs = C.parse_submits(cmd)
    if not subs:
        return
    root = C.workspace(payload)
    records = C.submission_records(root) if root is not None else []
    problems, notes, asks = [], [], []
    for n, sub in enumerate(subs, 1):
        tag = f"submit {n} ({C.submit_label(sub)}): " if len(subs) > 1 else ""
        for out, found in zip((problems, notes, asks), submit_checks(payload, root, records, sub)):
            out += [tag + x for x in found]

    if problems:
        C.emit("PreToolUse", permissionDecision="deny",
               permissionDecisionReason="kaggle-grandmaster blocked an invalid submission (it would waste a daily slot):\n- "
                                        + "\n- ".join(problems + notes))
        return

    if root is not None:
        st = C.S.load(root)
        limit = st.daily_submissions if st else 5
        used = C.submissions_today(root, records)
        left = C.server_remaining_today(root, records)
        more = f" and this command submits {len(subs)} times" if len(subs) > 1 else ""
        if used + len(subs) > limit:
            asks.append(f"{used}/{limit} submissions already logged today (UTC){more}; submitting may fail or spend "
                        "tomorrow's planning budget")
        elif left is not None and left < len(subs):
            asks.append(f"after the last submit Kaggle reported {left} submission{'' if left == 1 else 's'} left "
                        f"today for the team{more} (teammates' and website submissions count too)")
    if asks:
        C.emit("PreToolUse", permissionDecision="ask",
               permissionDecisionReason="kaggle-grandmaster: " + "; ".join(asks) + ". Submit anyway?")
        return
    # otherwise: no output → normal permission flow
    if notes:
        sys.stderr.write("kaggle-grandmaster: " + "; ".join(notes) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
