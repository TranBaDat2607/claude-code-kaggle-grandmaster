"""PreToolUse guard for Bash/PowerShell.

1. Blocks commands that would print Kaggle credentials: kaggle.json, ~/.kaggle/access_token or
   credentials.json, KAGGLE_KEY / KAGGLE_API_TOKEN, `kaggle auth print-access-token`, and Kaggle CLI calls
   with VERBOSE / VERBOSE_OUTPUT on (the CLI then prints request headers, bearer token included).
2. Blocks a dataset or model upload whose folder holds credential-like files or links to files outside it.
3. Before `kaggle competitions submit -f FILE`: validates FILE against sample_submission (deny with reasons
   when invalid), and asks for confirmation when the same file was already submitted, when today's local
   count has reached the daily limit, or when Kaggle reported no submissions left today for the team.
4. Before a GPU `kaggle kernels push`: asks for confirmation when the cached weekly GPU quota
   (minus kernels still running) is exhausted or smaller than the push's -t cap.
Valid submissions produce no output, so the normal permission flow is unchanged.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import _common as C
except Exception:  # pragma: no cover - fail open
    sys.exit(0)

SECRET_READ = re.compile(
    r"(?:\b(?:cat|type|more|less|head|tail|bat|get-content|gc|nl|strings|xxd|od|print|echo|code|notepad)\b[^|;&\n]*"
    r"(?:kaggle\.json|\.kaggle[/\\](?:access_token(?:\.txt)?|credentials\.json)))"
    r"|(?:\b(?:echo|printenv|print|write-output|write-host)\b[^|;&\n]*\$?(?:env:)?KAGGLE_(?:KEY|API_TOKEN))"
    r"|(?:\bprintenv\s+KAGGLE_(?:KEY|API_TOKEN))"
    r"|(?:\benv\b\s*\|\s*(?:grep|findstr|select-string)\b[^|;&\n]*KAGGLE)",
    re.IGNORECASE,
)
PRINT_TOKEN = re.compile(r"\bkaggle(?:\.exe)?\s+auth\s+print-access-token\b", re.IGNORECASE)
# kaggle 2.2.4 / kagglesdk: VERBOSE or VERBOSE_OUTPUT = 1|true|yes prints every request's headers unmasked.
VERBOSE_ON = re.compile(r"\bVERBOSE(?:_OUTPUT)?\s*=\s*[\"']?(?:1|true|yes)\b", re.IGNORECASE)
KAGGLE_CALL = re.compile(r"\b(?:kaggle(?:\.exe)?\s+[a-z]|kgkit\b)", re.IGNORECASE)


def secret_reason(cmd: str) -> str | None:
    if SECRET_READ.search(cmd):
        return ("this command would print Kaggle API credentials. Never display or copy the key; check auth with "
                "`kaggle competitions list` instead.")
    if PRINT_TOKEN.search(cmd):
        return "`kaggle auth print-access-token` prints a live access token. Never display it."
    if VERBOSE_ON.search(cmd) and KAGGLE_CALL.search(cmd):
        return ("with VERBOSE / VERBOSE_OUTPUT on, the Kaggle CLI prints every request's headers, including the "
                "bearer token. Run it without them; read the error body instead (snippet in skill `kaggle-cli`).")
    return None


def upload_guard(payload: dict, up: dict) -> bool:
    """Deny a dataset/model upload that would send credential files; True when it emitted a decision."""
    folder = Path(up.get("path") or ".")
    if not folder.is_absolute():
        folder = Path(payload.get("cwd") or ".") / folder
    if not folder.is_dir():
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


def gpu_push_guard(payload: dict, push: dict) -> None:
    """Ask before a GPU kernel push when the cached weekly quota cannot cover it."""
    target = C.push_target(payload, push)
    root = C.workspace(payload)
    if target is None or root is None or not C.push_uses_gpu(push, target[1]):
        return
    q = C.G.cached_quota(root)
    if q is None:
        sys.stderr.write("kaggle-grandmaster: GPU quota unknown - run `python -m kgkit gpu quota` before pushing.\n")
        return
    avail = C.G.available_hours(root, q)
    try:
        cap = float(push["timeout"]) / 3600 if push.get("timeout") else None
    except ValueError:
        cap = None
    when = f" (quota checked {q.get('fetched_at')}, resets {q.get('refresh_at')})"
    if avail <= 0.05:
        C.emit("PreToolUse", permissionDecision="ask",
               permissionDecisionReason=f"kaggle-grandmaster: the weekly Kaggle GPU quota looks exhausted: "
                                        f"{avail:.2f}h plannable after running kernels{when}. Push anyway?")
    elif cap is not None and cap > avail:
        C.emit("PreToolUse", permissionDecision="ask",
               permissionDecisionReason=f"kaggle-grandmaster: this GPU run is capped at {cap:.2f}h but only "
                                        f"{avail:.2f}h of GPU quota is plannable{when}. Push anyway?")
    elif cap is None:
        sys.stderr.write("kaggle-grandmaster: GPU push without -t: a hung session can burn up to 12h of quota. "
                         "Prefer `python -m kgkit gpu push <dir>` (quota check + cap + run log).\n")


def main() -> None:
    payload = C.read_input()
    cmd = C.command_of(payload)
    if not cmd:
        return

    reason = secret_reason(cmd)
    if reason:
        C.emit("PreToolUse", permissionDecision="deny", permissionDecisionReason="kaggle-grandmaster: " + reason)
        return

    up = C.parse_upload(cmd)
    if up is not None:
        upload_guard(payload, up)
        return

    push = C.parse_push(cmd)
    if push is not None:
        gpu_push_guard(payload, push)
        return

    sub = C.parse_submit(cmd)
    if sub is None:
        return
    root = C.workspace(payload)
    cwd = Path(payload.get("cwd") or ".")
    problems, notes, asks = [], [], []

    if sub.get("file") and not sub.get("kernel"):
        f = Path(sub["file"])
        if not f.is_absolute():
            f = cwd / f
        if not f.is_file():
            problems.append(f"submission file not found: {f}")
        elif root is not None and (prev := C.previous_submission(root, C.file_sha256(f))):
            asks.append(f"this exact file was already submitted at {prev.get('time_utc')} "
                        f"(message {prev.get('message')!r}); Kaggle would score it the same and it uses a daily slot")
        if f.is_file() and f.suffix.lower() == ".csv" and root is not None:
            sample = C.find_sample(root)
            if sample is not None:
                errs, warns = C.validate_csv(f, sample)
                problems += errs
                notes += warns
    if not sub.get("message"):
        notes.append("no -m message: include the ledger experiment id and CV, e.g. -m \"0012_lgbm-te | CV 0.8123\"")

    if problems:
        C.emit("PreToolUse", permissionDecision="deny",
               permissionDecisionReason="kaggle-grandmaster blocked an invalid submission (it would waste a daily slot):\n- "
                                        + "\n- ".join(problems + notes))
        return

    if root is not None:
        st = C.S.load(root)
        limit = st.daily_submissions if st else 5
        used = C.submissions_today(root)
        left = C.server_remaining_today(root)
        if used >= limit:
            asks.append(f"{used}/{limit} submissions already logged today (UTC); submitting may fail or spend "
                        "tomorrow's planning budget")
        elif left == 0:
            asks.append("after the last submit Kaggle reported 0 submissions left today for the team (teammates' and "
                        "website submissions count too)")
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
