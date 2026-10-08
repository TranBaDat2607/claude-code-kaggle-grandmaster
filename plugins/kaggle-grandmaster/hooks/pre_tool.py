"""PreToolUse guard for Bash/PowerShell.

1. Blocks commands that would print Kaggle credentials (kaggle.json / KAGGLE_KEY).
2. Before `kaggle competitions submit -f FILE`: validates FILE against sample_submission
   (deny with reasons when invalid) and asks for confirmation when today's local submission
   count has reached the competition's daily limit.
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
    r"(?:\b(?:cat|type|more|less|head|tail|bat|get-content|gc|nl|strings|xxd|od|print|echo|code|notepad)\b[^|;&\n]*kaggle\.json)"
    r"|(?:\b(?:echo|printenv|print|write-output|write-host)\b[^|;&\n]*\$?(?:env:)?KAGGLE_(?:KEY|API_TOKEN))"
    r"|(?:\bprintenv\s+KAGGLE_(?:KEY|API_TOKEN))"
    r"|(?:\benv\b\s*\|\s*(?:grep|findstr|select-string)\b[^|;&\n]*KAGGLE)",
    re.IGNORECASE,
)


def main() -> None:
    payload = C.read_input()
    cmd = C.command_of(payload)
    if not cmd:
        return

    if SECRET_READ.search(cmd):
        C.emit("PreToolUse", permissionDecision="deny",
               permissionDecisionReason="kaggle-grandmaster: this command would print Kaggle API credentials. "
                                        "Never display or copy the key; check auth with `kaggle competitions list` instead.")
        return

    sub = C.parse_submit(cmd)
    if sub is None:
        return
    root = C.workspace(payload)
    cwd = Path(payload.get("cwd") or ".")
    problems, notes = [], []

    if sub.get("file") and not sub.get("kernel"):
        f = Path(sub["file"])
        if not f.is_absolute():
            f = cwd / f
        if not f.is_file():
            problems.append(f"submission file not found: {f}")
        elif f.suffix.lower() == ".csv" and root is not None:
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
        if used >= limit:
            C.emit("PreToolUse", permissionDecision="ask",
                   permissionDecisionReason=f"kaggle-grandmaster: {used}/{limit} submissions already logged today (UTC). "
                                            "Submitting may fail or spend tomorrow's planning budget — confirm?")
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
