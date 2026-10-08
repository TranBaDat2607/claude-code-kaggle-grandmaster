"""PostToolUse: after `kaggle competitions submit`, log the submission locally and remind Claude to
fetch the public score and attach it to the experiment ledger."""

from __future__ import annotations

import datetime as dt
import json
import sys

try:
    import _common as C
except Exception:  # pragma: no cover - fail open
    sys.exit(0)


def _response_text(payload: dict) -> str:
    r = payload.get("tool_response")
    if isinstance(r, dict):
        return " ".join(str(r.get(k, "")) for k in ("stdout", "stderr", "output", "error"))
    return str(r or "")


def main() -> None:
    payload = C.read_input()
    sub = C.parse_submit(C.command_of(payload))
    if sub is None:
        return
    root = C.workspace(payload)
    if root is None:
        return
    text = _response_text(payload)
    low = text.lower()
    failed = any(w in low for w in ("error", "exception", "forbidden", "401", "403", "404", "400 client"))
    ok = "successfully submitted" in low
    status = "submitted" if ok else ("failed" if failed else "unknown")
    m = C.EXP_ID_RE.search(sub.get("message") or "") or C.EXP_ID_RE.search(sub.get("file") or "")
    exp_id = m.group(1) if m else None
    rec = {
        "time_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "competition": sub.get("competition"),
        "file": sub.get("file"),
        "kernel": sub.get("kernel"),
        "message": sub.get("message"),
        "exp_id": exp_id,
        "status": status,
    }
    log = C.submissions_log(root)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    if status == "failed":
        C.emit("PostToolUse", additionalContext="kaggle-grandmaster: the submission appears to have failed — read the "
                                                "CLI output; 403 usually means the competition rules were not accepted.")
        return
    st = C.S.load(root)
    slug = sub.get("competition") or (st.slug if st else "<slug>")
    used = C.submissions_today(root)
    limit = st.daily_submissions if st else 5
    ctx = (f"kaggle-grandmaster: submission logged ({used}/{limit} today, UTC). Poll `kaggle competitions submissions "
           f"{slug} --format json` in ~1-2 minutes; when the public score appears, run "
           f"`python -m kgkit ledger lb {exp_id or '<exp_id>'} <score>` and compare with CV.")
    if not exp_id:
        ctx += " (No ledger experiment id found in the message/file name — identify which experiment this was.)"
    C.emit("PostToolUse", additionalContext=ctx)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
