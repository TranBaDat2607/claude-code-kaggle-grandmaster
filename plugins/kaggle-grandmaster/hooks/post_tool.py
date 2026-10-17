"""PostToolUse: after `kaggle competitions submit`, log the submission locally (with the file's SHA-256 and
what Kaggle says is left today) and remind Claude to fetch the public score and attach it to the
experiment ledger. After each GPU or TPU `kaggle kernels push` in the command (`--help` calls aside),
record the run in the GPU-hour log; a re-push of a kernel replaces its earlier reservation. The Kaggle CLI
exits 0 after several failed writes; say so when one shows up in the output."""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

try:
    import _common as C
except Exception:  # pragma: no cover - fail open
    sys.exit(0)


def _response_text(payload: dict) -> str:
    r = payload.get("tool_response")
    if isinstance(r, dict):
        return " ".join(str(r.get(k, "")) for k in ("stdout", "stderr", "output", "error"))
    return str(r or "")


# A successful `submit -k` (code competition) prints only "N submissions remaining today."
SUBMIT_OK_RE = re.compile(r"successfully submitted|\d+\s+submissions?\s+remaining\s+today", re.IGNORECASE)
REMAINING_RE = re.compile(r"(\d+)\s+submissions?\s+remaining\s+today", re.IGNORECASE)
# Printed by kaggle 2.2.4 with exit status 0: a refused upload, a 404 competition, failed writes.
SUBMIT_FAILED = ("error", "exception", "forbidden", "401", "403", "404", "400 client", "could not submit",
                 "could not find competition")
SILENT_FAILURE_RE = re.compile(r"(?:Dataset(?: version)? creation|Kernel push|Model(?: instance)?(?: version)? "
                               r"creation) error\b[^\n]*", re.IGNORECASE)


def _kernel_runtime_h(root, kernel: str | None) -> float | None:
    """Measured wall time (h) of the latest finished run of ``kernel`` in the GPU-hour log."""
    if not kernel:
        return None
    for r in reversed(C.G.runs(root)):
        if r.get("kernel") != kernel:
            continue
        t0, t1 = C.G._parse_time(r.get("pushed_at")), C.G._parse_time(r.get("finished_at"))
        if t0 and t1 and t1 > t0:
            return round((t1 - t0).total_seconds() / 3600, 1)
    return None


def _failure_hint(low: str) -> str:
    if any(w in low for w in ("400 client", "bad request", "failed_precondition")):
        return ("kaggle-grandmaster: Kaggle refused the submission (400). The CLI hides the reason, which is in the "
                "response body: re-run it through the Python API and print `e.response.text` (snippet in skill "
                "`kaggle-cli`). In code competitions the usual cause is FAILED_PRECONDITION 'Your Notebook uses "
                "non-versioned datasets': attach a versioned dataset (or pin a dataset version) to the notebook, "
                "push a new kernel version and submit that version.")
    return ("kaggle-grandmaster: the submission appears to have failed. Read the CLI output; 403 usually means "
            "the competition rules were not accepted.")


def _poll_hint(root, sub: dict, st, slug: str) -> str:
    poll = f"`kaggle competitions submissions {slug} --format json`"
    if not (sub.get("kernel") or (st and st.code_competition)):
        return f"Poll {poll} in ~1-2 minutes"
    limit = st.runtime_limit_hours if st else None
    cap = f", up to the {limit:g}h runtime limit" if limit else ""
    took = _kernel_runtime_h(root, sub.get("kernel"))
    est = (f"its last run took {took:g}h, so expect about that times the hidden/visible test size ratio{cap}"
           if took is not None else f"often 1-9h{cap}")
    return (f"This is a code submission: Kaggle reruns the notebook on the hidden test before scoring, so grading "
            f"takes as long as that run ({est}). Poll {poll} in the background every ~30 minutes, not in a tight "
            f"loop")


PUSHED_RE = re.compile(r"version\s+(\d+)\s+successfully pushed(?:[^\n]*?/code/([\w.-]+/[\w.-]+))?", re.IGNORECASE)


def _pushed_version(text: str, kernel: str, n_pushes: int) -> int | None:
    """The version Kaggle reported for ``kernel``: matched on the progress URL when the command pushed several
    kernels, else the only success line."""
    hits = [(int(m.group(1)), (m.group(2) or "").lower()) for m in PUSHED_RE.finditer(text)]
    slug = kernel.split("/")[-1].lower()
    for v, ref in hits:
        if ref and ref.split("/")[-1] == slug:
            return v
    if n_pushes == 1 and len(hits) == 1:
        return hits[0][0]
    return None


def log_gpu_push(payload: dict, push: dict, n_pushes: int = 1) -> str | None:
    """A raw GPU/TPU `kaggle kernels push` spends quota too: record it so budgets account for it."""
    root = C.workspace(payload)
    target = C.push_target(payload, push)
    res = C.push_resource(push, target[1]) if target is not None else None
    if root is None or res is None:
        return None
    kernel = target[1].get("id") or "?"
    version = _pushed_version(_response_text(payload), kernel, n_pushes)
    if version is None:
        return None
    try:
        hours = round(float(push["timeout"]) / 3600, 3) if push.get("timeout") else C.G.session_limit(res)
    except ValueError:
        hours = C.G.session_limit(res)
    acc = push.get("accelerator") or C.G.kernel_meta_accelerator(target[1])
    C.G.record_push(root, {"kernel": kernel, "version": version, "name": target[0].name, "accelerator": acc,
                           "resource": res, "hours": hours,
                           "pushed_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                           "status": "pushed", "via": "kaggle-cli"})
    queue = (" TPU jobs usually wait 10-24h in the queue; `kaggle kernels logs` is empty until the session ends."
             if res == "TPU" else "")
    return (f"kaggle-grandmaster: {res} kernel {kernel} v{version} logged (cap {hours}h of weekly quota"
            + ("" if push.get("timeout") else ": no -t, so the full session limit is reserved") + f").{queue} Poll "
            f"with `python -m kgkit gpu wait {kernel}` in the background (not a tight loop); it releases the "
            f"reservation when the run ends. For kgkit-built training kernels then run `python -m kgkit gpu "
            f"collect {kernel}`.")


def silent_failure_note(payload: dict) -> str | None:
    """An upload or push that printed one of the CLI's exit-0 errors: say nothing was created."""
    cmd = C.command_of(payload)
    if C.parse_upload(cmd) is None and C.parse_push(cmd) is None:
        return None
    m = SILENT_FAILURE_RE.search(_response_text(payload))
    if not m:
        return None
    hint = ""
    if "session count" in m.group(0).lower():
        hint = (" Kaggle allows only so many batch sessions at once (2 GPU, 1 TPU): wait for a running kernel to "
                "finish (`python -m kgkit gpu status`) and push again.")
    return (f"kaggle-grandmaster: the Kaggle CLI printed `{C.SF.clean(m.group(0))[:300]}` but exited 0, so nothing "
            "was created or started and commands chained after it with && still ran. Do not wait for, attach or "
            f"submit this version.{hint}")


def log_submission(payload: dict, sub: dict) -> str | None:
    root = C.workspace(payload)
    if root is None:
        return None
    # An upload/push error earlier in the same command must not be read as this submit's failure.
    text = SILENT_FAILURE_RE.sub("", _response_text(payload))
    low = text.lower()
    failed = any(w in low for w in SUBMIT_FAILED)
    ok = SUBMIT_OK_RE.search(text) is not None and "could not submit" not in low
    status = "submitted" if ok else ("failed" if failed else "unknown")
    left = REMAINING_RE.search(text)
    sha = size = None
    if sub.get("file") and not sub.get("kernel"):
        f = Path(sub["file"])
        f = f if f.is_absolute() else Path(payload.get("cwd") or ".") / f
        sha = C.file_sha256(f)
        try:
            size = f.stat().st_size
        except OSError:
            pass
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
        "sha256": sha,
        "size": size,
        "remaining_today": int(left.group(1)) if left else None,
    }
    log = C.submissions_log(root)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    if status == "failed":
        return _failure_hint(low)
    st = C.S.load(root)
    slug = sub.get("competition") or (st.slug if st else "<slug>")
    used = C.submissions_today(root)
    limit = st.daily_submissions if st else 5
    count = (f"Kaggle reports {left.group(1)} left today for the team" if left
             else f"{used}/{limit} today in the local log, UTC")
    ctx = (f"kaggle-grandmaster: submission logged ({count}). {_poll_hint(root, sub, st, slug)}; "
           f"when the public score appears, run `python -m kgkit ledger lb {exp_id or '<exp_id>'} <score>` and "
           f"compare with CV.")
    if not exp_id:
        ctx += " (No ledger experiment id found in the message/file name — identify which experiment this was.)"
    return ctx


def main() -> None:
    payload = C.read_input()
    cmd = C.command_of(payload)
    # One command can chain an upload, a push and a submit: report on each, in one message.
    notes = [silent_failure_note(payload)]
    pushes = C.parse_pushes(cmd)
    for push in pushes:
        notes.append(log_gpu_push(payload, push, len(pushes)))
    sub = C.parse_submit(cmd)
    if sub is not None:
        notes.append(log_submission(payload, sub))
    notes = [n for n in notes if n]
    if notes:
        C.emit("PostToolUse", additionalContext="\n\n".join(notes))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
