"""PostToolUse: after `kaggle competitions submit`, log the submission locally (with the file's SHA-256 and
what Kaggle says is left today) and remind Claude to fetch the public score and attach it to the
experiment ledger. After each GPU or TPU `kaggle kernels push` in the command (`--help` calls aside),
record the run in the GPU-hour log; a re-push of a kernel replaces its earlier reservation. `cwd` here is
where the command left the shell: relative paths are read from where its `cd` steps were at that point, and
a push whose folder cannot be told is reported, not logged. A command that went on in the background has
left `cwd` where it started, and Kaggle's reply comes after this hook: its pushes are logged as pending, their
whole cap reserved, until `kgkit gpu status` / `gpu wait` see Kaggle run them. The Kaggle CLI exits 0 after
several failed writes; say so when one shows up in the output."""

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


# Claude Code's Bash and PowerShell results for a command that goes on in the background (asked for with
# run_in_background, or moved there on its timeout or by the user): a task id and none of the output, which goes
# to a file no hook reads. The text form is what the model sees.
BACKGROUND_RE = re.compile(r"running in background with ID|moved to the background|backgrounded by user",
                           re.IGNORECASE)


def _backgrounded(payload: dict) -> bool:
    if (payload.get("tool_input") or {}).get("run_in_background"):
        return True
    r = payload.get("tool_response")
    if isinstance(r, dict):
        return bool(r.get("backgroundTaskId"))
    return BACKGROUND_RE.search(str(r or "")) is not None


def _background_since(payload: dict) -> dt.datetime:
    """When a backgrounded command started: now if it was asked to run in the background; one moved there
    later ran in the foreground first, for its timeout at most (Claude Code's default: 2 minutes)."""
    now = dt.datetime.now(dt.timezone.utc)
    ti = payload.get("tool_input") or {}
    if ti.get("run_in_background"):
        return now
    r = payload.get("tool_response")
    ms = (r.get("timedOutAfterMs") if isinstance(r, dict) else None) or ti.get("timeout") or 120_000
    try:
        return now - dt.timedelta(milliseconds=float(ms))
    except (TypeError, ValueError, OverflowError):
        return now - dt.timedelta(minutes=2)


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


def _take_version(hits: list, kernel: str, n_pushes: int) -> int | None:
    """Take from ``hits`` [(version, kernel ref)] the version Kaggle reported for ``kernel``: matched on the
    progress URL, else the only success line when the command pushed once."""
    slug = kernel.split("/")[-1].lower()
    for i, (_, ref) in enumerate(hits):
        if ref and ref.split("/")[-1] == slug:
            return hits.pop(i)[0]
    if n_pushes == 1 and len(hits) == 1:
        return hits.pop()[0]
    return None


def log_gpu_pushes(payload: dict, pushes: list[dict]) -> list[str]:
    """Log each GPU/TPU push of the command that Kaggle accepted. A success line that no push accounts for
    (its folder depends on a `cd` the hook cannot trace) is reported, not guessed: a guess logs another kernel.
    In a command that went on in the background Kaggle has not answered yet: its pushes are logged as pending."""
    root = C.workspace(payload)
    if root is None or not pushes:
        return []
    since = _background_since(payload) if _backgrounded(payload) else None
    hits = [(int(m.group(1)), (m.group(2) or "").lower()) for m in PUSHED_RE.finditer(_response_text(payload))]
    notes, unplaced = [], False
    for push in pushes:
        target = C.push_target(payload, push, ran=since is None)  # in the background: cwd is where it started
        version = _take_version(hits, target[1].get("id") or "?", len(pushes)) if target else None
        if version is not None:
            notes.append(log_gpu_push(root, push, target, version))
        elif since is not None and target is not None:
            notes.append(log_pending_push(root, push, target, since))
        elif since is not None:
            unplaced = True
    logged = {(str(r.get("kernel") or "").lower(), r.get("version")) for r in C.G.runs(root)}
    missed = [f"{ref or 'a kernel'} v{v}" for v, ref in hits if (ref, v) not in logged]
    if missed:
        notes.append(f"kaggle-grandmaster: Kaggle reports {', '.join(missed)} pushed, but the hook could not tell "
                     "which kernel folder the push used (typically after a `cd` with a variable, in a loop or in a "
                     "pipe), so the run is not in the GPU-hour log and, if it runs on a GPU or TPU, quota checks "
                     "do not reserve its hours. Push each kernel with an absolute `-p` path, or with `python -m "
                     "kgkit gpu push <dir>`, so the run is logged.")
    if unplaced:
        notes.append("kaggle-grandmaster: this command runs in the background and pushes a kernel from a folder the "
                     "hook cannot tell (typically after a `cd` with a variable, in a loop or in a pipe), so the run "
                     "is not in the GPU-hour log and, if it runs on a GPU or TPU, quota checks do not reserve its "
                     "hours. Push each kernel with an absolute `-p` path, or with `python -m kgkit gpu push <dir>`, "
                     "so the run is logged.")
    return [n for n in notes if n]


def _push_record(push: dict, target: tuple, res: str, version: int | None, status: str, at: dt.datetime) -> dict:
    try:
        hours = round(float(push["timeout"]) / 3600, 3) if push.get("timeout") else C.G.session_limit(res)
    except ValueError:
        hours = C.G.session_limit(res)
    return {"kernel": target[1].get("id") or "?", "version": version, "name": target[0].name,
            "accelerator": push.get("accelerator") or C.G.kernel_meta_accelerator(target[1]), "resource": res,
            "hours": hours, "pushed_at": at.strftime("%Y-%m-%dT%H:%M:%SZ"), "status": status, "via": "kaggle-cli"}


def log_pending_push(root: Path, push: dict, target: tuple, since: dt.datetime) -> str | None:
    """A GPU/TPU push in a command that went on in the background: the run is logged as pending, its whole cap
    reserved, until Kaggle lists a version of the kernel run after ``since`` (when the command started)."""
    res = C.push_resource(push, target[1])
    if res is None:
        return None
    rec = dict(_push_record(push, target, res, None, C.G.PENDING, since), background=True)
    C.G.record_push(root, rec)
    kernel, hours = rec["kernel"], rec["hours"]
    return (f"kaggle-grandmaster: this command runs in the background, so Kaggle's reply to its {res} push of "
            f"{kernel} (with the version) is not in this result. The run is logged as pending, with its cap of "
            f"{hours}h of weekly quota reserved" + ("" if push.get("timeout") else " (no -t: the full session limit)")
            + f", until Kaggle lists a version of it run after the command started. `python -m kgkit gpu wait "
            f"{kernel}`, run in the background, waits for the push and then for the run (`gpu status` and `gpu "
            f"quota` check it too). If the command ends without 'successfully pushed', run `python -m kgkit gpu "
            f"release {kernel}`.")


def log_gpu_push(root: Path, push: dict, target: tuple, version: int) -> str | None:
    """A raw GPU/TPU `kaggle kernels push` spends quota too: record it so budgets account for it."""
    res = C.push_resource(push, target[1])
    if res is None:
        return None
    rec = _push_record(push, target, res, version, "pushed", dt.datetime.now(dt.timezone.utc))
    if C.G.record_push(root, rec) is not rec:
        return None  # this version is in the log already
    kernel, hours = rec["kernel"], rec["hours"]
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
    sha = size = f = None
    if sub.get("file") and not sub.get("kernel"):
        f = C.path_at(payload, sub["file"], sub.get("at", 0), ran=True)
    if f is not None:
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
    notes = [silent_failure_note(payload), *log_gpu_pushes(payload, C.parse_pushes(cmd))]
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
