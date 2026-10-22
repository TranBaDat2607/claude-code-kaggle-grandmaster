"""PostToolUse: after each `kaggle competitions submit` in the command that Kaggle replied to, log the
submission locally (with the file's SHA-256 and what Kaggle said was left today after it) and remind Claude
to fetch the public score and attach it to the experiment ledger. Kaggle's replies are matched to the
submits in output order; a submit with no reply (text quoting one, or one after a step that failed) is not
logged, unless the command went on in the background. After each GPU or TPU `kaggle kernels push` in the
command (`--help` calls aside), record the run in the GPU-hour log; a re-push of a kernel replaces its earlier
reservation. `cwd` here is where the command left the shell: relative paths are read from where its `cd` steps
were at that point, and a push whose folder cannot be told is reported, not logged. A command that went on in
the background has left `cwd` where it started, and Kaggle's reply comes after this hook: its pushes are logged
as pending, their whole cap reserved, until `kgkit gpu status` / `gpu wait` see Kaggle run them. The Kaggle CLI
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


# Kaggle's reply to each submit, as the Kaggle CLI (2.2.4) prints it. Accepted: "N submissions remaining today." (left
# out with -q, or when the competition comes from the CLI config), then the server's message with no newline after
# it: "Successfully submitted to <title>" for a file, nothing for a notebook. Refused: "Could not submit to
# competition" (a failed upload) or "Could not find competition" with exit status 0, or an HTTP error on stderr.
REPLY_RE = re.compile(r"(?<!\d)(?P<left>\d+)\s+submissions?\s+remaining\s+today|(?P<ok>successfully submitted)"
                      r"|(?P<failed>could not submit to competition|could not find competition"
                      r"|\b[45]\d\d (?:client|server) error\b[^\n]*competition|authentication required)",
                      re.IGNORECASE)
RETRIED_RE = re.compile(r"request failed:[^\n]*will retry", re.IGNORECASE)  # an error the CLI retried: no reply
# Printed by kaggle 2.2.4 with exit status 0 after a failed upload, push or model write.
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


def _failure_hint(low: str, label: str = "") -> str:
    if any(w in low for w in ("400 client", "bad request", "failed_precondition")):
        return (f"kaggle-grandmaster: Kaggle refused the submission{label} (400). The CLI hides the reason, which is "
                "in the response body: re-run it through the Python API and print `e.response.text` (snippet in skill "
                "`kaggle-cli`). In code competitions the usual cause is FAILED_PRECONDITION 'Your Notebook uses "
                "non-versioned datasets': attach a versioned dataset (or pin a dataset version) to the notebook, "
                "push a new kernel version and submit that version.")
    return (f"kaggle-grandmaster: the submission{label} appears to have failed. Read the CLI output; 403 usually "
            "means the competition rules were not accepted.")


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


def _replies(text: str) -> list[tuple[str, int | str | None]]:
    """Kaggle's replies to the submits in a command's output, in order: ("left", N), ("ok", None) or ("failed", the
    error). Errors in a row are one reply: one refused submit can print more than one. A count printed right after
    the previous submit's message is glued to the title that message ends with, and a title can end in a number
    ("... Episode 10" + "3 submissions"): its last digit is taken (daily limits above 10 are rare)."""
    out, text = [], RETRIED_RE.sub("", text)
    for m in REPLY_RE.finditer(text):
        kind = m.lastgroup
        if kind == "left":
            glued = text[text.rfind("\n", 0, m.start()) + 1:m.start()].strip()
            out.append((kind, int(m.group("left")[-1] if glued else m.group("left"))))
        elif kind == "ok":
            out.append((kind, None))
        elif not (out and out[-1][0] == "failed"):
            out.append((kind, m.group(0).lower()))
    return out


def _match_replies(subs: list[dict], replies: list, unseen: bool) -> list[tuple]:
    """(submit, status, remaining count, error) for the submits Kaggle replied to, its replies taken in order. A
    submit with no reply never ran (an earlier step failed) or is only text, and is left out. Its status is
    "unknown" instead when the output is not all there (``unseen``: the command went on in the background), or when
    Kaggle prints nothing for it once accepted (a notebook submit with -q or without the competition: it takes a
    reply only if that is an error). Replies left over come from a submit that ran more than once, in a loop: they
    are logged as the last one's."""
    out, i = [], 0
    for sub in subs:
        silent = bool(sub.get("kernel")) and bool(sub.get("quiet") or not sub.get("competition"))
        if i < len(replies) and (replies[i][0] == "failed" or not silent):
            i = _take_reply(sub, replies, i, out)
        elif silent or unseen:
            out.append((sub, "unknown", None, None))
    while subs and i < len(replies):
        if replies[i][0] == "failed":
            i += 1
        else:
            i = _take_reply(dict(subs[-1], repeat=True), replies, i, out)
    return out


def _take_reply(sub: dict, replies: list, i: int, out: list) -> int:
    kind, value = replies[i]
    i += 1
    if kind == "left" and i < len(replies) and replies[i][0] == "ok":
        i += 1  # the server's message after the count: "Successfully submitted to ..." for a file
    out.append((sub, "failed", None, value) if kind == "failed"
               else (sub, "submitted", value if kind == "left" else None, None))
    return i


def log_submissions(payload: dict, subs: list[dict]) -> list[str]:
    """Log each submit in the command that Kaggle replied to (see `_match_replies`), and say what comes next."""
    root = C.workspace(payload)
    if root is None or not subs:
        return []
    # An upload/push error earlier in the same command must not be read as a submit's failure.
    text = SILENT_FAILURE_RE.sub("", _response_text(payload))
    background = _backgrounded(payload)  # Kaggle has not replied yet, and cwd is where the command started
    outcomes = _match_replies(subs, _replies(text), background)
    st = C.S.load(root)
    recs = [_log_submission(payload, root, sub, status, left, ran=not background)
            for sub, status, left, _ in outcomes]
    notes = []
    for (sub, status, _, error), rec in zip(outcomes, recs):
        label = f" {C.submit_label(sub)}" if len(subs) > 1 else ""
        if sub.get("repeat"):
            continue
        notes.append(_failure_hint(error, label) if status == "failed" else _logged_note(root, st, sub, rec, label))
    slug = subs[-1].get("competition") or (st.slug if st else "<slug>")
    statuses = [status for _, status, _, _ in outcomes]
    if "submitted" in statuses and "failed" in statuses:
        notes.append(f"kaggle-grandmaster: Kaggle accepted {statuses.count('submitted')} and refused "
                     f"{statuses.count('failed')} of the submits in this command. The hook matches Kaggle's replies "
                     "to the submits in output order, but the CLI prints HTTP errors on stderr, after all of stdout: "
                     f"check which went through with `kaggle competitions submissions {slug}` before attaching scores.")
    repeats = sum(1 for sub, *_ in outcomes if sub.get("repeat"))
    if repeats:
        notes.append(f"kaggle-grandmaster: Kaggle accepted {repeats} more submit(s) than this command has `kaggle "
                     "competitions submit` calls (one runs in a loop?). They are logged too, with the arguments of the "
                     f"last call as written: check `kaggle competitions submissions {slug}` for which notebook or "
                     "file each one was.")
    return notes


def _log_submission(payload: dict, root: Path, sub: dict, status: str, left: int | None, ran: bool) -> dict:
    sha = size = f = None
    if sub.get("file") and not sub.get("kernel"):
        f = C.path_at(payload, sub["file"], sub.get("at", 0), ran=ran)
    if f is not None:
        sha = C.file_sha256(f)
        try:
            size = f.stat().st_size
        except OSError:
            pass
    m = C.EXP_ID_RE.search(sub.get("message") or "") or C.EXP_ID_RE.search(sub.get("file") or "")
    rec = {
        "time_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "competition": sub.get("competition"),
        "file": sub.get("file"),
        "kernel": sub.get("kernel"),
        "message": sub.get("message"),
        "exp_id": m.group(1) if m else None,
        "status": status,
        "sha256": sha,
        "size": size,
        "remaining_today": left,
    }
    log = C.submissions_log(root)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def _logged_note(root: Path, st, sub: dict, rec: dict, label: str) -> str:
    slug = sub.get("competition") or (st.slug if st else "<slug>")
    left, exp_id = rec["remaining_today"], rec["exp_id"]
    count = (f"Kaggle reports {left} left today for the team" if left is not None
             else f"{C.submissions_today(root)}/{st.daily_submissions if st else 5} today in the local log, UTC")
    unknown = "" if rec["status"] == "submitted" else ", outcome unknown: Kaggle's reply is not in this output"
    ctx = (f"kaggle-grandmaster: submission{label} logged{unknown} ({count}). {_poll_hint(root, sub, st, slug)}; "
           f"when the public score appears, run `python -m kgkit ledger lb {exp_id or '<exp_id>'} <score>` and "
           f"compare with CV.")
    if not exp_id:
        ctx += " (No ledger experiment id found in the message/file name — identify which experiment this was.)"
    return ctx


def main() -> None:
    payload = C.read_input()
    cmd = C.command_of(payload)
    # One command can chain an upload, pushes and submits: report on each, in one message.
    notes = [silent_failure_note(payload), *log_gpu_pushes(payload, C.parse_pushes(cmd)),
             *log_submissions(payload, C.parse_submits(cmd))]
    notes = [n for n in notes if n]
    if notes:
        C.emit("PostToolUse", additionalContext="\n\n".join(notes))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
