"""A /kg-grind loop's state, kept in the workspace so it outlives the conversation that runs it (#125).

The loop runs inside a Claude Code session. A long one is compacted (the summary can lose the budget and the
progress), and a closed session takes the loop with it while the runs it started go on: Kaggle kernels always, a
local training process sometimes. So the grind lives in ``.kaggle-gm/grind.json``:

- **One grind per workspace.** Two loops would edit the same scripts and replace each other's accepted baseline;
  ``start`` refuses while one is in progress (``--resume`` continues it).
- **Owned by a session.** The session that drives it (``${CLAUDE_SESSION_ID}`` in /kg-grind) beats a heartbeat with
  every ``status``. A session that was active in the last ``STALE_S`` seconds is not taken over unless asked.
- **Time counts while a session drives it.** Each session's stretch is a segment from its start to its last
  heartbeat; the budget is spent by their sum, so a night between two sessions costs nothing.
- **Experiments done are counted from the ledger** (records logged since the start): nothing to remember.
- **Runs in flight are reconciled** (``reconcile``): every run the loop started (``note_started``) is checked against
  the ledger (logged: done), its ``artifacts/<name>/run_<pid>.json`` (the templates' heartbeat: is that process
  alive on this machine? which folds are done, which resume from a checkpoint?) and the Kaggle run log (running,
  finished and waiting for ``gpu collect``, or collected).

``stop`` appends the grind to ``.kaggle-gm/grinds.jsonl`` and removes the state. Standard library only: the
session brief reads it.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

from . import filelock as FL
from . import state as S

STATE_FILE = "grind.json"
HISTORY_FILE = "grinds.jsonl"
STALE_S = 1800  # a session silent for longer no longer drives the grind: another may resume it


class GrindError(Exception):
    """A request the grind state refuses (a second grind, a live owner, nothing to resume)."""


def _path(root) -> Path:
    return Path(root) / S.STATE_DIR / STATE_FILE


def load(root) -> dict | None:
    try:
        g = json.loads(_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return g if isinstance(g, dict) and g.get("segments") else None


def _save(root, g: dict) -> None:
    p = _path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(g, indent=1), encoding="utf-8")
    FL.replace(tmp, p)


def _when(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


def _ago(s: float) -> str:
    s = max(0.0, s)
    return f"{s / 60:.0f} min ago" if s < 5400 else f"{s / 3600:.1f} h ago"


def parse_budget(text: str) -> dict:
    """``4h``, ``90m``, ``10`` (experiments) or both (``4h,10``) -> {"hours": 4.0, "experiments": 10}."""
    out: dict = {}
    for part in [p.strip().lower() for p in str(text).split(",") if p.strip()]:
        if part.endswith("h"):
            out["hours"] = float(part[:-1])
        elif part.endswith("m") and not part.endswith("exp"):
            out["hours"] = float(part[:-1]) / 60
        else:
            out["experiments"] = int(part.removesuffix("experiments").removesuffix("exp").strip())
    if not out:
        raise GrindError(f"budget {text!r}: give hours (4h, 90m), a number of experiments (10), or both (4h,10)")
    return out


def start(root, session: str, hours: float | None = None, experiments: int | None = None, focus: str = "",
          resume: bool = False, takeover: bool = False, now: float | None = None) -> dict:
    """Start a grind, or with ``resume`` continue the one in progress from ``session``."""
    now = time.time() if now is None else now
    with FL.locked(_path(root)):
        g = load(root)
        if g is not None and not resume:
            raise GrindError(f"a grind is already in progress (started {_when(g['started'])}, driven by session "
                             f"{g['owner'][:12]}, last active {_ago(now - g['segments'][-1]['last'])}): continue it with "
                             "`kgkit grind start --resume`, or end it with `kgkit grind stop` first")
        if g is None and resume:
            raise GrindError("no grind in progress to resume: start one with `kgkit grind start --budget <4h|10>`")
        if g is not None:  # resume
            if g["owner"] == session:
                g["segments"][-1]["last"] = now
            else:
                idle = now - g["segments"][-1]["last"]
                if idle < STALE_S and not takeover:
                    raise GrindError(f"the grind is driven by another session ({g['owner'][:12]}, active "
                                     f"{_ago(idle)}): two loops would race each other. If that session is gone, "
                                     "resume with --takeover")
                g["owner"] = session
                g["segments"].append({"session": session, "start": now, "last": now})
            _save(root, g)
            return g
        budget = {}
        if hours:
            budget["hours"] = float(hours)
        if experiments:
            budget["experiments"] = int(experiments)
        if not budget:
            raise GrindError("a grind needs a budget: --budget 4h, --budget 10 (experiments) or both (4h,10)")
        g = {"id": time.strftime("%Y%m%d-%H%M%S", time.localtime(now)), "started": now, "owner": session,
             "budget": budget, "focus": focus or "", "segments": [{"session": session, "start": now, "last": now}],
             "in_flight": []}
        _save(root, g)
        return g


def _beat(g: dict, session: str | None, now: float) -> bool:
    if session and g["owner"] == session:
        g["segments"][-1]["last"] = max(g["segments"][-1]["last"], now)
        return True
    return False


def note_started(root, session: str, name: str, cmd: str, now: float | None = None) -> dict:
    """Record a run the loop is about to start (its ledger / artifacts name and its command)."""
    now = time.time() if now is None else now
    with FL.locked(_path(root)):
        g = load(root)
        if g is None:
            raise GrindError("no grind in progress: `kgkit grind start` first")
        g["in_flight"] = [r for r in g.get("in_flight", []) if r["name"] != name]
        g["in_flight"].append({"name": name, "cmd": cmd, "started": now, "session": session})
        _beat(g, session, now)
        _save(root, g)
        return g


def active_hours(g: dict) -> float:
    return sum(max(0.0, s["last"] - s["start"]) for s in g["segments"]) / 3600


def _ledger(root) -> list[dict]:
    p = Path(root) / S.STATE_DIR / "ledger.jsonl"
    out = []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            r = json.loads(line)
            r["_t"] = time.mktime(time.strptime(r.get("time", ""), "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, TypeError, OverflowError):
            continue
        out.append(r)
    return out


def experiments_done(root, g: dict) -> int:
    return sum(1 for r in _ledger(root) if r["_t"] >= g["started"] - 1)


def pid_alive(pid: int) -> bool | None:
    """Whether process ``pid`` of this machine is still running; None when that cannot be told. On Windows by the
    Win32 API: os.kill(pid, 0) would send it CTRL_C_EVENT there."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenProcess.restype = wintypes.HANDLE
            h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return ctypes.get_last_error() == 5  # access denied: it exists; otherwise it is gone
            try:
                code = wintypes.DWORD()
                if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                    return None
                return code.value == 259  # STILL_ACTIVE
            finally:
                k32.CloseHandle(h)
        except (OSError, AttributeError):
            return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _fold_done(art: Path, f: int) -> bool:
    st = art / f"fold{f}_status.json"
    try:
        if json.loads(st.read_text(encoding="utf-8")).get("status") == "done":
            return True
    except (OSError, ValueError):
        pass
    return (art / f"fold{f}_oof.npy").is_file()


def _epochs_done(art: Path, f: int) -> int:
    try:
        return len(json.loads((art / f"curve_fold{f}.json").read_text(encoding="utf-8")).get("points") or [])
    except (OSError, ValueError):
        return 0


def _local(root, entry: dict, logged: dict) -> dict:
    name = entry["name"]
    out = {"name": name, "cmd": entry.get("cmd", "")}
    rec = logged.get(name)
    if rec is not None and rec["_t"] >= entry["started"] - 5:
        return {**out, "state": "done", "detail": f"logged as {rec['id']}"}
    art = Path(root) / "artifacts" / name
    runs = []
    for p in sorted(art.glob("run_*.json")):
        try:
            runs.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    if not runs:
        if any(art.glob("fold*_oof.npy")):
            return {**out, "state": "stopped", "detail": "folds on disk but no heartbeat file (not a template run?)",
                    "resume": f"check artifacts/{name}/ and re-run `{entry.get('cmd', '')}` for the missing folds"}
        return {**out, "state": "unknown", "detail": "nothing of it on disk: it never started, or does not write "
                                                     f"artifacts/{name}/"}
    me = socket.gethostname()
    live = [r for r in runs if not r.get("finished") and r.get("host") == me and pid_alive(r.get("pid")) is True]
    if live:
        beat = max(r.get("heartbeat") or 0 for r in live)
        return {**out, "state": "running", "detail": f"process {live[0]['pid']} is running (heartbeat "
                                                     f"{_ago(time.time() - beat)}): wait for it"}
    elsewhere = [r for r in runs if not r.get("finished") and r.get("host") != me]
    if elsewhere:
        return {**out, "state": "unknown", "detail": f"it runs on {elsewhere[0].get('host')}, not this machine: "
                                                     "check there"}
    folds = sorted({f for r in runs for f in (r.get("folds") or [])})
    done = [f for f in folds if _fold_done(art, f)]
    left = [f for f in folds if f not in done]
    if not left:
        return {**out, "state": "stopped", "folds_done": done, "folds_left": [],
                "detail": "every fold it ran is done but it is not in the ledger",
                "resume": f"`{entry.get('cmd', '')} --assemble` logs it (or the run that completes the set does)"}
    notes = [f"fold {f} resumes from its checkpoint (after epoch {_epochs_done(art, f)})"
             for f in left if (art / f"fold{f}_resume.pt").is_file()]
    detail = f"its process is gone: folds done {done or 'none'}, left {left}" + (f"; {'; '.join(notes)}" if notes else "")
    return {**out, "state": "stopped", "folds_done": done, "folds_left": left, "detail": detail,
            "resume": f"re-run `{entry.get('cmd', '')}` with `--folds {' '.join(map(str, left))}`"}


def _kaggle(root, g: dict) -> list[dict]:
    from . import gpu as G  # standard library only too

    out = []
    for r in G.runs(root):
        t = G._parse_time(r.get("pushed_at"))
        if t is None or t.timestamp() < g["started"] - 5 or r.get("status") in (G.DELETED, "superseded"):
            continue
        k, s = r.get("kernel", "?"), r.get("status")
        if s in G.ACTIVE:
            state, detail, nxt = "running", f"on Kaggle, {s}", f"`kgkit gpu wait {k} --collect` in the background"
        elif s in G.FINISHED:
            state, detail, nxt = "finished", f"ended on Kaggle ({s}), not collected", f"`kgkit gpu collect {k}`"
        else:
            state, detail, nxt = "done", f"collected ({s})", ""
        out.append({"name": r.get("name") or k, "kernel": k, "state": state, "detail": detail, "resume": nxt})
    return out


def reconcile(root, g: dict) -> list[dict]:
    """Every run the grind started, and the Kaggle runs pushed since it began: done, running, stopped (with how to
    resume), finished on Kaggle (to collect) or unknown."""
    logged = {}
    for r in _ledger(root):
        logged[r.get("name")] = r  # the latest record of each name
    return [_local(root, e, logged) for e in g.get("in_flight", [])] + _kaggle(root, g)


def status(root, session: str | None = None, now: float | None = None) -> dict | None:
    """The grind with its progress: a heartbeat when ``session`` drives it."""
    now = time.time() if now is None else now
    with FL.locked(_path(root)):
        g = load(root)
        if g is None:
            return None
        if _beat(g, session, now):
            _save(root, g)
    out = dict(g)
    out["active_hours"] = active_hours(g)
    out["experiments_done"] = experiments_done(root, g)
    if "hours" in g["budget"]:
        out["hours_left"] = max(0.0, g["budget"]["hours"] - out["active_hours"])
    if "experiments" in g["budget"]:
        out["experiments_left"] = max(0, g["budget"]["experiments"] - out["experiments_done"])
    out["spent"] = out.get("hours_left", 1) <= 0 or out.get("experiments_left", 1) <= 0
    out["idle_s"] = now - g["segments"][-1]["last"]
    return out


def stop(root, session: str | None = None, reason: str = "", now: float | None = None) -> dict | None:
    now = time.time() if now is None else now
    with FL.locked(_path(root)):
        g = load(root)
        if g is None:
            return None
        _beat(g, session, now)
        g.update(stopped_at=now, stopped_reason=reason or "", active_hours=round(active_hours(g), 3),
                 experiments_done=experiments_done(root, g))
        hist = Path(root) / S.STATE_DIR / HISTORY_FILE
        with FL.locked(hist):
            with hist.open("a", encoding="utf-8") as f:
                f.write(json.dumps(g) + "\n")
        _path(root).unlink(missing_ok=True)
        return g


def _progress(s: dict) -> str:
    bits = []
    if "experiments" in s["budget"]:
        bits.append(f"{s['experiments_done']}/{s['budget']['experiments']} experiments")
    else:
        bits.append(f"{s['experiments_done']} experiments")
    if "hours" in s["budget"]:
        bits.append(f"{s['active_hours']:.1f} h of {s['budget']['hours']:g} h active")
    return ", ".join(bits)


def brief_line(root, now: float | None = None) -> str:
    """The session brief's line for a grind in progress ('' when none): it is re-injected after every compaction and
    in a new session, so the loop is continued rather than restarted or forgotten."""
    try:
        s = status(root, None, now)
    except (OSError, FL.LockTimeout):
        return ""
    if s is None:
        return ""
    runs = reconcile(root, s)
    counts: dict = {}
    for r in runs:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    flight = ", ".join(f"{n} {k}" for k, n in counts.items() if k != "done")
    return (f"Grind in progress since {_when(s['started'])} (session {s['owner'][:12]}, last active "
            f"{_ago(s['idle_s'])}): {_progress(s)}" + ("; BUDGET SPENT" if s["spent"] else "")
            + (f"; runs it started: {flight}" if flight else "")
            + ". Continue it with `/kg-grind resume` (`kgkit grind status` lists each run and how to pick it up); "
              "end it with `kgkit grind stop`.")


def format_status(s: dict) -> str:
    lines = [f"grind {s['id']}: {_progress(s)}" + (f", focus: {s['focus']}" if s.get("focus") else ""),
             f"driven by session {s['owner'][:12]}, last active {_ago(s['idle_s'])}; {len(s['segments'])} session(s)"]
    if "hours_left" in s:
        lines.append(f"hours left: {s['hours_left']:.2f}")
    if "experiments_left" in s:
        lines.append(f"experiments left: {s['experiments_left']}")
    if s["spent"]:
        lines.append("BUDGET SPENT: finish the run in flight, write the report, then `kgkit grind stop`")
    for r in s.get("runs", []):
        lines.append(f"  [{r['state']}] {r['name']}: {r['detail']}" + (f" -> {r['resume']}" if r.get("resume") else ""))
    return "\n".join(lines)
