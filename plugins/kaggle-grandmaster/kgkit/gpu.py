"""Training on Kaggle's GPUs (and TPUs): quota, remote training kernels, and accelerator-hour accounting.

Kaggle gives each account a weekly GPU quota (``kaggle quota``; 30 h/week at the time of writing) and a
separate TPU quota (20 h/week), neither of which rolls over, with a per-session limit of 12 h (GPU) or
9 h (TPU). This module turns a local training command into a kernel that runs it on Kaggle (both T4s
busy, or one process per TPU chip; deadline-aware, resumable), and keeps ``.kaggle-gm/gpu_runs.jsonl``
so every accelerator-hour can be traced to the experiment it bought::

    python -m kgkit gpu quota                                   # live GPU + TPU quota, reset countdown
    python -m kgkit gpu build --name cnx384 --hours 6 \\
        --cmd "python src/train_image.py --name cnx384 --img-size 384 --folds {fold}"
    python -m kgkit gpu push kernels/cnx384-train               # asks nothing; check quota first
    python -m kgkit gpu push kernels/my-notebook --hours 1      # any kernel folder (kernel-metadata.json)
    python -m kgkit gpu wait <user>/cnx384-train                # poll until done (run in background)
    python -m kgkit gpu collect <user>/cnx384-train             # outputs -> ledger + GPU-hour log
    python -m kgkit gpu collect <user>/my-notebook              # any kernel: files -> data/ext/kout-my-notebook/
    python -m kgkit gpu plan                                    # budget until the deadline
    python -m kgkit gpu release <user>/<kernel>                 # drop the reservation of a finished run

Standard library only: hooks import it to read the cached quota and the run log.
"""

from __future__ import annotations

import base64
import csv
import datetime as dt
import difflib
import hashlib
import io
import itertools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
import zlib
from pathlib import Path, PurePosixPath

from . import evalset as E
from . import kaggle_cli as KC
from . import safety
from . import state as S

# The runner is part of the package, not of the plugin around it: `kgkit vendor` copies only the package, and a copy
# must fill in the runner its own config format was written for (the plugin's may be another version's).
KGKIT_DIR = Path(__file__).resolve().parent
RUNNER = KGKIT_DIR / "kaggle_gpu_runner.py"
RUNS_FILE = "gpu_runs.jsonl"
QUOTA_FILE = "gpu_quota.json"
SESSION_LIMIT_H = 12.0
SESSION_LIMITS = {"GPU": 12.0, "TPU": 9.0, "CPU": 12.0}
# Batch sessions an account may run at once; a push beyond this fails ("Maximum batch GPU session count
# of 2 reached", printed with exit status 0).
SESSION_SLOTS = {"GPU": 2, "TPU": 1}
# How long a pushed run may sit in the queue before an entry that never reported "running" is stale.
QUEUE_ALLOWANCE_H = {"GPU": 1.0, "TPU": 36.0, "CPU": 1.0}
# `kaggle quota` is tried this many times, this many seconds apart: the CLI fails now and then on a flaky network.
QUOTA_TRIES, QUOTA_RETRY_S = 2, 3.0
# A run logged since a cached quota counts against it when it may have spent at least this many hours of it.
UNSEEN_MIN_H = 0.01
# The `-t` of a push is a hard kill: Kaggle ends the session at the cap, wherever the script is, and reports the run as
# cancelled. A cancelled run that got within this many seconds of its cap (or this fraction of it, if that is less) was
# stopped by it; the log's last line is up to one progress message earlier than the kill.
CAP_SLACK_S, CAP_SLACK_FRAC = 600.0, 0.1
OUTPUT_TIMEOUT = 180.0  # seconds for the request that lists a stopped session's files and log
ACCELERATORS = {
    "NvidiaTeslaT4": "2x T4 16 GB (two GPUs per quota hour: run two folds in parallel)",
    "NvidiaTeslaP100": "1x P100 16 GB (one GPU; no Triton/torch.compile, sm_60)",
    "TpuV5E8": "TPU v5e-8, 8 chips (bf16; TPU quota; queued for hours; one process per chip for {fold} tasks)",
    "TpuV6E8": "TPU v6e-8, 8 chips (separate pool, often no capacity)",
}
TPU_CHIPS = {"TpuV5E8": 8, "TpuV6E8": 8}
# Kaggle stores these as TpuV5E8 (verified Oct 2026): ask for the shape that will actually run.
TPU_REWRITTEN = {"TpuV5E1", "TpuV5E4", "TpuV6E1"}
# What a push may ask for (`--accelerator`, or machine_shape in kernel-metadata.json; verified Oct 2026): the shapes
# `gpu build` targets, and Tpu1VmV38 (TPU v3-8, its own queue). Kaggle answers any other name with a bare 400, such
# as one `kaggle kernels pull -m` wrote: the shape of the kernel's last session, which can carry hardware details a
# push cannot ask for (NvidiaTeslaT4Highmem).
MACHINE_SHAPES = {"NvidiaTeslaT4": "2x T4", "NvidiaTeslaP100": "1x P100", "TpuV5E8": "TPU v5e-8",
                  "TpuV6E8": "TPU v6e-8", "Tpu1VmV38": "TPU v3-8"}
WEIGHT_RE = r"\.(pt|pth|bin|ckpt|safetensors)$"
BUNDLE_EXCLUDE_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "wandb", ".venv", "venv"}
BUNDLE_EXCLUDE_SUFFIXES = {".pyc", ".pt", ".pth", ".bin", ".ckpt", ".safetensors", ".npy", ".npz", ".pkl",
                           ".parquet", ".feather", ".zip", ".png", ".jpg", ".jpeg", ".dcm"}
# Kaggle refuses a kernel source of 1 MB or more with a bare 400 Bad Request from SaveKernel (its editor says it "must
# be less than 1 megabytes"; Oct 2026: a 2.6 MB `gpu build` script refused, the same kernel at 0.57 MB taken). The code
# bundle is in the script as base64, 4/3 of its size.
SOURCE_MAX_BYTES = 1_000_000
# The fold split goes in the bundle column by column (`pack_split`); the runner writes data/folds.csv back from it.
SPLIT_FILE, SPLIT_ARC = "data/folds.csv", "_kg/split"
SPLIT_CODES_MAX = 256  # a column with at most this many distinct values (the folds) takes one byte per row
SPLIT_REF_KB = 64  # a column bigger than this as (compressed) text is read from a training file in the session instead
# Where `gpu build` links attached datasets and kernel outputs in the run (competition data is always in data/).
INPUT_LAYOUTS = {"flat": "into data/, beside the competition data",
                 "nested": "each in its own folder: data/ext/<slug>/, kernel outputs data/ext/kout-<slug>/"}
# A push in a command that went on in the background: Kaggle's reply (and the version) came after the hook ran,
# so the run may not have started. It reserves its whole cap and takes no status from Kaggle, which still reports
# an earlier version, until Kaggle lists a version of the kernel run after it (`confirm_pending`).
PENDING = "pending"
ACTIVE = (PENDING, "pushed", "queued", "running")
FINISHED = ("complete", "error", "cancelled")
# The statuses of a run whose session is over (`collect` adds the last three): the runs whose hours count.
ENDED = (*FINISHED, "collected", "partial", "failed")
CLOCK_SKEW_S = 60  # this machine's clock vs Kaggle's, when a pending push is matched to the kernel's last run


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _slug(s: str, n: int = 50) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:n].strip("-") or "run"


def _root(root: str | Path | None) -> Path:
    if root is not None:
        return Path(root)
    return S.find_root() or Path.cwd()


def resource_of(accelerator: str | None) -> str:
    """'TPU', 'CPU' or 'GPU' for an accelerator / machine_shape name (unknown names count as GPU)."""
    a = str(accelerator or "").strip().lower()
    if a.startswith("tpu"):
        return "TPU"
    if a in ("none", "cpu", "no"):
        return "CPU"
    return "GPU"


def run_resource(r: dict) -> str:
    return r.get("resource") or resource_of(r.get("accelerator"))


def session_limit(resource: str) -> float:
    return SESSION_LIMITS.get(resource, SESSION_LIMIT_H)


def machine_shape_problem(shape: str | None) -> str | None:
    """Why a push must not ask Kaggle for ``shape`` (a machine_shape or --accelerator), with the shape to ask for
    instead; None when Kaggle takes it, or when it asks for no accelerator."""
    s = str(shape or "").strip()
    if not s or s in MACHINE_SHAPES or s.lower() in ("none", "cpu", "no"):
        return None
    if s in TPU_REWRITTEN:
        return f"machine_shape {s} is stored by Kaggle as TpuV5E8: ask for TpuV5E8"
    low = s.lower()
    # an accepted name with hardware details after it (a recorded shape), the one accepted name holding it (T4), else
    # the closest name (a typo)
    near = max((k for k in MACHINE_SHAPES if low.startswith(k.lower())), key=len, default=None)
    holding = [k for k in MACHINE_SHAPES if low in k.lower()]
    near = near or (holding[0] if len(holding) == 1 else None)
    if near is None:
        score = {k: difflib.SequenceMatcher(None, low, k.lower()).ratio() for k in MACHINE_SHAPES}
        near = max(score, key=score.get)
        near = near if score[near] >= 0.6 else None
    return (f"machine_shape {s!r} is not one Kaggle's kernel push is known to take (Kaggle answers an unknown shape "
            "with a bare 400 Bad Request)" + (f": ask for {near}" if near else "") + ". Shapes a push can ask for: "
            + ", ".join(f"{k} ({v})" for k, v in MACHINE_SHAPES.items()))


# ----------------------------------------------------------------------------- kaggle CLI
PY_NAME_RE = re.compile(r"(?:python|pypy)[\d.]*(?:\.exe)?", re.IGNORECASE)
PIP_SH_RE = re.compile(r"""'''exec' (["']?)(.+?)\1 "\$0" "\$@"\s*$""")  # pip's #!/bin/sh line 2, for long paths
_PLAIN: set[str] = set()  # interpreters the Kaggle CLI cannot be imported from: the plain CLI runs instead


def kaggle_cmd() -> list[str]:
    if os.environ.get("KGKIT_KAGGLE"):
        return shlex.split(os.environ["KGKIT_KAGGLE"], posix=os.name != "nt")
    exe = shutil.which("kaggle")
    return [exe] if exe else [sys.executable, "-m", "kaggle"]


def kaggle_python() -> str | None:
    """The Python the Kaggle CLI on PATH runs on, from its script's #! line (`#!/usr/bin/env python3` and pip's
    `#!/bin/sh` form included), or this Python when the CLI is not on PATH (it then runs as `python -m kaggle`).
    None when that cannot be told: a Windows .exe launcher, or a #! line naming no Python (a .py file must never be
    handed to a shell)."""
    exe = shutil.which("kaggle")
    if exe is None:
        return sys.executable
    try:
        lines = Path(exe).read_bytes()[:1024].decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    words = lines[0][2:].split() if lines and lines[0].startswith("#!") else []
    if len(words) > 1 and Path(words[0]).name == "env":
        words = [shutil.which(words[1]) or ""]
    elif words and Path(words[0]).name == "sh":
        m = PIP_SH_RE.match(lines[1]) if len(lines) > 1 else None
        words = [m.group(2)] if m else []
    py = words[0] if words else ""
    return py if PY_NAME_RE.fullmatch(Path(py).name) and Path(py).is_file() else None


def _run_cli(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace",
                          env=safety.kaggle_env())


def run_kaggle(args: list[str], timeout: float = 600) -> subprocess.CompletedProcess:
    """Run the Kaggle CLI: through `kaggle_cli.py` under the CLI's own Python when `kaggle_python` can tell it, so a
    refused request shows Kaggle's reason; otherwise as `KGKIT_KAGGLE` (used as given), or plain."""
    py = None if os.environ.get("KGKIT_KAGGLE") else kaggle_python()
    if py and py not in _PLAIN:
        try:
            r = _run_cli([py, str(Path(KC.__file__).resolve()), *args], timeout)
        except OSError:
            r = None
        if r is not None and r.returncode != KC.NO_CLI:
            return r
        _PLAIN.add(py)
    return _run_cli([*kaggle_cmd(), *args], timeout)


_USERNAME: dict[str, str | None] = {}


def kaggle_username(probe: bool = True) -> str | None:
    """The Kaggle username, never the key or token: KAGGLE_USERNAME, else the ``username`` field of
    kaggle.json, else (``probe``) the ``username`` line of ``kaggle config view``, which the CLI resolves
    for access-token and OAuth logins too (CLI 2.x; it prints no secret)."""
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    cfg_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle")
    try:
        user = json.loads((cfg_dir / "kaggle.json").read_text(encoding="utf-8")).get("username")
        if user:
            return str(user)
    except (OSError, ValueError, AttributeError):
        pass
    if not probe:
        return None
    if "probe" not in _USERNAME:
        user = None
        try:
            r = run_kaggle(["config", "view"], timeout=60)
            m = re.search(r"^\s*-\s*username:\s*(\S+)\s*$", r.stdout or "", re.MULTILINE)
            user = m.group(1) if m and m.group(1) != "None" else None
        except (OSError, subprocess.SubprocessError):
            pass
        _USERNAME["probe"] = user
    return _USERNAME["probe"]


# ----------------------------------------------------------------------------- quota
def _hours(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    m = re.match(r"\s*([0-9.]+)\s*h?", str(v))
    return float(m.group(1)) if m else 0.0


def _quota_row(r: dict) -> dict:
    refresh = _parse_time(r.get("refreshAt") or r.get("refresh_at"))
    return {"used_h": _hours(r.get("used")), "remaining_h": _hours(r.get("remaining")),
            "total_h": _hours(r.get("total")), "refresh_at": _iso(refresh) if refresh else None,
            "fetched_at": _iso(_now())}


def parse_quota(text: str) -> dict | None:
    """Parse ``kaggle quota --format json``: the GPU row, with the TPU row (if any) under ``tpu``."""
    try:
        rows = json.loads(text[text.find("["):] if "[" in text else text)
    except ValueError:
        return None
    by = {str(r.get("resource", "")).upper(): r for r in (rows if isinstance(rows, list) else [])
          if isinstance(r, dict)}
    if "GPU" not in by:
        return None
    q = _quota_row(by["GPU"])
    if "TPU" in by:
        q["tpu"] = _quota_row(by["TPU"])
    return q


def quota_path(root: Path) -> Path:
    return root / S.STATE_DIR / QUOTA_FILE


def _cli_failure(r: subprocess.CompletedProcess) -> str:
    """What a failed CLI call says went wrong, on one line: Kaggle's reason when `run_kaggle` printed it, else the
    last line of output (a traceback ends in its exception), else the exit status."""
    lines = [l.strip() for l in f"{r.stdout or ''}\n{r.stderr or ''}".splitlines() if l.strip()]
    why = next((l for l in lines if l.startswith(KC.REASON)), lines[-1] if lines else f"exit status {r.returncode}")
    return why if len(why) <= 200 else why[:197] + "..."


def live_quota(root: str | Path | None = None) -> tuple[dict | None, str | None]:
    """(quota, None) from the Kaggle API, cached in .kaggle-gm/gpu_quota.json; (None, why) when the CLI failed
    ``QUOTA_TRIES`` times, ``why`` being 'failed twice (<its last line>)'. A call that timed out or could not start is
    not repeated: a hung network or a missing CLI does not recover in seconds."""
    root = _root(root)
    why = "(not tried)"
    for i in range(QUOTA_TRIES):
        if i:
            time.sleep(QUOTA_RETRY_S)
        try:
            r = run_kaggle(["quota", "--format", "json"], timeout=60)
        except subprocess.TimeoutExpired:
            return None, "timed out after 60 s"
        except (OSError, subprocess.SubprocessError) as e:
            return None, f"could not run the Kaggle CLI ({type(e).__name__}: {e})"
        q = parse_quota(r.stdout) if r.returncode == 0 else None
        if q is not None:
            p = quota_path(root)
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(q, indent=1), encoding="utf-8")
            except OSError:  # a read-only workspace: the live value is still good for this call
                pass
            return q, None
        why = f"({_cli_failure(r) if r.returncode else 'no GPU row in its output'})"
    return None, f"failed{' twice' if QUOTA_TRIES == 2 else f' {QUOTA_TRIES} times'} {why}"


def fetch_quota(root: str | Path | None = None) -> dict | None:
    """Live quota from the Kaggle API (`live_quota`), cached in .kaggle-gm/gpu_quota.json. None when the CLI fails."""
    return live_quota(root)[0]


def _rolled(q: dict) -> dict:
    """``q`` as of now: when the week rolled over since it was taken, a full quota as of the latest reset that has
    passed (``reset_at``, also ``as_of``), with ``refresh_at`` the next one."""
    refresh = _parse_time(q.get("refresh_at"))
    now = _now()
    if refresh is None or now < refresh:
        return q
    reset = refresh + dt.timedelta(days=7) * ((now - refresh) // dt.timedelta(days=7))
    return {**q, "remaining_h": q.get("total_h", 0.0), "used_h": 0.0, "stale_reset": True, "reset_at": _iso(reset),
            "as_of": _iso(reset), "refresh_at": _iso(reset + dt.timedelta(days=7))}


def _as_of(q: dict) -> dt.datetime:
    """When ``q``'s figures were true: when it was fetched, or the reset it rolled over to."""
    return _parse_time(q.get("as_of")) or _parse_time(q.get("fetched_at")) or _now()


def cached_quota(root: str | Path, resource: str = "GPU") -> dict | None:
    p = quota_path(Path(root))
    try:
        q = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if resource == "TPU":
        q = q.get("tpu") if isinstance(q.get("tpu"), dict) else None
        if q is None:
            return None
    return _rolled(q)


def _stale(r: dict, now: dt.datetime) -> bool:
    """An active-looking entry that must have ended: started (or, never seen running, pushed) longer ago
    than its cap plus the queue allowance."""
    res = run_resource(r)
    cap = float(r.get("hours") or session_limit(res))
    started = _parse_time(r.get("started_at"))
    if started is not None:
        return (now - started).total_seconds() / 3600 > cap + 1
    pushed = _parse_time(r.get("pushed_at"))
    if pushed is None:
        return False
    return (now - pushed).total_seconds() / 3600 > cap + QUEUE_ALLOWANCE_H.get(res, 1.0)


def active_runs(root: str | Path, resource: str | None = None) -> list[dict]:
    """Runs the log still has as pushed/queued/running, minus entries that must have ended long ago."""
    now = _now()
    return [r for r in runs(root) if r.get("status") in ACTIVE and not _stale(r, now)
            and (resource is None or run_resource(r) == resource)]


def run_label(r: dict) -> str:
    return f"{r.get('kernel', '?')}" + (f" v{r['version']}" if r.get("version") else "")


def reserved_hours(r: dict, fetched: dt.datetime | None = None) -> float:
    """Quota an active run may still spend after the quota snapshot taken at ``fetched``. A queued run
    (TPU jobs wait for hours) or a pending push reserves its whole cap; a running one its cap minus the time it
    has run."""
    res = run_resource(r)
    cap = float(r.get("hours") or session_limit(res))
    start = _parse_time(r.get("started_at"))
    if start is None and res != "TPU" and r.get("status") not in ("queued", PENDING):
        start = _parse_time(r.get("pushed_at"))  # GPU batch sessions usually start within minutes
    if start is None:
        return cap
    used = max(0.0, ((fetched or _now()) - start).total_seconds() / 3600)
    return max(0.0, cap - used)


def in_flight_hours(root: str | Path, quota: dict | None = None, resource: str = "GPU") -> float:
    """Upper bound of quota that kernels still running will consume after the quota snapshot."""
    fetched = _as_of(quota or {})
    return sum(reserved_hours(r, fetched) for r in active_runs(root, resource))


def hours_key(resource: str) -> str:
    """The run-log field that holds the quota hours a run used."""
    return "tpu_hours" if resource == "TPU" else "gpu_hours"


def session_hours(r: dict, ran_s: float | None = None) -> float | None:
    """The hours the session of a run that has ended lasted, as far as the log can tell: from when it was seen running
    (a GPU or CPU run never seen running: from its push, as `spent_since` does) to when it was seen ended, and at least
    ``ran_s``, how long its session log shows it ran. Never more than its cap: Kaggle kills the session there. None
    when neither is known (a TPU run never seen running, and no log)."""
    start = _parse_time(r.get("started_at"))
    if start is None and run_resource(r) != "TPU":
        start = _parse_time(r.get("pushed_at"))
    end = _parse_time(r.get("finished_at")) or _parse_time(r.get("collected_at"))
    h = max(0.0, (end - start).total_seconds() / 3600) if start is not None and end is not None else None
    if ran_s is not None:
        h = max(h or 0.0, ran_s / 3600)
    if h is None:
        return None
    cap = float(r.get("hours") or 0)
    return round(min(h, cap) if cap else h, 3)


def used_hours(r: dict) -> float | None:
    """The quota hours a run used: the figure recorded for it (`collect`, or the session time when the log saw it end),
    else, for a run that ended without one (an older kgkit, or an entry closed by hand), its `session_hours`."""
    v = r.get(hours_key(run_resource(r)))
    if v is not None:
        return float(v)
    return session_hours(r) if r.get("status") in ENDED else None


def spent_since(r: dict, t: dt.datetime) -> float:
    """The most quota hours a run that is no longer in flight may have spent after ``t``, the time of a quota
    snapshot: its measured hours (``collect``), else its cap, less what it ran before ``t``, and never more than the
    time between ``t`` (or its start) and when the log last saw it end. A TPU run never seen running spent nothing
    unless collected: it may have sat in the queue."""
    res = run_resource(r)
    pushed = _parse_time(r.get("pushed_at"))
    start = _parse_time(r.get("started_at")) or (pushed if res != "TPU" else None)
    end = _parse_time(r.get("finished_at")) or _parse_time(r.get("collected_at")) or _now()
    from_t = max(t, start or pushed or t)
    window = (end - from_t).total_seconds() / 3600
    if pushed is None or window <= 0:
        return 0.0
    measured = r.get(hours_key(res))
    if measured is None and start is None:
        return 0.0
    total = float(measured if measured is not None else r.get("hours") or session_limit(res))
    before = max(0.0, (t - start).total_seconds() / 3600) if start is not None else 0.0
    return min(window, max(0.0, total - before))


def unseen_spend(root: str | Path, quota: dict, resource: str = "GPU") -> list[tuple[dict, float]]:
    """(run, hours) for runs of ``resource`` that are no longer in flight but may have spent quota after ``quota``
    was taken, so its figure is too high by that much. Nothing for a quota fetched live just now: every run it
    could not see had ended before it."""
    t, now = _as_of(quota), _now()
    out = []
    for r in runs(root):
        if r.get("status") == PENDING or run_resource(r) != resource:
            continue
        if r.get("status") in ACTIVE and not _stale(r, now):  # still in flight: `in_flight_hours`
            continue
        h = spent_since(r, t)
        if h >= UNSEEN_MIN_H:
            out.append((r, h))
    return out


def available_hours(root: str | Path, quota: dict | None = None, resource: str = "GPU") -> float | None:
    """Quota hours left to plan with: what ``quota`` says, less what running kernels may still spend and what runs
    logged since the snapshot spent."""
    q = quota or cached_quota(root, resource)
    if q is None:
        return None
    return q["remaining_h"] - in_flight_hours(root, q, resource) - sum(h for _, h in unseen_spend(root, q, resource))


def _when(q: dict) -> tuple[str, str]:
    """('2026-10-24 06:27 UTC', '3.0 days'): when ``q`` was fetched and how long ago."""
    t = _parse_time(q.get("fetched_at"))
    if t is None:
        return "an unknown time", "unknown age"
    h = max(0.0, (_now() - t).total_seconds() / 3600)
    return t.strftime("%Y-%m-%d %H:%M UTC"), "<1h" if h < 1 else f"{h:.0f}h" if h < 48 else f"{h / 24:.1f} days"


def quota_age(q: dict) -> str:
    """'cached 2026-10-24 06:27 UTC, 3.0 days ago' for a quota read from the cache, with the reset that passed since
    when its week is over."""
    when, age = _when(q)
    s = f"cached {when}, {age} ago"
    if q.get("stale_reset"):
        s += f"; the weekly reset at {_iso(_as_of(q))} has passed since, so a full week is counted"
    return s


def _since_note(root: str | Path, q: dict, resource: str) -> str:
    unseen = unseen_spend(root, q, resource)
    return (f"; {len(unseen)} run(s) logged since then count {sum(h for _, h in unseen):.2f}h against it"
            if unseen else "")


def current_quota(root: str | Path | None = None, refresh: bool = True, resource: str = "GPU"
                  ) -> tuple[dict | None, str]:
    """(quota, note) of ``resource`` (its dict also carries ``tpu`` for the GPU one). With ``refresh`` the live quota
    (note ''); when the live call fails, or without ``refresh``, the cached one with a note saying so, why, and how
    old it is. (None, note) when there is nothing to show."""
    root = _root(root)
    why = None
    if refresh:
        live, why = live_quota(root)
        if live is not None:
            q = live.get("tpu") if resource == "TPU" else live
            return (q, "") if q is not None else (None, "`kaggle quota` lists no TPU quota for this account")
    q = cached_quota(root, resource)
    if q is None:
        return None, (f"`kaggle quota` {why} and nothing is cached" if why else
                      "no quota cached yet (`python -m kgkit gpu quota` fetches it)")
    when, age = _when(q)
    note = (f"`kaggle quota` {why}; showing the cached value from {when} ({age} old, may be stale)" if why else
            f"{when} ({age} old; not checked live)")
    if q.get("stale_reset"):
        note += f"; the weekly reset at {_iso(_as_of(q))} has passed since, so a full week is counted"
    return q, note + _since_note(root, q, resource)


def tpu_of(q: dict | None) -> dict | None:
    """The TPU quota in a GPU quota dict, as of now."""
    t = (q or {}).get("tpu")
    return _rolled(t) if isinstance(t, dict) else None


def quota_line(root: str | Path) -> str:
    """Cached Kaggle GPU quota (no network) and remote runs that may still be in flight, for the session brief
    and `kgkit status`; empty when neither is on file."""
    q = cached_quota(root)
    active = active_runs(root)
    if q is None and not active:
        return ""
    s = ""
    if q is not None:
        avail = available_hours(root, q)
        s = f"Kaggle GPU quota ({quota_age(q)}): {avail:.1f}h plannable of {q.get('total_h', 0):.0f}h"
        unseen = unseen_spend(root, q)
        if unseen:
            s += f" ({len(unseen)} run(s) logged since then counted, {sum(h for _, h in unseen):.1f}h)"
        refresh = _parse_time(q.get("refresh_at"))
        if refresh is not None:
            h = (refresh - _now()).total_seconds() / 3600
            s += f", resets in {h:.0f}h"
            if 0 < h <= 36 and avail >= 1:
                s += " - USE IT OR LOSE IT: queue long useful runs before the reset"
        s += "."
    if active:
        s += (f" Remote runs not yet seen finished: {', '.join(run_label(r) for r in active)} "
              "(`python -m kgkit gpu status` checks them and releases their quota reservations).")
    return s.strip()


def resets_before(q: dict, deadline: str | None) -> int:
    """Weekly resets from the quota's refreshAt (7-day steps) that land before the deadline day ends."""
    if not deadline:
        return 0
    dl = _parse_time(deadline[:10] + "T23:59:00Z")
    if dl is None:
        return 0
    t = _parse_time(q.get("refresh_at")) or _now() + dt.timedelta(days=7)
    n = 0
    while t < dl:
        n += 1
        t += dt.timedelta(days=7)
    return n


def _reservation_lines(root: Path, q: dict, resource: str) -> list[str]:
    fetched = _as_of(q)
    out = []
    for r in active_runs(root, resource):
        out.append(f"    - {run_label(r)} [{r.get('status')}, cap {float(r.get('hours') or 0):g}h"
                   + (", via kaggle CLI" if r.get("via") == "kaggle-cli" else "")
                   + f"]: ~{reserved_hours(r, fetched):.2f}h reserved")
    if out:
        out.append("    finished early or never ran? `kgkit gpu status` re-checks them; "
                   "`kgkit gpu release <kernel>` drops a reservation by hand")
    return out


def _quota_lines(root: Path, q: dict, note: str = "", label: str = "cached") -> list[str]:
    """The quota report for ``q``: its figures, where they come from (``note``, from `current_quota`, under
    ``label``), the runs logged since it was taken and the ones still in flight, the reset, and the TPU quota."""
    unseen = unseen_spend(root, q)
    spent = sum(h for _, h in unseen)
    inflight = in_flight_hours(root, q)
    left = q["remaining_h"] - spent - inflight
    lines = [f"Kaggle GPU quota: {q['used_h']:.2f}h used, {q['remaining_h']:.2f}h of {q['total_h']:.2f}h left"]
    if note:
        lines.append(f"  {label}: {note}")
    if unseen:
        lines.append("  logged since then: " + ", ".join(f"{run_label(r)} ~{h:.2f}h" for r, h in unseen)
                     + " (spent after the figures above were taken)")
    if inflight:
        lines.append(f"  in flight: up to {inflight:.2f}h more for running kernels -> {left:.2f}h plannable")
        lines += _reservation_lines(root, q, "GPU")
    elif unseen:
        lines.append(f"  -> {left:.2f}h plannable")
    refresh_at = _parse_time(q.get("refresh_at"))
    if refresh_at:
        h = (refresh_at - _now()).total_seconds() / 3600
        lines.append(f"  resets {_iso(refresh_at)} (in {h:.1f}h). Unused quota does NOT roll over.")
        if 0 < h <= 36 and left >= 1.0:
            lines.append(f"  USE IT OR LOSE IT: {left:.1f}h expire in {h:.0f}h - queue long, useful jobs now "
                         "(extra seeds / full-fold runs of the current best, final-resolution models).")
    tq = tpu_of(q)
    if tq is not None:
        tin = in_flight_hours(root, tq, "TPU")
        tleft = available_hours(root, tq, "TPU")
        lines.append(f"Kaggle TPU quota: {tq['remaining_h']:.2f}h of {tq['total_h']:.2f}h left"
                     + (f", up to {tin:.2f}h in flight" if tin else "")
                     + (f" -> {tleft:.2f}h plannable" if tleft != tq["remaining_h"] else "")
                     + " (separate quota; 9h sessions, one batch TPU session at a time, queued 10-24h)")
        if tin:
            lines += _reservation_lines(root, tq, "TPU")
    return lines


def quota_report(root: str | Path | None = None, refresh: bool = True) -> str:
    root = _root(root)
    if refresh:
        refresh_runs(root)
    q, note = current_quota(root, refresh)
    if q is None:
        return f"GPU quota unknown: {note} (is the Kaggle CLI >= 2.2 installed and authenticated?)"
    return "\n".join(_quota_lines(root, q, note, "NOT LIVE" if refresh else "cached"))


# ----------------------------------------------------------------------------- run log
def runs_path(root: str | Path) -> Path:
    return Path(root) / S.STATE_DIR / RUNS_FILE


def runs(root: str | Path) -> list[dict]:
    p = runs_path(root)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _write_runs(root: str | Path, recs: list[dict]) -> None:
    p = runs_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(x) + "\n" for x in recs), encoding="utf-8")
    tmp.replace(p)


def log_run(root: str | Path, rec: dict) -> dict:
    p = runs_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def record_push(root: str | Path, rec: dict) -> dict:
    """Log a new push. Older still-active entries of the same kernel are marked superseded: Kaggle reports
    only a kernel's latest version, and a re-push must not hold a second reservation. A version already in
    the log is the same run, logged once: its entry is returned and nothing is written."""
    recs = runs(root)
    if rec.get("version") is not None:
        for r in recs:
            if r.get("kernel") == rec.get("kernel") and r.get("version") == rec.get("version"):
                return r
    for r in recs:
        if r.get("kernel") == rec.get("kernel") and r.get("status") in ACTIVE:
            r.update(status="superseded", finished_at=r.get("finished_at") or _iso(_now()))
    recs.append(rec)
    _write_runs(root, recs)
    return rec


def update_run(root: str | Path, kernel: str, version: int | None = None, **fields) -> dict | None:
    """Update the latest run of ``kernel`` (optionally a given version); a pending push is not a run yet."""
    recs = runs(root)
    for r in reversed(recs):
        if (r.get("kernel") == kernel and r.get("status") != PENDING
                and (version is None or r.get("version") in (None, version))):
            r.update(fields)
            _write_runs(root, recs)
            return r
    return None


def note_status(root: str | Path, kernel: str, status: str) -> None:
    """Record a status read from Kaggle. A finished status closes every active entry of the kernel (Kaggle
    reports only the latest version, and earlier versions ended before it) and charges the latest run its session
    time (`session_hours`), whether or not anything is ever collected from it; a running one stamps
    ``started_at`` the first time it is seen, so a queued TPU job is not charged for its queue time. A pending
    push takes neither: the status is still an earlier version's."""
    recs = runs(root)
    mine = [r for r in recs if r.get("kernel") == kernel and r.get("status") != PENDING]
    if not mine or status not in (*FINISHED, "queued", "running"):
        return
    now = _iso(_now())
    if status in FINISHED:
        for r in mine:
            if r.get("status") in ACTIVE:
                r.update(status=status, finished_at=r.get("finished_at") or now)
                key = hours_key(run_resource(r))
                h = session_hours(r) if r is mine[-1] and r.get(key) is None else None
                if h is not None:
                    r[key] = h
    else:
        last = mine[-1]
        if last.get("status") in ACTIVE:
            last["status"] = status
            if status == "running" and not last.get("started_at"):
                last["started_at"] = now
    _write_runs(root, recs)


def release(root: str | Path, kernel: str, status: str = "released") -> int:
    """Close every active entry of ``kernel`` by hand (it finished, was deleted, or never started)."""
    recs = runs(root)
    n = 0
    for r in recs:
        if r.get("kernel") == kernel and r.get("status") in ACTIVE:
            r.update(status=status, finished_at=_iso(_now()))
            n += 1
    if n:
        _write_runs(root, recs)
    return n


def confirm_pending(root: str | Path, kernels: list[str] | None = None) -> list[dict]:
    """Pending pushes (of ``kernels``, default all) whose kernel Kaggle lists as run after the background command
    started: the push went through, so each becomes an ordinary entry timed from that run, and Kaggle's status is
    its own from then on. One listing call, made only when the log has a pending entry. Returns those confirmed."""
    recs = runs(root)
    pend = [r for r in recs if r.get("status") == PENDING and (kernels is None or r.get("kernel") in kernels)]
    if not pend:
        return []
    ran = {ref.lower(): t for ref, t in recent_kernels(20) or [] if t is not None}
    done = []
    for r in pend:
        t, since = ran.get(str(r.get("kernel")).lower()), _parse_time(r.get("pushed_at"))
        if t is not None and since is not None and t >= since - dt.timedelta(seconds=CLOCK_SKEW_S):
            r.update(status="pushed", pushed_at=_iso(t))
            done.append(r)
    if done:
        _write_runs(root, recs)
    return done


def refresh_runs(root: str | Path) -> dict[str, str]:
    """Ask Kaggle for the status of every kernel the log still has as active (one call per kernel). Pending
    pushes are looked up first (`confirm_pending`); one Kaggle has not run yet is reported as pending."""
    confirm_pending(root)
    recs = runs(root)
    kernels = list(dict.fromkeys(r["kernel"] for r in recs if r.get("status") in ACTIVE and r.get("kernel")))
    pending = {r.get("kernel") for r in recs if r.get("status") == PENDING}
    out = {}
    for k in kernels:
        if k in pending:
            out[k] = PENDING
            continue
        try:
            s, _ = kernel_status(k)
        except (OSError, subprocess.SubprocessError):
            s = "unknown"
        out[k] = s
        note_status(root, k, s)
    return out


# ----------------------------------------------------------------------------- build
def _git(root: Path) -> tuple[str | None, bool | None]:
    try:
        c = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                           timeout=5).stdout.strip()
        d = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True,
                                timeout=5).stdout.strip())
        return (c or None, d if c else None)
    except (OSError, subprocess.SubprocessError):
        return None, None


def _add_tree(z: zipfile.ZipFile, src: Path, arc: str, base: Path | None = None) -> int:
    """Add ``src`` under ``arc``. Credential-like files, and anything whose real path leaves ``base`` (through
    a link at any level, ``src`` included), never go in: the bundle is embedded in the kernel's code, which
    is uploaded to Kaggle."""
    base = (base or (src if src.is_dir() else src.parent)).resolve()
    if not safety.inside(src, base) or (src.is_file() and safety.looks_secret(src.name)):
        raise SystemExit(f"--include {src}: looks like a credential file or links outside the project; "
                         "it would be uploaded inside the kernel")
    if src.is_file():
        z.write(src, arc)
        return 1
    n, skipped = 0, []
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if any(part in BUNDLE_EXCLUDE_DIRS for part in rel.parts) or not p.is_file():
            continue
        if p.suffix.lower() in BUNDLE_EXCLUDE_SUFFIXES:
            continue
        if safety.looks_secret(p.name) or not safety.inside(p, base):
            skipped.append(f"{arc}/{rel.as_posix()}")
            continue
        z.write(p, f"{arc}/{rel.as_posix()}")
        n += 1
    if skipped:
        print("left out of the kernel bundle (credential-like files or links outside the project): "
              + ", ".join(skipped[:10]) + (f" and {len(skipped) - 10} more" if len(skipped) > 10 else ""))
    return n


def _split_rows(data: bytes):
    """The rows of a split file's bytes, blank lines skipped (kgkit's fingerprint and pandas skip them too). UTF-8 is
    decoded strictly (UnicodeDecodeError), so no cell is changed on the way."""
    return filter(None, csv.reader(io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig", newline="")))


def _batches(rows, size: int = 1 << 16):
    while True:
        batch = list(itertools.islice(rows, size))
        if not batch:
            return
        yield batch


def _cells(rows) -> bytes:
    """Rows as the split's digest takes them: cells joined by \\x1f, each row ended by \\x1e. The runner takes the same
    digest of the file it writes."""
    return "".join("\x1f".join(r) + "\x1e" for r in rows).encode("utf-8")


def _as_range(state, cells: tuple, n: int):
    """(start, step) while a column, of which ``cells`` are rows n, n+1, ..., is the integers start, start + step, ...
    written as Python writes them; False once it is not."""
    if state is None:
        try:
            a, b = int(cells[0]), int(cells[1])
        except (ValueError, IndexError):
            return False
        state = (a, b - a) if a != b else False
    if state is False:
        return False
    lo = state[0] + state[1] * n
    return state if cells == tuple(map(str, range(lo, lo + state[1] * len(cells), state[1]))) else False


def _as_codes(values: dict | None, codes: bytearray, cells: tuple) -> dict | None:
    """Add ``cells`` to a column's codes, one byte per row (``values``: cell -> code, in order of first appearance);
    None once the column has more than SPLIT_CODES_MAX distinct values."""
    if values is None:
        return None
    for v in dict.fromkeys(cells):
        values.setdefault(v, len(values))
    if len(values) > SPLIT_CODES_MAX:
        codes.clear()
        return None
    codes.extend(map(values.__getitem__, cells))
    return values


def _training_files(root: Path) -> list[str]:
    """CSV files under data/ that a split's columns can be read from in the session, where the competition data and
    the inputs are linked into data/ too: the file `kgkit folds` read (.kaggle-gm/folds_spec.json), then
    data/train.csv. Paths relative to the project root."""
    names = []
    try:
        names.append(str(json.loads((root / S.STATE_DIR / "folds_spec.json").read_text(encoding="utf-8"))["train"]))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    out = []
    for name in [*names, "data/train.csv"]:
        p = Path(name)
        if p.is_absolute():
            try:
                p = p.relative_to(root.absolute())
            except ValueError:
                continue
        rel = PurePosixPath(os.path.normpath(p).replace("\\", "/"))
        if (rel.parts[:1] == ("data",) and ".." not in rel.parts and rel.suffix.lower() == ".csv"
                and rel.as_posix() not in (SPLIT_FILE, *out) and (root / rel).is_file()):
            out.append(rel.as_posix())
    return out


def _same_columns(path: Path, data: bytes, want: dict[int, str]) -> set[int]:
    """The split's columns ``want`` (index -> name) that the CSV at ``path`` has under the same name, cell for cell and
    row for row (blank lines skipped in both)."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as f:
            theirs = filter(None, csv.reader(f))
            head = next(theirs, [])
            at = {j: head.index(name) for j, name in want.items() if name in head}
            if not at:
                return set()
            mine = _split_rows(data)
            next(mine)
            for a, b in itertools.zip_longest(mine, theirs):
                if a is None or b is None:
                    return set()
                at = {j: i for j, i in at.items() if i < len(b) and a[j] == b[i]}
                if not at:
                    return set()
    except (OSError, UnicodeDecodeError, csv.Error):
        return set()
    return set(at)


def pack_split(root: Path, data: bytes) -> tuple[dict, dict[str, bytes]] | None:
    """The fold split file's bytes as the runner writes data/folds.csv back from them: a manifest (header, rows, a
    digest of the cells, the form of each column) and the bundle members holding the columns. A column is an integer
    range (nothing to store), one byte per row (at most SPLIT_CODES_MAX distinct values: the folds), or its cells as
    text; a text column bigger than SPLIT_REF_KB is read in the session from a training file that has it, row for row
    (`_training_files`), and the runner checks it there. Every cell comes back as it was; the file is written with LF
    line endings and the quotes its cells need. None when the file is not such a table (not UTF-8, or rows of other
    widths): it then goes in as it is."""
    try:
        rows = _split_rows(data)
        header = next(rows)
        digest = hashlib.sha1(_cells([header]))
        ranges, values, codes, n = [None] * len(header), [{} for _ in header], [bytearray() for _ in header], 0
        for batch in _batches(rows):
            if any(len(r) != len(header) for r in batch):
                return None
            digest.update(_cells(batch))
            for j, cells in enumerate(zip(*batch)):
                ranges[j] = _as_range(ranges[j], cells, n)
                values[j] = _as_codes(values[j], codes[j], cells)
            n += len(batch)
    except (StopIteration, UnicodeDecodeError, csv.Error):
        return None
    text = {j: [] for j in range(len(header)) if not ranges[j] and values[j] is None}
    if text:
        rows = _split_rows(data)
        next(rows)
        for batch in _batches(rows):
            for j, parts in text.items():
                parts.append(json.dumps([r[j] for r in batch]))
        text = {j: "\n".join(parts).encode("ascii") for j, parts in text.items()}
    big = {j: header[j] for j, t in text.items() if len(zlib.compress(t, 6)) > SPLIT_REF_KB * 1024}
    refs = {}
    for rel in _training_files(root) if big else []:
        refs.update(dict.fromkeys(_same_columns(root / rel, data, {j: c for j, c in big.items() if j not in refs}), rel))
    members, columns = {}, []
    for j in range(len(header)):
        member = f"{SPLIT_ARC}/{j}"
        if ranges[j]:
            columns.append({"range": list(ranges[j])})
        elif values[j] is not None:
            members[member] = bytes(codes[j])
            columns.append({"codes": member, "values": list(values[j])})
        elif j in refs:
            columns.append({"ref": refs[j]})
        else:
            members[member] = text[j]
            columns.append({"text": member})
    return ({"file": SPLIT_FILE, "rows": n, "header": header, "columns": columns, "digest": digest.hexdigest()},
            members)


def split_summary(split: dict) -> str:
    """How a fold split reaches the session: '700000 rows; id: a range, fold: 5 values' (`pack_split`), or the
    dataset it is linked from (`--split-from`)."""
    if split.get("from"):
        return f"linked from the dataset {split['from']}"
    how = [f"{name}: " + ("a range" if "range" in c else f"{len(c['values'])} values" if "codes" in c
                          else f"read from {c['ref']}" if "ref" in c else "text")
           for name, c in zip(split["header"], split["columns"])]
    return f"{split['rows']} rows; " + ", ".join(how)


def build_bundle(root: Path, include: list[str], baseline_dir: Path | None = None,
                 extra: dict[str, bytes] | None = None) -> bytes:
    """Zip of the code a remote run needs, and ``extra`` members (the fold split, `pack_split`). Data never goes in
    here: attach it as a dataset/competition."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        _add_tree(z, KGKIT_DIR, "kgkit")
        z.write(root / S.STATE_DIR / S.STATE_FILE, f"{S.STATE_DIR}/{S.STATE_FILE}")
        for arc, blob in (extra or {}).items():
            z.writestr(arc, blob)
        for inc in include:
            p = (root / inc)
            if not p.exists():
                raise SystemExit(f"--include {inc}: not found under {root}")
            _add_tree(z, p, Path(inc).as_posix().strip("/"), base=root)
        if baseline_dir is not None:
            for c in sorted(baseline_dir.glob("curve_fold*.json")):
                z.write(c, f"_kg/baseline/{c.name}")
    return buf.getvalue()


def bundle_parts(payload: bytes) -> list[tuple[str, int]]:
    """(part, bytes it takes in the kernel script) of a code bundle, biggest first: each top-level folder or file, with
    the fold split and the baseline's learning curves as parts of their own."""
    parts: dict[str, int] = {}
    for i in zipfile.ZipFile(io.BytesIO(payload)).infolist():
        name = i.filename
        part = ("the fold split" if name.startswith(f"{SPLIT_ARC}/") or name == SPLIT_FILE
                else "the baseline's learning curves" if name.startswith("_kg/baseline/")
                else name.split("/")[0] + ("/" if "/" in name else ""))
        parts[part] = parts.get(part, 0) + i.compress_size
    return sorted(((p, b * 4 // 3) for p, b in parts.items()), key=lambda pb: -pb[1])


def _size(b: float) -> str:
    return f"{b / 1e6:.2f} MB" if b >= 1e6 else f"{b / 1e3:.0f} KB"


def _csv_header(path: Path) -> list[str]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as f:
            return next(filter(None, csv.reader(f)), [])
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


def script_size_warning(root: Path, size: int, payload: bytes, split: dict | None) -> str:
    """'' for a kernel script Kaggle takes; else what takes the space and how to make room. ``split``: the config's
    (`pack_split`'s manifest, the dataset of --split-from, or None for data/folds.csv as it is)."""
    if size < SOURCE_MAX_BYTES:
        return ""
    parts = bundle_parts(payload)
    biggest = ", ".join(f"{p} {_size(b)}" for p, b in parts[:3] if b >= 1000)
    dataset = ("put data/folds.csv in a private dataset (`kaggle datasets create`) and build with --split-from "
               "<owner>/<slug>; the run links it from there, checked against yours")
    packed = split is not None and "columns" in split
    text = [name for name, c in zip(split["header"], split["columns"]) if "text" in c] if packed else []
    if not parts or parts[0][0] != "the fold split" or (split is not None and not packed):
        how = "Keep data out of --include and attach it as a dataset (--dataset)."
    elif split is None:
        how = (f"data/folds.csv goes in as it is (it is not UTF-8, or its rows differ in width): save it as a plain "
               f"UTF-8 CSV to have it packed, or {dataset}.")
    elif text:
        held = {}  # text column -> the first training file with a column of that name
        for rel in _training_files(root):
            for name in _csv_header(root / rel):
                if name in text:
                    held.setdefault(name, rel)
        order = "; ".join(f"{name!r} is not the column of {rel} row for row: written in {rel}'s row order with its "
                          "values, it would be read from there in the session instead" for name, rel in held.items())
        how = (f"Its column(s) {', '.join(map(repr, text))} go in as text" + (f" ({order})" if order else "")
               + f". Or {dataset}.")
    else:
        how = f"It is too big even packed: {dataset}."
    return (f"WARNING: the kernel script is {_size(size)}, and Kaggle refuses a kernel source of 1 MB or more with a "
            f"bare 400 Bad Request (`gpu push` stops before sending it). Its biggest parts: {biggest}. {how}")


def kernel_output_dir(kernel: str) -> str:
    """Where a kernel's output files go, relative to the project root: the folder ``gpu build --input-layout nested``
    mounts it in, and the one `collect` fills for a kernel that `gpu build` did not make."""
    return f"data/ext/kout-{kernel.split('/')[-1]}"


def input_sources(datasets: list[str], kernels: list[str], layout: str = "flat") -> list[dict]:
    """``--dataset`` / ``--kernel`` arguments as the runner's sources: {ref, kind, dest}, ``dest`` being the
    folder (relative to the project root, so the same locally) that the input's files appear in. An argument is
    ``owner/slug``, placed by ``layout``, or ``owner/slug=<folder>`` with the folder ``data`` or one under it."""
    if layout not in INPUT_LAYOUTS:
        raise SystemExit(f"--input-layout {layout}: one of {', '.join(INPUT_LAYOUTS)}")
    out = []
    for kind, args in (("dataset", datasets), ("kernel", kernels)):
        for arg in dict.fromkeys(args):
            ref, given, dest = (x.strip() for x in arg.partition("="))
            if not re.fullmatch(r"[^/\s=]+/[^/\s=]+", ref):
                raise SystemExit(f"--{kind} {arg}: expected <owner>/<slug>, or <owner>/<slug>=<folder under data/>")
            if given:
                p = PurePosixPath(dest.replace("\\", "/"))
                if p.is_absolute() or ".." in p.parts or p.parts[:1] != ("data",):
                    raise SystemExit(f"--{kind} {arg}: the folder must be data or one under it, relative to the "
                                     f"project root (e.g. {ref}=data/ext/{ref.split('/')[1]})")
                dest = p.as_posix()
            elif layout == "nested":
                dest = kernel_output_dir(ref) if kind == "kernel" else f"data/ext/{ref.split('/')[1]}"
            else:
                dest = "data"
            out.append({"ref": ref, "kind": kind, "dest": dest, "auto": not given})
    for i, s in enumerate(out):  # two inputs share a folder only when both were given it
        for o in out[i + 1:]:
            same_input = (o["kind"], o["ref"]) == (s["kind"], s["ref"])
            if o["dest"] == s["dest"] != "data" and not same_input and (s["auto"] or o["auto"]):
                m = o if o["auto"] else s
                raise SystemExit(f"{s['ref']} and {o['ref']} would share the folder {s['dest']}/: give {m['ref']} "
                                 f"its own, e.g. --{m['kind']} {m['ref']}={s['dest']}-{m['ref'].split('/')[0]}")
    for s in out:
        del s["auto"]
    return out


def _accepted_or_best(cands: list[dict], gib: bool) -> dict | None:
    """The accepted baseline (latest record decided keep/baseline), else the highest CV — the same rule
    as ``kgkit.experiment.Ledger.baseline`` (duplicated here: this module stays standard-library only)."""
    cands = [r for r in cands if not r.get("recheck")]  # re-drawn-split runs are not comparable references
    for r in reversed(cands):
        if r.get("decision") in ("baseline", "keep"):
            return r
    return sorted(cands, key=lambda r: r["cv"], reverse=gib)[0] if cands else None


def _resolve_baseline(root: Path, ref: str) -> tuple[Path, dict | None]:
    """``ref``: 'baseline' (the accepted baseline; 'best' is an alias), a ledger id/name, or an
    artifacts/<name> directory with curve_fold*.json."""
    p = Path(ref)
    if (root / p).is_dir():
        return root / p, None
    recs = []
    lp = root / S.STATE_DIR / "ledger.jsonl"
    if lp.is_file():
        recs = [json.loads(l) for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
    st = S.load(root)
    gib = st.greater_is_better if st else True
    if ref in ("best", "baseline"):
        cands = [r for r in recs if isinstance(r.get("cv"), (int, float)) and r.get("model") != "blend"
                 and (root / "artifacts" / r.get("name", "")).is_dir()]
        rec = _accepted_or_best(cands, gib)
    else:
        rec = next((r for r in reversed(recs) if r.get("id") == ref or r.get("name") == ref), None)
    if rec is None:
        raise SystemExit(f"--prune-against {ref}: no such ledger experiment with fold artefacts")
    d = root / "artifacts" / rec["name"]
    if not list(d.glob("curve_fold*.json")):
        raise SystemExit(f"--prune-against {ref}: {d} has no curve_fold*.json (re-run it once with the "
                         "current templates, which record learning curves)")
    return d, rec


def build_kernel(root: str | Path | None, name: str, cmd: str, hours: float, folds: list[int] | None = None,
                 accelerator: str = "NvidiaTeslaT4", user: str | None = None, datasets: list[str] = (),
                 kernels: list[str] = (), models: list[str] = (), resume_from: list[str] = (),
                 include: list[str] = ("src",), internet: bool = True, pip: list[str] = (),
                 wheel_sources: list[str] = (), assemble_cmd: str | None = None, post_cmds: list[str] = (),
                 prune_against: str | None = None, prune_margin: float | None = None,
                 prune_min_frac: float = 0.3, slug: str | None = None, out_dir: str | Path | None = None,
                 public: bool = False, extra_jobs: list[tuple[str, str]] = (), input_layout: str = "flat",
                 split_from: str | None = None) -> Path:
    """Write ``kernels/<slug>/`` (kernel-metadata.json + generated runner script + kg-train.json).

    ``extra_jobs`` ((name, cmd) pairs) share the session: e.g. two 1-fold screens of different ideas
    run side by side on the two T4s instead of leaving one GPU idle. ``datasets`` / ``kernels`` are linked
    into the run as :func:`input_sources` places them. data/folds.csv goes in the code bundle as :func:`pack_split`
    packs it, or, with ``split_from`` (a dataset holding a copy of it, for a split too big for a kernel script), is
    linked from that dataset; either way the runner checks it against yours before anything runs."""
    root = _root(root)
    st = S.load(root)
    if st is None:
        raise SystemExit("not inside a competition workspace (no .kaggle-gm/competition.json)")
    try:  # before any work, not after the bundle is built
        runner = RUNNER.read_text(encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"this kgkit ({KGKIT_DIR}) has no kernel runner ({e}): copy it again from the plugin's, "
                         "`PYTHONPATH=\"$KGKIT_HOME\" python -m kgkit vendor <dest>`")
    sources = input_sources(list(datasets), list(kernels), input_layout)
    if accelerator not in ACCELERATORS and accelerator != "none":
        problem = machine_shape_problem(accelerator)
        raise SystemExit(f"--accelerator: {problem}" if problem else
                         f"--accelerator {accelerator}: `gpu build` targets {', '.join(ACCELERATORS)} or none (a "
                         "kernel of your own can ask for other shapes: `gpu push <dir> --hours H`)")
    res = resource_of(accelerator)
    if not 0 < hours <= session_limit(res):
        raise SystemExit(f"--hours must be in (0, {session_limit(res):g}] (Kaggle {res} session limit)")
    user = user or kaggle_username()
    if not user:
        raise SystemExit("cannot determine your Kaggle username (KAGGLE_USERNAME, kaggle.json and "
                         "`kaggle config view` gave none): pass --user")
    fp = root / SPLIT_FILE
    all_folds, split_data = None, fp.read_bytes() if fp.is_file() else None
    if split_data is not None:
        try:
            all_folds = E.fold_ids(split_data)
        except ValueError as e:
            raise SystemExit(f"data/folds.csv: {e}")
    if split_from is not None and not re.fullmatch(r"[^/\s=]+/[^/\s=]+", split_from):
        raise SystemExit(f"--split-from {split_from}: expected <owner>/<slug> of a dataset holding data/folds.csv")
    if split_from is not None and split_data is None:
        raise SystemExit("--split-from: there is no data/folds.csv here to check the dataset's copy against")
    if folds is None:
        if "{fold}" in cmd and all_folds is None:
            raise SystemExit("--folds not given and data/folds.csv missing")
        folds = all_folds if "{fold}" in cmd else []
    screen = all_folds is not None and set(folds) < set(all_folds)  # a subset of folds cannot be assembled
    jobs = []
    for i, (jname, jcmd) in enumerate([(name, cmd), *extra_jobs]):
        if len(extra_jobs) and "{fold}" not in jcmd:
            raise SystemExit(f"job {jname}: with several jobs every command needs a {{fold}} placeholder")
        asm = assemble_cmd if i == 0 else None
        if asm is None and "{fold}" in jcmd and not screen:
            m = re.search(r"--folds\s+\{fold\}", jcmd)
            asm = jcmd[:m.start()] + "--assemble" + jcmd[m.end():] if m else None
        jobs.append({"name": jname, "cmd": jcmd, "assemble_cmd": asm, "folds": folds if "{fold}" in jcmd else []})
    if len({j["name"] for j in jobs}) != len(jobs):
        raise SystemExit("job names must be unique (they are the artifacts/<name>/ folders)")
    baseline_dir, prune = None, None
    if prune_against:
        baseline_dir, rec = _resolve_baseline(root, prune_against)
        if prune_margin is None:
            std = (rec or {}).get("cv_std")
            prune_margin = float(std) if std else None
        if prune_margin is None:
            raise SystemExit("--prune-margin required (baseline has no cv_std in the ledger)")
        prune = {"baseline": prune_against, "margin": prune_margin, "min_frac": prune_min_frac}

    kslug = _slug(slug or f"{name}-train")
    if len(kslug) < 5:
        kslug = _slug(f"{kslug}-train")
    resume_src = list(resume_from)
    commit, dirty = _git(root)
    built = _iso(_now())
    split, split_members = None, {}
    if split_from:
        split = {"file": SPLIT_FILE, "from": split_from, "bytes": len(split_data),
                 "sha1": hashlib.sha1(split_data).hexdigest()}
    elif split_data is not None:  # the frozen split must be identical remotely (folds_hash matches the local ledger)
        split, split_members = pack_split(root, split_data) or (None, {SPLIT_FILE: split_data})
    config = {
        "jobs": jobs, "post_cmds": list(post_cmds),
        "hours": hours, "accelerator": accelerator, "pip": list(pip), "wheel_sources": list(wheel_sources),
        "sources": [{"ref": st.slug, "kind": "competition", "dest": "data"}, *sources], "input_layout": input_layout,
        "resume_sources": resume_src, "prune": prune,
        "git_commit": commit, "git_dirty": dirty, "build_id": f"{kslug}@{built}",
        "allow_cpu": accelerator == "none", "resource": res, "tpu_chips": TPU_CHIPS.get(accelerator, 0),
        "split": split,
    }
    payload = build_bundle(root, list(include), baseline_dir, split_members)
    script = runner.replace("__KG_CONFIG_B64__", base64.b64encode(json.dumps(config).encode()).decode(), 1)
    script = script.replace("__KG_PAYLOAD_B64__", base64.b64encode(payload).decode(), 1)
    size = len(script.encode("utf-8"))

    kdir = Path(out_dir) if out_dir else root / "kernels" / kslug
    kdir.mkdir(parents=True, exist_ok=True)
    code_file = f"{kslug.replace('-', '_')}.py"
    (kdir / code_file).write_text(script, encoding="utf-8")
    meta = {
        "id": f"{user}/{kslug}", "title": kslug.replace("-", " "), "code_file": code_file, "language": "python",
        "kernel_type": "script", "is_private": not public, "enable_gpu": res == "GPU",
        "enable_tpu": res == "TPU", "enable_internet": internet,
        "dataset_sources": [*dict.fromkeys([*(s["ref"] for s in sources if s["kind"] == "dataset"), *wheel_sources,
                                            *([split_from] if split_from else [])])],
        "competition_sources": [st.slug],
        "kernel_sources": [*dict.fromkeys([*(s["ref"] for s in sources if s["kind"] == "kernel"), *resume_src])],
        "model_sources": list(models),
    }
    if accelerator != "none":
        meta["machine_shape"] = accelerator
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    train_meta = {"kernel": meta["id"], "name": name, "jobs": [j["name"] for j in jobs], "hours": hours,
                  "accelerator": accelerator, "resource": res, "folds": folds, "cmd": cmd, "build_id": config["build_id"], "prune": prune,
                  "resume_from": resume_src, "bundle_kb": round(len(payload) / 1024, 1),
                  "script_kb": round(size / 1024, 1),
                  "split": split_summary(split) if split else None if split_data is None else "in the bundle as it is",
                  "input_layout": input_layout, "inputs": config["sources"]}
    (kdir / "kg-train.json").write_text(json.dumps(train_meta, indent=2), encoding="utf-8")
    warning = script_size_warning(root, size, payload, split)
    if warning:
        print(warning)
    packed = split is not None and "columns" in split
    refs = [f"{n!r} from {c['ref']}" for n, c in zip(split["header"], split["columns"]) if "ref" in c] if packed else []
    if refs:
        print(f"note: the run reads data/folds.csv's column(s) {', '.join(refs)} in the session (too big to pack "
              "as text), and stops before training if they differ there from yours")
    flat = [s["ref"] for s in sources if s["dest"] == "data"]
    if len(flat) > 1:
        names = ", ".join(flat[:4]) + (f" and {len(flat) - 4} more" if len(flat) > 4 else "")
        print(f"note: {len(flat)} inputs are linked side by side into data/ ({names}): a name two of them share is "
              "linked from the first only (the run log lists each). --input-layout nested gives each its own folder "
              "(data/ext/<slug>/, kernel outputs data/ext/kout-<slug>/)")
    return kdir


# ----------------------------------------------------------------------------- push / status
STATUS_WORDS = (("cancel", "cancelled"), ("error", "error"), ("complete", "complete"), ("running", "running"),
                ("queued", "queued"), ("new_script", "queued"))


def normalize_status(text: str) -> str:
    low = text.lower()
    m = re.search(r'has status "([^"]+)"', low)
    s = m.group(1) if m else low
    for key, val in STATUS_WORDS:
        if key in s:
            return val
    return "unknown"


def _truthy(v) -> bool:
    return v is True or str(v).lower() == "true"


def kernel_meta_accelerator(meta: dict) -> str:
    """The accelerator a kernel-metadata.json asks for: its machine_shape (Kaggle's API marks enable_gpu and
    enable_tpu deprecated in its favour), also without either flag; else the default shape of enable_tpu or
    enable_gpu; else 'none'. Under enable_tpu, a shape that is not a TPU's gives the TPU default."""
    shape = str(meta.get("machine_shape") or "").strip()
    if shape.lower() in ("none", "cpu", "no"):
        shape = ""
    if _truthy(meta.get("enable_tpu")):
        return shape if shape.lower().startswith("tpu") else "TpuV5E8"
    if shape or _truthy(meta.get("enable_gpu")):
        return shape or "NvidiaTeslaT4"
    return "none"


def push_spec(kdir: Path, hours: float | None = None) -> dict:
    """What a push needs: from kg-train.json (written by `gpu build`), else from kernel-metadata.json for
    any other kernel folder (a notebook, an inference or diagnostics kernel), which needs ``hours``."""
    tp, mp = kdir / "kg-train.json", kdir / "kernel-metadata.json"
    if tp.is_file():
        tm = json.loads(tp.read_text(encoding="utf-8"))
        if hours is not None:
            tm["hours"] = hours
        return tm
    if not mp.is_file():
        raise SystemExit(f"{kdir}: no kernel-metadata.json (not a kernel folder)")
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
    except ValueError as e:
        raise SystemExit(f"{mp}: invalid JSON ({e})")
    if not meta.get("id"):
        raise SystemExit(f"{mp}: no \"id\": \"<username>/<slug>\" (your username: `python -m kgkit whoami`)")
    if hours is None:
        raise SystemExit(f"{kdir} was not built by `kgkit gpu build` (no kg-train.json): pass --hours H, the "
                         "session cap, so the quota check and the -t timeout use it")
    slug = str(meta["id"]).split("/")[-1]
    return {"kernel": meta["id"], "name": slug, "jobs": ["notebook"], "hours": hours,
            "accelerator": kernel_meta_accelerator(meta), "folds": None, "cmd": None}


def pending_sources(kdir: Path) -> list[tuple[str, str]]:
    """Attached kernel outputs that have no finished version yet. Kaggle binds a kernel's inputs when it is
    pushed, not when it starts: a job queued behind a run that is still going starts without its output."""
    try:
        meta = json.loads((kdir / "kernel-metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for src in meta.get("kernel_sources") or []:
        try:
            s, _ = kernel_status(src)
        except (OSError, subprocess.SubprocessError):
            continue
        if s in ("queued", "running"):
            out.append((src, s))
    return out


def recent_kernels(max_kernels: int = 8) -> list[tuple[str, dt.datetime | None]] | None:
    """(ref, lastRunTime) of the account's own ``max_kernels`` most recently run kernels, latest first;
    lastRunTime is when Kaggle ran a kernel's latest version (UTC). None when the listing fails."""
    try:
        r = run_kaggle(["kernels", "list", "-m", "--sort-by", "dateRun", "--page-size", str(max_kernels),
                        "--format", "json"], timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    text = r.stdout or ""
    try:
        rows = json.loads(text[text.find("["):]) if r.returncode == 0 and "[" in text else None
    except ValueError:
        rows = None
    if not isinstance(rows, list):
        return None
    return [(row["ref"], _parse_time(row.get("lastRunTime"))) for row in rows
            if isinstance(row, dict) and row.get("ref")]


def busy_sessions(max_kernels: int = 8, within_h: float = 48.0) -> list[tuple[str, str]] | None:
    """(kernel, status) of the account's own kernels queued or running now, in any competition: the
    most recently run ``max_kernels`` are checked. None when the listing fails."""
    recent = recent_kernels(max_kernels)
    if recent is None:
        return None
    out = []
    for ref, t in recent:
        if t is None or (_now() - t).total_seconds() / 3600 > within_h:
            continue
        try:
            s, _ = kernel_status(ref)
        except (OSError, subprocess.SubprocessError):
            continue
        if s in ("queued", "running"):
            out.append((ref, s))
    return out


def _session_preflight(root: Path, resource: str, force: bool) -> None:
    busy = busy_sessions()
    if not busy:
        return
    known = {r.get("kernel"): run_resource(r) for r in runs(root)}
    same = [k for k, _ in busy if known.get(k) == resource]
    slots = SESSION_SLOTS.get(resource, 2)
    print(f"busy sessions on your account: " + ", ".join(f"{k} ({s})" for k, s in busy))
    if len(same) >= slots and not force:
        raise SystemExit(f"refusing to push: Kaggle allows {slots} batch {resource} session(s) at a time and "
                         f"{', '.join(same)} hold them. Wait for one to finish (`kgkit gpu wait <kernel>`), "
                         "or pass --force.")
    if len(busy) >= slots:
        print(f"note: Kaggle allows {slots} batch {resource} session(s) at a time; if the kernels above use "
              f"{resource}s the push fails with 'Maximum batch {resource} session count'")


# Kaggle's answer to a push. Refused: the Kaggle CLI prints requests' error for the SaveKernel call. Accepted without
# some inputs: Kaggle CLI 2.2.4 prints "The following are not valid dataset sources and could not be added to the
# kernel: [...]" before that push's "Kernel version N successfully pushed", and the run starts without them.
REFUSED_RE = re.compile(r"\b4\d\d client error\b[^\n]*savekernel", re.IGNORECASE)
DROPPED_RE = re.compile(r"the following are not valid (\w+) sources and could not be added to the kernel: *([^\n]*)",
                        re.IGNORECASE)
PUSHED_URL_RE = re.compile(r"successfully pushed[^\n]*?/code/([\w.-]+/[\w.-]+)", re.IGNORECASE)


def kernel_source(kdir: str | Path) -> tuple[str, int] | None:
    """(name, bytes) of the source file of the kernel in ``kdir`` (kernel-metadata.json's code_file); None when it
    cannot be told."""
    try:
        name = str(json.loads((Path(kdir) / "kernel-metadata.json").read_text(encoding="utf-8"))["code_file"])
        return name, (Path(kdir) / name).stat().st_size
    except (OSError, ValueError, KeyError, TypeError):
        return None


def source_problem(kdir: str | Path | None) -> str | None:
    """Why Kaggle would refuse the source of the kernel in ``kdir``: 1 MB or more. None when it would not, or when the
    size cannot be told."""
    src = kernel_source(kdir) if kdir is not None else None
    if src is None or src[1] < SOURCE_MAX_BYTES:
        return None
    return (f"its source {src[0]} is {_size(src[1])}, and Kaggle refuses a kernel source of 1 MB or more with a bare "
            "400 Bad Request")


def refusal_causes(kdir: str | Path | None, shape: str | None) -> list[str]:
    """What Kaggle is known to answer with a bare 400 in a push of the kernel in ``kdir`` asking for machine_shape
    ``shape``: a source of 1 MB or more, a shape it does not take."""
    return [p for p in (source_problem(kdir), machine_shape_problem(shape)) if p]


def push_failure_hint(output: str, shape: str | None = None, kdir: str | Path | None = None) -> str:
    """What the output of a failed kernel push means and what to change; '' when kgkit has nothing to add.
    ``shape``: the machine_shape the push asked for; ``kdir``: the kernel folder pushed."""
    if "session count" in output.lower():
        return ("Kaggle allows 2 batch GPU sessions and 1 batch TPU session at a time: wait for a running kernel to "
                "finish (`kgkit gpu status`), or free a queued one (`kgkit gpu cancel <kernel>`).")
    if not REFUSED_RE.search(output):
        return ""
    causes = refusal_causes(kdir, shape)
    if causes:
        return f"Kaggle refused the push: {'; '.join(causes)}."
    if KC.REASON in output:
        return "Kaggle refused the push; its reason is above."
    src = kernel_source(kdir) if kdir is not None else None
    fine = ([f"the source {src[0]} is {_size(src[1])}, under 1 MB"] if src else []) + [
        f"machine_shape {shape or 'none'} is one Kaggle accepts"]
    return ("Kaggle refused the push, and the Kaggle CLI does not print why. The causes kgkit can check are ruled out ("
            + "; ".join(fine) + "). Check each source in kernel-metadata.json next: a competition, dataset, kernel or "
            "model that does not exist, or that this account cannot read. Kaggle's reason is in the response body "
            "(skill kaggle-cli, 'A refused request').")


def dropped_sources(output: str) -> dict[str, list[str]]:
    """Inputs Kaggle left out of the pushes it accepted, by the kernel each push reports ('' when none is named)."""
    out: dict[str, list[str]] = {}
    for m in DROPPED_RE.finditer(output):
        nxt = PUSHED_URL_RE.search(output, m.end())
        out.setdefault(nxt.group(1) if nxt else "", []).append(f"{m.group(1).lower()} {m.group(2).strip()}")
    return out


def dropped_sources_warning(output: str) -> str:
    """'' unless Kaggle accepted a push but left inputs out of it: then what that means and what to do."""
    found = dropped_sources(output)
    if not found:
        return ""
    what = "; ".join((f"{k}: " if k else "") + ", ".join(v) for k, v in found.items())
    kernel = next((k for k in found if k), "<kernel>")
    return (f"Kaggle accepted the push but left out inputs it could not attach ({what}), so the run starts without "
            "them. Check each slug in kernel-metadata.json and that this account can read it, push again, and stop "
            f"this version (`python -m kgkit gpu cancel {kernel}` says how).")


def push(root: str | Path | None, kdir: str | Path, force: bool = False, hours: float | None = None,
         check_sources: bool = True, check_sessions: bool = True) -> dict:
    root = _root(root)
    kdir = Path(kdir)
    tm = push_spec(kdir, hours)
    hours, acc = float(tm["hours"]), tm["accelerator"]
    res = resource_of(acc)
    problem = machine_shape_problem(acc)  # before any Kaggle call: Kaggle would answer a bare 400
    if problem and (acc in TPU_REWRITTEN or not force):
        spec = "kg-train.json" if (kdir / "kg-train.json").is_file() else "kernel-metadata.json"
        raise SystemExit(f"refusing to push {kdir} ({spec}): {problem}"
                         + ("" if acc in TPU_REWRITTEN else ". If Kaggle has added that shape, --force pushes it."))
    if problem:
        print(f"warning: {problem}; pushing it anyway (--force)")
    big = source_problem(kdir)  # before any Kaggle call too
    if big and not force:
        built = (kdir / "kg-train.json").is_file()
        raise SystemExit(f"refusing to push {kdir}: {big}. "
                         + ("Rebuild it: `gpu build` prints its biggest parts. Data belongs in a dataset (--dataset), "
                            "not in --include; a fold split too big to pack, in --split-from. " if built else
                            "Move data, weights and images out of the source into a dataset, and read them from "
                            "/kaggle/input. ")
                         + "If Kaggle takes bigger sources now, --force pushes it.")
    if big:
        print(f"warning: {big}; pushing it anyway (--force)")
    limit = session_limit(res)
    if not 0 < hours <= limit:
        raise SystemExit(f"--hours must be in (0, {limit:g}] (Kaggle {res} session limit)")
    if check_sources:
        pend = pending_sources(kdir)
        if pend and not force:
            raise SystemExit("refusing to push: inputs are bound when a kernel is pushed, not when it starts, and "
                             + ", ".join(f"{k} is still {s}" for k, s in pend) + ". The run would start without "
                             "their new output (or with an older version). Wait for them (`kgkit gpu wait <kernel>`), "
                             "or pass --force.")
    if res != "CPU":
        if check_sessions:
            _session_preflight(root, res, force)
        refresh_runs(root)
        q, note = current_quota(root, True, res)
        avail = available_hours(root, q, res) if q else None
        if avail is None:
            print(f"warning: {res} quota unknown - {note}; pushing without a budget check")
        elif avail < hours and not force:
            raise SystemExit(f"refusing to push: run is capped at {hours:.2f}h but only {avail:.2f}h of {res} quota "
                             f"is plannable before {q.get('refresh_at')}"
                             + (f" (a cached quota: {note})" if note else "")
                             + ". Lower --hours, wait for the reset, check the reservations (`kgkit gpu quota`), "
                             "or pass --force.")
        else:
            if note:
                print(f"warning: {res} budget checked against a cached quota, not a live one: {note}")
            print(f"{res} budget: run capped at {hours:.2f}h; {avail:.2f}h plannable -> "
                  f"{avail - hours:.2f}h left in the worst case")
    prev = latest_run(root, tm["kernel"])
    if prev.get("stopped_by") == "cap" and hours <= float(prev.get("hours") or 0):
        print(f"warning: the last run of {tm['kernel']}" + (f" (v{prev['version']})" if prev.get("version") else "")
              + f" was stopped by its {float(prev['hours']):g}h cap and this push caps at {hours:g}h: unless the "
              "script got faster, it stops at the same point again. Raise --hours, or have the kernel save its "
              "results earlier.")
    args = ["kernels", "push", "-p", str(kdir), "-t", str(int(hours * 3600))]
    if acc != "none":
        args += ["--accelerator", acc]
    r = run_kaggle(args, timeout=900)
    out = (r.stdout or "") + (r.stderr or "")
    print(out.strip())
    m = re.search(r"version\s+(\d+)\s+successfully pushed", out, re.IGNORECASE)
    ok = r.returncode == 0 and m is not None
    if not ok:
        hint = push_failure_hint(out, acc, kdir)
        raise SystemExit("kernel push failed (see output above); nothing was started." + (f" {hint}" if hint else ""))
    rec = {"kernel": tm["kernel"], "version": int(m.group(1)), "name": tm["name"],
           "jobs": tm.get("jobs") or [tm["name"]], "accelerator": acc, "resource": res, "hours": hours,
           "folds": tm.get("folds"), "cmd": tm.get("cmd"), "build_id": tm.get("build_id"), "prune": tm.get("prune"),
           "resume_from": tm.get("resume_from"), "pushed_at": _iso(_now()), "status": "pushed"}
    print(cap_note(hours, res, bool(tm.get("cmd"))))
    dropped = [s for v in dropped_sources(out).values() for s in v]
    if dropped:
        rec["dropped_sources"] = dropped
        print(f"WARNING: {dropped_sources_warning(out)}")
    record_push(root, rec)
    return rec


def kernel_status(kernel: str) -> tuple[str, str]:
    r = run_kaggle(["kernels", "status", kernel], timeout=120)
    text = (r.stdout or "") + (r.stderr or "")
    return (normalize_status(text) if r.returncode == 0 else "unknown"), text.strip()


def cancel_advice(kernel: str, status: str) -> str:
    url = f"https://www.kaggle.com/code/{kernel}"
    if status == "queued":
        return (f"{kernel} is queued. A queued session has no session id yet, so the API cannot cancel it; the "
                f"only way to free its slot is to delete the kernel (irreversible: every version and output goes):\n"
                f"  kaggle kernels delete {kernel} -y\n"
                "Ask the user before running it, then push again (the queue starts from zero). Afterwards: "
                f"`kgkit gpu release {kernel}`.")
    if status == "running":
        return (f"{kernel} is running. Stop it from the website ({url}, Edit -> session controls -> Stop "
                f"session) or let its -t cap end it; then `kgkit gpu release {kernel}` if it was not collected.")
    return f"{kernel}: {status}; nothing to cancel." + (f" `kgkit gpu release {kernel}` clears a stale reservation."
                                                         if status in FINISHED else "")


def latest_run(root: str | Path, kernel: str) -> dict:
    """The log entry of ``kernel``'s latest run ({} when there is none); a pending push is not a run yet."""
    return next((r for r in reversed(runs(root)) if r.get("kernel") == kernel and r.get("status") != PENDING), {})


def cap_note(hours: float, resource: str, built: bool) -> str:
    """What a push's ``--hours`` (the ``-t`` of the push) does: it is a hard kill, not only a quota reservation."""
    limit = session_limit(resource)
    note = (f"note: the cap of {hours:g}h is a hard stop, not only a quota reservation: Kaggle kills the session at "
            f"{hours:g}h wherever the script is (status 'cancelled') and keeps only the files already in "
            "/kaggle/working. A run that ends early is charged only the hours it ran, so leave a margin: about "
            f"1.3-1.5x the measured runtime (the session limit is {limit:g}h).")
    if built:
        return note + (" The runner stops its jobs a few minutes before the cap, with their checkpoints saved "
                       "(`gpu build ... --resume-from <kernel>` continues them).")
    return note + (" A kernel of your own has no deadline of its own: save the model before a long inference or "
                   "scoring phase, so a stop costs that phase only.")


def log_entries(text: str) -> list[dict]:
    """The entries ``{stream_name, time, data}`` of a session log as ``kaggle kernels logs`` prints it (a JSON array,
    ``time`` being seconds since the session started); [] when ``text`` is not one. A log cut off mid-way is read line
    by line."""
    text = text or ""
    try:
        rows = json.loads(text[text.find("["):]) if "[" in text else None
    except ValueError:
        rows = []
        for line in text.splitlines():
            try:
                rows.append(json.loads(line.strip().lstrip("[,").rstrip(",]")))
            except ValueError:
                continue
    return [r for r in rows if isinstance(r, dict) and "data" in r] if isinstance(rows, list) else []


def log_tail(entries: list[dict], n: int = 60) -> list[str]:
    """The last ``n`` lines of the text a session log's entries hold."""
    return "".join(str(e.get("data") or "") for e in entries).splitlines()[-n:]


def session_output(kernel: str, timeout: float = OUTPUT_TIMEOUT) -> tuple[list[str] | None, bool, str, str | None]:
    """(file names, more files than listed, log text, None) of the last session of ``kernel``, without downloading
    anything (`kgkit.kaggle_outputs`, in a child process); (None, False, '', why) when Kaggle could not be asked.
    Never raises."""
    try:
        r = subprocess.run([kaggle_python() or sys.executable, "-m", "kgkit.kaggle_outputs"],
                           input=json.dumps({"kernel": kernel, "max_pages": 5}), capture_output=True,
                           encoding="utf-8", errors="replace", timeout=timeout, cwd=str(KGKIT_DIR.parent),
                           env=safety.kaggle_env())
    except subprocess.TimeoutExpired:
        return None, False, "", "the request to Kaggle timed out"
    except OSError as e:
        return None, False, "", f"the request to Kaggle did not start: {e}"
    try:
        res = json.loads((r.stdout or "").strip().splitlines()[-1])
        error, files = res.get("error"), [str(f) for f in res.get("files") or []]
        more, log = bool(res.get("more")), str(res.get("log") or "")
    except (ValueError, IndexError, TypeError, AttributeError):
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        return None, False, "", ("the request to Kaggle failed" + (f": {safety.clean(tail[-1])[:300]}" if tail
                                                                    else f" (exit {r.returncode})"))
    if error:
        return None, False, "", safety.clean(str(error))[:300]
    return files, more, log, None


def stopped_by_cap(run: dict, status: str, ran_s: float | None) -> bool:
    """Whether a run that ended as ``status`` after at least ``ran_s`` seconds was killed by its cap."""
    cap_s = float(run.get("hours") or 0) * 3600
    return bool(status == "cancelled" and cap_s and ran_s is not None
                and ran_s >= cap_s - min(CAP_SLACK_S, CAP_SLACK_FRAC * cap_s))


def stop_lines(kernel: str, run: dict, status: str, ran_s: float | None, files: list[str] | None, more: bool,
               why: str | None) -> list[str]:
    """What to tell about a run that ended as ``cancelled`` or ``error``: whether its cap stopped it, the files it saved
    before the end (Kaggle keeps them) and how to get them, and what to do next. ``ran_s``: how long it is known to
    have run, at least; ``files`` None when they could not be listed (``why``)."""
    cap_h = float(run.get("hours") or 0)
    ran = f"{ran_s / 3600:.2f}h" if ran_s is not None else ""
    built = bool(run.get("cmd"))
    capped = stopped_by_cap(run, status, ran_s)
    out = []
    if capped:
        what = "-t cap" if run.get("via") == "kaggle-cli" else "--hours cap"
        out.append(f"stopped by its {what} ({cap_h:g}h): it ran at least {ran}. The cap is a hard kill - Kaggle ends "
                   "the session there, wherever the script is - not only a quota reservation.")
    elif ran and cap_h:
        out.append(f"ran at least {ran} of its {cap_h:g}h cap; the cap does not look like the cause.")
    get = f"`kaggle kernels output {kernel} -p <dir>` (--file-pattern <regex> picks files)"
    if files is None:
        out.append(f"could not list the files it saved ({why}); {get} downloads them.")
    elif files:
        names = (", ".join(files[:8]) + (f" and {len(files) - 8} more" if len(files) > 8 else "")
                 + (" (list cut off)" if more else ""))
        out.append(f"files it saved before the stop, still downloadable: {names}. Get them with {get} or "
                   f"`python -m kgkit gpu collect {kernel}`"
                   + ("." if built else f" (into {kernel_output_dir(kernel)}/, with the hours it used)."))
    else:
        out.append("it saved no files to /kaggle/working before the stop, so nothing is downloadable.")
    if built:
        out.append(f"to continue: `gpu build ... --resume-from {kernel}` (finished folds are skipped, timed-out ones "
                   "continue from their checkpoint).")
    elif capped:
        out.append("to run it again: a cap of about 1.3-1.5x the time it needs (a run that ends early is charged only "
                   "the hours it ran, though the quota check reserves the whole cap while it runs), and have the "
                   "kernel save the model, then the predictions, as soon as each exists, so a stop costs the last "
                   "phase only.")
    return out


def report_stop(root: str | Path, kernel: str, status: str, text: str = "", watched_s: float | None = None,
                tail: bool = True) -> None:
    """Print what a run that ended as ``cancelled`` or ``error`` left (`stop_lines`): the status text, the log's
    tail (``tail``), whether its cap stopped it (recorded on the run as ``stopped_by``) and the files it saved.
    ``watched_s``: how long ``wait`` saw it run, which bounds a log that stops well before the end."""
    if text:
        print(text)
    files, more, log, why = session_output(kernel)
    if not log:
        try:
            log = run_kaggle(["kernels", "logs", kernel], timeout=300).stdout or ""
        except (OSError, subprocess.SubprocessError):
            pass
    entries = log_entries(log)
    if tail:
        lines = log_tail(entries) if entries else log.splitlines()[-60:]
        if lines:
            print("\n".join(lines))
    times = [e["time"] for e in entries if isinstance(e.get("time"), (int, float))]
    known = [t for t in (max(times, default=None), watched_s) if t is not None]
    ran_s = max(known) if known else None
    run = latest_run(root, kernel)
    fields = {"stopped_by": "cap"} if stopped_by_cap(run, status, ran_s) else {}
    key, h = hours_key(run_resource(run)), session_hours(run, ran_s)
    if run and h is not None and h > (run.get(key) or 0):  # the log can show a run lasted longer than it was watched
        fields[key] = h
    if fields:
        update_run(root, kernel, run.get("version"), **fields)
    lines = stop_lines(kernel, run, status, ran_s, files, more, why)
    for i, line in enumerate(lines):
        print((f"{kernel}: " if i == 0 else "  ") + line)


def wait(root: str | Path | None, kernel: str, interval: float = 60, max_hours: float = SESSION_LIMIT_H + 1) -> str:
    """Poll ``kernel`` until its run ends. A pending push is waited for first: until Kaggle runs it, the status
    Kaggle reports is an earlier version's."""
    root = _root(root)
    t_end = time.time() + max_hours * 3600
    last = None
    seen_running = None  # when a poll last found it running: it ran at least until then
    while True:
        pend = next((r for r in runs(root) if r.get("kernel") == kernel and r.get("status") == PENDING), None)
        if pend is not None and confirm_pending(root, [kernel]):
            pend = None
        if pend is not None:
            s, text = PENDING, ""
        else:
            try:
                s, text = kernel_status(kernel)
            except (OSError, subprocess.SubprocessError) as e:
                s, text = "unknown", str(e)
        if s != last:
            print(f"{_iso(_now())} {kernel}: {s}" + (" (waiting for its push from a background command to reach "
                                                     "Kaggle)" if pend is not None else ""), flush=True)
            note_status(root, kernel, s)
            last = s
        if s == "running":
            seen_running = _now()
        if pend is not None and _stale(pend, _now()):
            print(f"no version of {kernel} has run on Kaggle since its push from a background command at "
                  f"{pend.get('pushed_at')}: that push most likely failed (read that command's output). "
                  f"`kgkit gpu release {kernel}` drops its reservation.")
            return s
        if s in FINISHED:
            if s != "complete":
                started = _parse_time(latest_run(root, kernel).get("started_at"))
                watched = (seen_running - started).total_seconds() if seen_running and started else None
                report_stop(root, kernel, s, text, watched)
            return s
        if time.time() > t_end:
            print("stopped waiting (max time reached); the kernel may still be running")
            return s
        time.sleep(max(30.0, interval))


# ----------------------------------------------------------------------------- collect
def _next_exp_num(ledger_path: Path) -> int:
    nums = [0]
    if ledger_path.is_file():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            try:
                head = json.loads(line)["id"].split("_")[0]
            except (ValueError, KeyError):
                continue
            if head.isdigit():
                nums.append(int(head))
    return max(nums) + 1


def _merge_tree(src: Path, dst: Path) -> int:
    n = 0
    for p in src.rglob("*"):
        if p.is_file():
            q = dst / p.relative_to(src)
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
            n += 1
    return n


def import_ledger(root: Path, out: Path, report: dict, kernel: str, version: int | None) -> list[dict]:
    """Append the remote run's ledger records to the local ledger with fresh ids (idempotent)."""
    lp = root / S.STATE_DIR / "ledger.jsonl"
    existing = []
    if lp.is_file():
        existing = [json.loads(l) for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
    seen = {(r.get("remote") or {}).get("key") for r in existing}
    gpu_h = round(report.get("elapsed_s", 0) / 3600, 3)
    used_key = hours_key(report_resource(report))
    imported = []
    for rec in report.get("ledger", []):
        key = f"{report.get('build_id') or kernel}:{rec['id']}"
        if key in seen:
            continue
        num = _next_exp_num(lp)
        new_id = f"{num:04d}_{rec['id'].split('_', 1)[1] if '_' in rec['id'] else _slug(rec['name'])}"
        old_id = rec["id"]
        src_art = out / "artifacts" / old_id
        if src_art.is_dir():
            _merge_tree(src_art, root / "artifacts" / new_id)
        for k in ("oof_path", "test_path", "holdout_path"):
            if rec.get(k):
                rec[k] = rec[k].replace(f"artifacts/{old_id}/", f"artifacts/{new_id}/")
        if rec.get("submission"):
            src_sub = out / "subs" / f"{old_id}.csv"
            if src_sub.is_file():
                (root / "subs").mkdir(exist_ok=True)
                shutil.copy2(src_sub, root / "subs" / f"{new_id}.csv")
                rec["submission"] = f"subs/{new_id}.csv"
            else:
                rec["submission"] = None
        rec.update(id=new_id, git_commit=report.get("git_commit"), git_dirty=report.get("git_dirty"),
                   **{used_key: gpu_h},
                   remote={"key": key, "kernel": kernel, "version": version, "remote_id": old_id,
                           "accelerator": report.get("accelerator"), "gpus": report.get("gpus"),
                           used_key: gpu_h, "gpu_util": report.get("gpu_util")})
        lp.parent.mkdir(parents=True, exist_ok=True)
        with lp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        imported.append(rec)
    return imported


def _baseline_fold_score(root: Path, ref: str | None, fold: int) -> tuple[str | None, float | None, float | None]:
    """(baseline id, its score on ``fold``, its fold std) for a ledger id/name, or the accepted baseline."""
    lp = root / S.STATE_DIR / "ledger.jsonl"
    if not lp.is_file():
        return None, None, None
    recs = []
    for line in lp.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("fold_scores") and r.get("model") != "blend":
            recs.append(r)
    if not ref or ref in ("best", "baseline"):
        st = S.load(root)
        gib = st.greater_is_better if st else True
        cands = [r for r in recs if isinstance(r.get("cv"), (int, float))]
        rec = _accepted_or_best(cands, gib)
    else:
        rec = next((r for r in reversed(recs) if r.get("id") == ref or r.get("name") == ref), None)
    if rec is None or fold >= len(rec["fold_scores"]):
        return None, None, None
    return rec["id"], rec["fold_scores"][fold], rec.get("cv_std")


def report_resource(report: dict) -> str:
    return report.get("resource") or resource_of(report.get("accelerator"))


MIN_DIAG_S, MIN_DIAG_SAMPLES = 1200, 20


def diagnose(report: dict) -> list[str]:
    """Findings for a finished remote run: inputs it did not have, then quota efficiency."""
    out = []
    rows = report.get("inputs") or []
    lost = [f"{r['source']} ({r['error']})" for r in rows if r.get("error")]
    if lost:
        out.append("inputs the run did not have: " + "; ".join(lost) + " - `gpu push` prints the inputs Kaggle "
                   "could not attach, and a kernel's output is bound when the kernel reading it is pushed")
    skipped = [f"{e} from {r['source']} (already {why})" for r in rows for e, why in (r.get("skipped") or {}).items()]
    if skipped:
        more = f"; {len(skipped) - 8} more in kg_run.json" if len(skipped) > 8 else ""
        out.append(f"{len(skipped)} input entries were not linked, as the name was already taken: "
                   + "; ".join(skipped[:8]) + more + ". Give each input its own folder: `gpu build --input-layout "
                   "nested` (data/ext/<slug>/, kernel outputs data/ext/kout-<slug>/) or `--dataset "
                   "<owner>/<slug>=data/<folder>`")
    util = report.get("gpu_util") or {}
    gpus = report.get("gpus") or []
    elapsed = report.get("elapsed_s")
    samples = max((u.get("n") or 0 for u in util.values()), default=None) if util else None
    if any(u.get("n") is None for u in util.values()):
        samples = None
    if report_resource(report) != "GPU":
        util = {}
    elif (elapsed is not None and elapsed < MIN_DIAG_S) or (samples is not None and samples < MIN_DIAG_SAMPLES):
        if util:
            out.append(f"too short to diagnose GPU utilisation ({(elapsed or 0) / 60:.0f} min, {samples or 0} "
                       "samples, mostly start-up: installs, downloads, model build); judge throughput on a run "
                       f"of >= {MIN_DIAG_S // 60} min")
        util = {}
    busy = [u for u in util.values() if u.get("util_mean", 0) > 5]
    if len(gpus) >= 2 and util and len(busy) < 2:
        out.append("only one GPU did work: the second T4 was idle for this quota hour - give every session >= 2 "
                   "tasks (2 folds, or two 1-fold screens via --extra-job)")
    means = [u["util_mean"] for u in busy]
    if means and min(means) < 60:
        out.append(f"GPU utilisation {min(means):.0f}% (mean): input-bound. Kaggle GPU sessions have few CPU cores "
                   "(typically 4) - pre-resize/cache inputs in a CPU kernel or locally (no GPU quota), use uint8 "
                   "tensors, GPU-side augmentation, fewer/cheaper CPU transforms")
    if report.get("deadline_hit"):
        out.append("hit the session deadline: resume from this output with `gpu build ... --resume-from "
                   "<this kernel>` (finished folds are skipped, timed-out folds continue from their checkpoint)")
    statuses = [s for j in report.get("jobs", []) for s in j.get("fold_status", {}).values()]
    secs = [s["seconds"] for s in statuses if s.get("status") == "done" and s.get("seconds")]
    if secs:
        per_fold_h = sum(secs) / len(secs) / 3600
        par = max(1, len(gpus))
        out.append(f"cost: {per_fold_h:.2f}h wall per fold = {per_fold_h / par:.2f} quota-hours per fold with "
                   f"{par} GPU(s) busy")
    for j in report.get("jobs", []):
        pruned = [k for k, s in j.get("fold_status", {}).items() if s.get("status") == "pruned"]
        if pruned:
            out.append(f"{j['name']}: pruned on folds {pruned} (trailing the baseline curve) - discard the idea or "
                       "change it materially instead of spending a full run")
    return out


def _status_of(kernel: str) -> str:
    try:
        return kernel_status(kernel)[0]
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _move_tree(src: Path, dst: Path) -> list[str]:
    """Move the files under ``src`` to ``dst`` (replacing a file of the same name), then ``src`` itself away; the moved
    files' paths relative to ``dst``."""
    moved = []
    for p in sorted(src.rglob("*")):
        if p.is_file():
            rel = p.relative_to(src)
            q = dst / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(p, q)
            except OSError:  # another disk
                shutil.copy2(p, q)
                p.unlink()
            moved.append(rel.as_posix())
    shutil.rmtree(src, ignore_errors=True)
    return moved


def _session_log(out: Path, kslug: str) -> list[dict]:
    """The entries of the session log that `kaggle kernels output` writes as <slug>.log beside the output files."""
    p = out / f"{kslug}.log"
    try:
        return log_entries(p.read_text(encoding="utf-8", errors="replace")) if p.is_file() else []
    except OSError:
        return []


def _raw_log_tail(out: Path, n: int) -> list[str]:
    """The last ``n`` lines of the first *.log in ``out``, when the session log is not the JSON `log_entries` reads."""
    for p in sorted(out.glob("*.log")):
        return p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    return []


def _collect_without_report(root: Path, kernel: str, out: Path, status: str | None, staged: bool,
                            weights: bool) -> dict:
    """`collect` of an output with no kg_run.json, which only the runner of a `gpu build` kernel writes. The outcome is
    what Kaggle reports (``status``, asked here when None) and the hours are the session's (the run log, and the
    session log's last time). A `gpu build` kernel whose runner did not finish is an error. Any other kernel just has
    files: the ones downloaded to ``out`` (``staged``) move to ``kernel_output_dir``, to be read like any input, and
    nothing reaches the ledger, as no kg-train.json says which experiment a file belongs to."""
    kslug = kernel.split("/")[-1]
    if status is None:
        status = _status_of(kernel)
    if status in FINISHED:
        note_status(root, kernel, status)  # older still-open entries of this kernel ended before this output
    run = latest_run(root, kernel)
    entries = _session_log(out, kslug)
    ran_s = max((e["time"] for e in entries if isinstance(e.get("time"), (int, float))), default=None)
    key, res = hours_key(run_resource(run)), run_resource(run)
    fields = {"collected_at": _iso(_now())}
    if run.get("status") in ENDED:  # an entry still in flight is not charged a session that may not be over
        h = session_hours(run, ran_s)
        if h is not None and h > (run.get(key) or 0):
            fields[key] = h
    if stopped_by_cap(run, status, ran_s):
        fields["stopped_by"] = "cap"
    used = fields.get(key, run.get(key))
    if run.get("cmd"):
        fields["status"] = status if status in ("error", "cancelled") else "error"
        update_run(root, kernel, run.get("version"), **fields)
        raise SystemExit(f"{kernel}: Kaggle reports {status}, and its output has no kg_run.json: the runner did not "
                         "finish. Last log lines:\n" + "\n".join(log_tail(entries, 40) if entries else
                                                                  _raw_log_tail(out, 40))
                         + f"\nWhat it saved is in {out}; `gpu build ... --resume-from {kernel}` continues the folds "
                         "that finished or have a checkpoint."
                         + (f" {used:.2f} {res}-hours accounted." if used is not None else ""))
    if staged:
        where = kernel_output_dir(kernel)
        dest = root / where
        before = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()} if dest.is_dir() else set()
        files = _move_tree(out, dest)
        kept = sorted(before - set(files))
    else:
        where, kept = out.as_posix(), []
        files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    if status in FINISHED:
        fields["status"] = status
    if run:
        update_run(root, kernel, run.get("version"), outputs=where, **fields)
    if used is not None:
        hours = f", {used:.2f} {res}-hours (session wall time)"
    elif not run:
        hours = ", no hours accounted: it is not in the run log (`gpu log`), so its accelerator is unknown"
    else:
        hours = ", no hours accounted yet: the run log does not have it ended and the session log shows no time"
    saved = [f for f in files if f != f"{kslug}.log"]  # the session log comes with them, but the kernel did not save it
    outcome = ("" if status in FINISHED else " (Kaggle could not be asked for its outcome)" if status == "unknown"
               else f" (Kaggle has it {status})")
    lines = [f"collected {kernel}" + (f" v{run['version']}" if run.get("version") else "") + f": {status}" + outcome
             + (f", stopped by its {float(run['hours']):g}h cap" if fields.get("stopped_by") == "cap" else "") + hours,
             f"  {len(saved)} file(s) in {where}/"
             + (": " + ", ".join(saved[:8]) + (f" and {len(saved) - 8} more" if len(saved) > 8 else "") if saved
                else " (it saved none)")
             + (f" (+ {kslug}.log, the session log)" if len(saved) < len(files) else "")]
    if kept:
        lines.append(f"  kept {len(kept)} file(s) already there that this version did not write: "
                     + ", ".join(kept[:8]) + (f" and {len(kept) - 8} more" if len(kept) > 8 else ""))
    if staged and not weights:
        lines.append("  weights (.pt .pth .bin .ckpt .safetensors) are not downloaded; --weights gets them")
    if status == "error" and entries:
        lines += ["  last log lines:"] + ["    " + l for l in log_tail(entries, 15)]
    lines.append("  nothing went into the ledger: no kg-train.json says which experiment a file belongs to. OOF "
                 f"predictions: `python -m kgkit ledger import <name> {where}/<oof>.npy --test {where}/<test>.npy`; "
                 "a score only: `python -m kgkit ledger add <name> <cv> <notes>`.")
    summary = "\n".join(lines)
    print(summary)
    return {"report": None, "imported": [], "status": status, "files": files, "dir": where, "summary": summary}


def collect(root: str | Path | None, kernel: str, from_dir: str | Path | None = None, weights: bool = False,
            version: int | None = None) -> dict:
    """Bring the output of ``kernel`` back and account the hours its session used. A kernel that `gpu build` made
    reports through its runner (kg_run.json): its experiments go into the ledger and its artifacts into the project.
    For any other kernel the outcome is Kaggle's status and its files go to ``kernel_output_dir``
    (`_collect_without_report`). ``from_dir``: an output folder downloaded already. A kernel still queued or running
    is not collected."""
    root = _root(root)
    confirm_pending(root, [kernel])  # a push from a background command that Kaggle ran is the run collected here
    kslug = kernel.split("/")[-1]
    kstatus = None  # what Kaggle reports for the kernel; `status` below is the runner's account of its jobs
    if from_dir is None:
        kstatus = _status_of(kernel)
        if kstatus in ("queued", "running"):
            raise SystemExit(f"{kernel} is still {kstatus}: what it has output so far is an earlier version's, or "
                             f"partial. `python -m kgkit gpu wait {kernel} --collect` collects when it ends.")
        out = root / S.STATE_DIR / "remote" / kslug
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        args = ["kernels", "output", kernel, "-p", str(out), "-o", "-q", "--page-size", "200"]
        if not weights:
            args += ["--file-pattern", r"^(?!.*" + WEIGHT_RE + ")"]
        r = run_kaggle(args, timeout=3600)
        if r.returncode != 0:
            raise SystemExit(f"kaggle kernels output failed:\n{r.stdout}{r.stderr}")
    else:
        out = Path(from_dir)
    rp = out / "kg_run.json"
    if not rp.is_file():
        return _collect_without_report(root, kernel, out, kstatus, from_dir is None, weights)
    report = json.loads(rp.read_text(encoding="utf-8"))
    jobs = report.get("jobs", [])
    for j in jobs:
        if (out / "artifacts" / j["name"]).is_dir():
            _merge_tree(out / "artifacts" / j["name"], root / "artifacts" / j["name"])
    # older still-open entries of this kernel ended before this output
    note_status(root, kernel, kstatus if kstatus in FINISHED else "complete")
    # a push still pending is not on Kaggle yet: this output is an earlier version's
    run = latest_run(root, kernel)
    version = version or run.get("version")
    imported = import_ledger(root, out, report, kernel, version)
    gpu_h = round(report.get("elapsed_s", 0) / 3600, 3)
    res = report_resource(report)
    used_key = hours_key(res)
    by_name = {r["name"]: r for r in imported}
    status = ("complete" if jobs and all(j["complete"] for j in jobs)
              else "partial" if any(j["folds_done"] for j in jobs) else "failed")
    fold_scores = {j["name"]: {k: s.get("best") for k, s in j["fold_status"].items() if s.get("best") is not None}
                   for j in jobs}
    fields = dict(status="collected" if status == "complete" else status, **{used_key: gpu_h},
                  collected_at=_iso(_now()), fold_scores=fold_scores,
                  folds_done={j["name"]: j["folds_done"] for j in jobs},
                  exp_ids=[r["id"] for r in imported] or run.get("exp_ids"),
                  cv=imported[0]["cv"] if len(imported) == 1 else run.get("cv"),
                  gpu_util=report.get("gpu_util"), deadline_hit=report.get("deadline_hit"))
    if run:
        update_run(root, kernel, run.get("version"), **fields)
    else:
        log_run(root, {"kernel": kernel, "version": version, "name": jobs[0]["name"] if jobs else kslug,
                       "jobs": [j["name"] for j in jobs], "accelerator": report.get("accelerator"),
                       "resource": res, "hours": report.get("budget_h"), **fields})
    lines = [f"collected {kernel}: {status}, {gpu_h:.2f} {res if res != 'CPU' else 'CPU'}-hours (session wall time)"]
    prune = run.get("prune") or {}
    for j in jobs:
        state = "complete" if j["complete"] else f"folds done {j['folds_done']}, incomplete {j['folds_incomplete']}"
        lines.append(f"- {j['name']}: {state}")
        rec = by_name.get(j["name"])
        if rec is not None:
            lines.append(f"    ledger {rec['id']} CV {rec['cv']:.5f}"
                         + (f" +/- {E.fmt_se(rec['cv_std'])}" if rec.get("cv_std") is not None else "")
                         + (f", submission {rec['submission']}" if rec.get("submission") else ""))
            continue
        for k, v in sorted(fold_scores[j["name"]].items()):
            bid, base, std = _baseline_fold_score(root, prune.get("baseline"), int(k))
            extra = ""
            if base is not None:
                sd = f", baseline fold std {E.fmt_se(std)}" if std is not None else ""
                extra = f" vs {bid} fold {k} {base:.5f} (delta {E.fmt_gain(v - base)}{sd})"
            lines.append(f"    fold {k}: best {v:.5f} [{j['fold_status'][k].get('status')}]{extra}")
        for k, txt in (j.get("errors") or {}).items():
            lines.append(f"    fold {k} FAILED - last lines:\n" + "\n".join("      " + l for l in txt.splitlines()[-15:]))
    for d in diagnose(report):
        lines.append(f"  * {d}")
    summary = "\n".join(lines)
    print(summary)
    return {"report": report, "imported": imported, "status": status, "summary": summary}


# ----------------------------------------------------------------------------- planning
def run_table(root: str | Path | None = None, n: int = 20) -> str:
    root = _root(root)
    rs = runs(root)[-n:]
    if not rs:
        return "(no remote GPU runs logged yet)"
    lines = ["| kernel | v | name | acc | cap h | used h | folds | status | result |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        res = ""
        if r.get("cv") is not None:
            res = f"CV {r['cv']:.5f} ({', '.join(r.get('exp_ids') or [])})"
        elif r.get("fold_scores"):
            res = "; ".join(f"{n} " + ", ".join(f"f{k}:{v:.4f}" for k, v in sorted(fs.items()))
                            for n, fs in r["fold_scores"].items() if fs)
        elif r.get("outputs"):
            res = f"files in {r['outputs']}"
        used_h = used_hours(r)
        used = f"{used_h:.2f}" if used_h is not None else ""
        lines.append(f"| {r.get('kernel', '').split('/')[-1]} | {r.get('version') or ''} | "
                     f"{', '.join(r.get('jobs') or [r.get('name') or ''])} | "
                     f"{(r.get('accelerator') or '').replace('NvidiaTesla', '')} | {r.get('hours')} | {used} | "
                     f"{r.get('folds')} | {r.get('status')}{' (cap)' if r.get('stopped_by') == 'cap' else ''} | {res} |")
    every = runs(root)
    total = sum(used_hours(r) or 0 for r in every if run_resource(r) == "GPU")
    lines.append(f"\ntotal GPU-hours accounted: {total:.2f}")
    tpu = sum(used_hours(r) or 0 for r in every if run_resource(r) == "TPU")
    if tpu:
        lines.append(f"total TPU-hours accounted: {tpu:.2f} (separate quota)")
    return "\n".join(lines)


def plan(root: str | Path | None = None, refresh: bool = True, deadline: str | None = None) -> str:
    """Budget the GPU hours left until the deadline (weekly resets, final-week reserve, measured costs)."""
    root = _root(root)
    st = S.load(root)
    if refresh:
        refresh_runs(root)
    q, note = current_quota(root, refresh)
    lines = (_quota_lines(root, q, note, "NOT LIVE" if refresh else "cached") if q else
             [f"GPU quota unknown: {note} (run `kaggle quota` to see why)."])
    deadline = deadline or (st.deadline if st else "")
    dl = _parse_time(deadline[:10] + "T23:59:00Z") if deadline else None
    now = _now()
    if q and dl:
        weekly = q.get("total_h") or 30.0
        avail_now = max(0.0, available_hours(root, q))
        resets = resets_before(q, deadline)
        total = avail_now + resets * weekly
        days = (dl - now).total_seconds() / 86400
        lines.append(f"Until the deadline ({dl.date()}, {days:.1f} days): {avail_now:.1f}h now + {resets} weekly "
                     f"reset(s) x {weekly:.0f}h = ~{total:.0f} GPU-hours in total.")
        if st and st.code_competition:
            lines.append("Code competition: committing each GPU inference notebook version spends your quota "
                         "(it runs on the public test) - keep a few hours for the final inference kernels and "
                         "their reruns.")
        if days <= 7:
            lines.append("FINAL WEEK: no new ideas. Spend on (1) full-fold/extra-seed runs of the chosen ensemble "
                         "members, (2) inference kernels; keep 1-2 sessions of slack for failures.")
        else:
            final_reserve = min(total, weekly)
            lines.append(f"Reserve ~{final_reserve:.0f}h (the final week's quota) for final models + inference; "
                         f"~{max(0.0, total - final_reserve):.0f}h for exploration. Exploration = 1-fold screens "
                         "at reduced resolution/epochs with pruning; promote only ideas that beat the baseline "
                         "fold by more than its fold std.")
    tq = tpu_of(q)
    if tq is not None:
        tpu_left = max(0.0, available_hours(root, tq, "TPU"))
        gpu_left = max(0.0, available_hours(root, q)) if q else None
        if tpu_left >= 1.0:
            lines.append(f"TPU: {tpu_left:.1f}h of {tq.get('total_h', 0):.0f}h plannable (separate weekly quota, also "
                         "lost at the reset). Batch TPU jobs wait 10-24h in the queue (one at a time per account, "
                         "9h sessions), so a TPU run finishes at ~now + 13h + 20 min start + its T4 time / 5 "
                         "(measured for a bandwidth-bound ConvNeXt). Worth it for training longer than a GPU "
                         "session, or when the GPU quota is gone; validate the XLA port first (skill kaggle-gpu, "
                         "TPU section).")
            if gpu_left is not None and gpu_left < 1.0:
                lines.append("GPU quota is used up: queue validated multi-hour training on the TPU now, about half a "
                             "day before it is needed, and keep a GPU fallback for after the reset.")
    rs = [r for r in runs(root) if r.get("gpu_hours") and isinstance(r.get("folds_done"), dict)]
    rs = [r for r in rs if sum(len(v) for v in r["folds_done"].values())]
    if rs:
        lines.append("Measured costs (quota-hours per completed fold, sessions packed as run):")
        by = {}
        for r in rs:
            n = sum(len(v) for v in r["folds_done"].values())
            by.setdefault(", ".join(r["folds_done"]), []).append(r["gpu_hours"] / n)
        for k, v in list(by.items())[-8:]:
            lines.append(f"  {k}: {sum(v) / len(v):.2f}h/fold")
    lines.append("Rules: both T4s busy every session; preprocessing in CPU kernels; smoke-test locally before "
                 "pushing; cap every push with --hours ~1.3x the estimate; never leave an interactive GPU "
                 "session idle (it burns quota); queue long jobs before the weekly reset.")
    return "\n".join(lines)
