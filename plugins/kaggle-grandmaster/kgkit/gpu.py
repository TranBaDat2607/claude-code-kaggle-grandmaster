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
    python -m kgkit gpu plan                                    # budget until the deadline
    python -m kgkit gpu release <user>/<kernel>                 # drop the reservation of a finished run

Standard library only: hooks import it to read the cached quota and the run log.
"""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from . import safety
from . import state as S

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PLUGIN_ROOT / "templates" / "kaggle_gpu_runner.py"
RUNS_FILE = "gpu_runs.jsonl"
QUOTA_FILE = "gpu_quota.json"
SESSION_LIMIT_H = 12.0
SESSION_LIMITS = {"GPU": 12.0, "TPU": 9.0, "CPU": 12.0}
# Batch sessions an account may run at once; a push beyond this fails ("Maximum batch GPU session count
# of 2 reached", printed with exit status 0).
SESSION_SLOTS = {"GPU": 2, "TPU": 1}
# How long a pushed run may sit in the queue before an entry that never reported "running" is stale.
QUEUE_ALLOWANCE_H = {"GPU": 1.0, "TPU": 36.0, "CPU": 1.0}
ACCELERATORS = {
    "NvidiaTeslaT4": "2x T4 16 GB (two GPUs per quota hour: run two folds in parallel)",
    "NvidiaTeslaP100": "1x P100 16 GB (one GPU; no Triton/torch.compile, sm_60)",
    "TpuV5E8": "TPU v5e-8, 8 chips (bf16; TPU quota; queued for hours; one process per chip for {fold} tasks)",
    "TpuV6E8": "TPU v6e-8, 8 chips (separate pool, often no capacity)",
}
TPU_CHIPS = {"TpuV5E8": 8, "TpuV6E8": 8}
# Kaggle stores these as TpuV5E8 (verified Oct 2026): ask for the shape that will actually run.
TPU_REWRITTEN = {"TpuV5E1", "TpuV5E4", "TpuV6E1"}
WEIGHT_RE = r"\.(pt|pth|bin|ckpt|safetensors)$"
BUNDLE_EXCLUDE_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "wandb", ".venv", "venv"}
BUNDLE_EXCLUDE_SUFFIXES = {".pyc", ".pt", ".pth", ".bin", ".ckpt", ".safetensors", ".npy", ".npz", ".pkl",
                           ".parquet", ".feather", ".zip", ".png", ".jpg", ".jpeg", ".dcm"}
BUNDLE_WARN_MB, BUNDLE_MAX_MB = 2.0, 20.0
ACTIVE = ("pushed", "queued", "running")
FINISHED = ("complete", "error", "cancelled")


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


# ----------------------------------------------------------------------------- kaggle CLI
def kaggle_cmd() -> list[str]:
    if os.environ.get("KGKIT_KAGGLE"):
        return shlex.split(os.environ["KGKIT_KAGGLE"], posix=os.name != "nt")
    exe = shutil.which("kaggle")
    return [exe] if exe else [sys.executable, "-m", "kaggle"]


def run_kaggle(args: list[str], timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run([*kaggle_cmd(), *args], capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace", env=safety.kaggle_env())


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


def fetch_quota(root: str | Path | None = None) -> dict | None:
    """Live quota from the Kaggle API, cached in .kaggle-gm/gpu_quota.json. None when the CLI fails."""
    root = _root(root)
    try:
        r = run_kaggle(["quota", "--format", "json"], timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    q = parse_quota(r.stdout) if r.returncode == 0 else None
    if q is not None:
        p = quota_path(root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(q, indent=1), encoding="utf-8")
    return q


def _rolled(q: dict) -> dict:
    refresh = _parse_time(q.get("refresh_at"))
    if refresh is not None and _now() >= refresh:  # the week rolled over since we looked
        q = {**q, "remaining_h": q.get("total_h", 0.0), "used_h": 0.0, "stale_reset": True}
    return q


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
    (TPU jobs wait for hours) reserves its whole cap; a running one its cap minus the time it has run."""
    res = run_resource(r)
    cap = float(r.get("hours") or session_limit(res))
    start = _parse_time(r.get("started_at"))
    if start is None and res != "TPU" and r.get("status") != "queued":
        start = _parse_time(r.get("pushed_at"))  # GPU batch sessions usually start within minutes
    if start is None:
        return cap
    used = max(0.0, ((fetched or _now()) - start).total_seconds() / 3600)
    return max(0.0, cap - used)


def in_flight_hours(root: str | Path, quota: dict | None = None, resource: str = "GPU") -> float:
    """Upper bound of quota that kernels still running will consume after the quota snapshot."""
    fetched = _parse_time((quota or {}).get("fetched_at")) or _now()
    return sum(reserved_hours(r, fetched) for r in active_runs(root, resource))


def available_hours(root: str | Path, quota: dict | None = None, resource: str = "GPU") -> float | None:
    q = quota or cached_quota(root, resource)
    if q is None:
        return None
    return q["remaining_h"] - in_flight_hours(root, q, resource)


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
    fetched = _parse_time(q.get("fetched_at")) or _now()
    out = []
    for r in active_runs(root, resource):
        out.append(f"    - {run_label(r)} [{r.get('status')}, cap {float(r.get('hours') or 0):g}h"
                   + (", via kaggle CLI" if r.get("via") == "kaggle-cli" else "")
                   + f"]: ~{reserved_hours(r, fetched):.2f}h reserved")
    if out:
        out.append("    finished early or never ran? `kgkit gpu status` re-checks them; "
                   "`kgkit gpu release <kernel>` drops a reservation by hand")
    return out


def quota_report(root: str | Path | None = None, refresh: bool = True) -> str:
    root = _root(root)
    if refresh:
        refresh_runs(root)
    q = fetch_quota(root) if refresh else None
    live = q is not None
    q = q or cached_quota(root)
    if q is None:
        return "GPU quota unknown: `kaggle quota` failed (is the Kaggle CLI >= 2.2 installed and authenticated?)"
    inflight = in_flight_hours(root, q)
    lines = [f"Kaggle GPU quota: {q['used_h']:.2f}h used, {q['remaining_h']:.2f}h of {q['total_h']:.2f}h left"
             + ("" if live else f" (cached {q.get('fetched_at')})")]
    if inflight:
        lines.append(f"  in flight: up to {inflight:.2f}h more for running kernels -> "
                     f"{q['remaining_h'] - inflight:.2f}h plannable")
        lines += _reservation_lines(root, q, "GPU")
    refresh_at = _parse_time(q.get("refresh_at"))
    if refresh_at:
        h = (refresh_at - _now()).total_seconds() / 3600
        lines.append(f"  resets {_iso(refresh_at)} (in {h:.1f}h). Unused quota does NOT roll over.")
        left = q["remaining_h"] - inflight
        if 0 < h <= 36 and left >= 1.0:
            lines.append(f"  USE IT OR LOSE IT: {left:.1f}h expire in {h:.0f}h - queue long, useful jobs now "
                         "(extra seeds / full-fold runs of the current best, final-resolution models).")
    tq = cached_quota(root, "TPU")
    if tq is not None:
        tin = in_flight_hours(root, tq, "TPU")
        lines.append(f"Kaggle TPU quota: {tq['remaining_h']:.2f}h of {tq['total_h']:.2f}h left"
                     + (f", up to {tin:.2f}h in flight -> {tq['remaining_h'] - tin:.2f}h plannable" if tin else "")
                     + " (separate quota; 9h sessions, one batch TPU session at a time, queued 10-24h)")
        if tin:
            lines += _reservation_lines(root, tq, "TPU")
    return "\n".join(lines)


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
    only a kernel's latest version, and a re-push must not hold a second reservation."""
    recs = runs(root)
    for r in recs:
        if r.get("kernel") == rec.get("kernel") and r.get("status") in ACTIVE:
            r.update(status="superseded", finished_at=r.get("finished_at") or _iso(_now()))
    recs.append(rec)
    _write_runs(root, recs)
    return rec


def update_run(root: str | Path, kernel: str, version: int | None = None, **fields) -> dict | None:
    """Update the latest run of ``kernel`` (optionally a given version)."""
    recs = runs(root)
    for r in reversed(recs):
        if r.get("kernel") == kernel and (version is None or r.get("version") in (None, version)):
            r.update(fields)
            _write_runs(root, recs)
            return r
    return None


def note_status(root: str | Path, kernel: str, status: str) -> None:
    """Record a status read from Kaggle. A finished status closes every active entry of the kernel (Kaggle
    reports only the latest version, and earlier versions ended before it); a running one stamps
    ``started_at`` the first time it is seen, so a queued TPU job is not charged for its queue time."""
    recs = runs(root)
    mine = [r for r in recs if r.get("kernel") == kernel]
    if not mine or status not in (*FINISHED, "queued", "running"):
        return
    now = _iso(_now())
    if status in FINISHED:
        for r in mine:
            if r.get("status") in ACTIVE:
                r.update(status=status, finished_at=r.get("finished_at") or now)
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


def refresh_runs(root: str | Path) -> dict[str, str]:
    """Ask Kaggle for the status of every kernel the log still has as active (one call per kernel)."""
    kernels = list(dict.fromkeys(r["kernel"] for r in runs(root) if r.get("status") in ACTIVE and r.get("kernel")))
    out = {}
    for k in kernels:
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


def build_bundle(root: Path, include: list[str], baseline_dir: Path | None = None) -> bytes:
    """Zip of the code a remote run needs. Data never goes in here: attach it as a dataset/competition."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        _add_tree(z, PLUGIN_ROOT / "kgkit", "kgkit")
        z.write(root / S.STATE_DIR / S.STATE_FILE, f"{S.STATE_DIR}/{S.STATE_FILE}")
        folds = root / "data" / "folds.csv"
        if folds.is_file():  # the frozen split must be identical remotely (folds_hash matches the local ledger)
            z.write(folds, "data/folds.csv")
        for inc in include:
            p = (root / inc)
            if not p.exists():
                raise SystemExit(f"--include {inc}: not found under {root}")
            _add_tree(z, p, Path(inc).as_posix().strip("/"), base=root)
        if baseline_dir is not None:
            for c in sorted(baseline_dir.glob("curve_fold*.json")):
                z.write(c, f"_kg/baseline/{c.name}")
    return buf.getvalue()


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
                 public: bool = False, extra_jobs: list[tuple[str, str]] = ()) -> Path:
    """Write ``kernels/<slug>/`` (kernel-metadata.json + generated runner script + kg-train.json).

    ``extra_jobs`` ((name, cmd) pairs) share the session: e.g. two 1-fold screens of different ideas
    run side by side on the two T4s instead of leaving one GPU idle."""
    root = _root(root)
    st = S.load(root)
    if st is None:
        raise SystemExit("not inside a competition workspace (no .kaggle-gm/competition.json)")
    if accelerator in TPU_REWRITTEN:
        raise SystemExit(f"--accelerator {accelerator}: Kaggle stores it as TpuV5E8; ask for TpuV5E8")
    if accelerator not in ACCELERATORS and accelerator != "none":
        raise SystemExit(f"--accelerator must be one of {sorted(ACCELERATORS)} or none")
    res = resource_of(accelerator)
    if not 0 < hours <= session_limit(res):
        raise SystemExit(f"--hours must be in (0, {session_limit(res):g}] (Kaggle {res} session limit)")
    user = user or kaggle_username()
    if not user:
        raise SystemExit("cannot determine your Kaggle username (KAGGLE_USERNAME, kaggle.json and "
                         "`kaggle config view` gave none): pass --user")
    fp = root / "data" / "folds.csv"
    all_folds = None
    if fp.is_file():
        rows = fp.read_text(encoding="utf-8").splitlines()
        idx = rows[0].split(",").index("fold")
        all_folds = sorted({int(l.split(",")[idx]) for l in rows[1:] if l.strip()})
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
    config = {
        "jobs": jobs, "post_cmds": list(post_cmds),
        "hours": hours, "accelerator": accelerator, "pip": list(pip), "wheel_sources": list(wheel_sources),
        "sources": [st.slug, *datasets, *kernels], "resume_sources": resume_src, "prune": prune,
        "git_commit": commit, "git_dirty": dirty, "build_id": f"{kslug}@{built}",
        "allow_cpu": accelerator == "none", "resource": res, "tpu_chips": TPU_CHIPS.get(accelerator, 0),
    }
    payload = build_bundle(root, list(include), baseline_dir)
    mb = len(payload) / 1e6
    if mb > BUNDLE_MAX_MB:
        raise SystemExit(f"code bundle is {mb:.1f} MB: keep data out of --include (upload it as a dataset)")
    if mb > BUNDLE_WARN_MB:
        print(f"note: code bundle is {mb:.1f} MB; anything big belongs in a Kaggle dataset")
    script = RUNNER.read_text(encoding="utf-8")
    script = script.replace("__KG_CONFIG_B64__", base64.b64encode(json.dumps(config).encode()).decode(), 1)
    script = script.replace("__KG_PAYLOAD_B64__", base64.b64encode(payload).decode(), 1)

    kdir = Path(out_dir) if out_dir else root / "kernels" / kslug
    kdir.mkdir(parents=True, exist_ok=True)
    code_file = f"{kslug.replace('-', '_')}.py"
    (kdir / code_file).write_text(script, encoding="utf-8")
    meta = {
        "id": f"{user}/{kslug}", "title": kslug.replace("-", " "), "code_file": code_file, "language": "python",
        "kernel_type": "script", "is_private": not public, "enable_gpu": res == "GPU",
        "enable_tpu": res == "TPU", "enable_internet": internet,
        "dataset_sources": [*datasets, *wheel_sources], "competition_sources": [st.slug],
        "kernel_sources": [*kernels, *resume_src], "model_sources": list(models),
    }
    if accelerator != "none":
        meta["machine_shape"] = accelerator
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    train_meta = {"kernel": meta["id"], "name": name, "jobs": [j["name"] for j in jobs], "hours": hours,
                  "accelerator": accelerator, "resource": res, "folds": folds, "cmd": cmd, "build_id": config["build_id"], "prune": prune,
                  "resume_from": resume_src, "bundle_kb": round(len(payload) / 1024, 1)}
    (kdir / "kg-train.json").write_text(json.dumps(train_meta, indent=2), encoding="utf-8")
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
    """The accelerator a kernel-metadata.json asks for: its machine_shape, or 'none' without GPU/TPU."""
    shape = str(meta.get("machine_shape") or "")
    if _truthy(meta.get("enable_tpu")):
        return shape if shape.lower().startswith("tpu") else "TpuV5E8"
    if _truthy(meta.get("enable_gpu")) or shape.startswith("Nvidia"):
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


def busy_sessions(max_kernels: int = 8, within_h: float = 48.0) -> list[tuple[str, str]] | None:
    """(kernel, status) of the account's own kernels queued or running now, in any competition: the
    most recently run ``max_kernels`` are checked. None when the listing fails."""
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
    out = []
    for row in rows:
        ref, t = row.get("ref"), _parse_time(row.get("lastRunTime"))
        if not ref or t is None or (_now() - t).total_seconds() / 3600 > within_h:
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


def push(root: str | Path | None, kdir: str | Path, force: bool = False, hours: float | None = None,
         check_sources: bool = True, check_sessions: bool = True) -> dict:
    root = _root(root)
    kdir = Path(kdir)
    tm = push_spec(kdir, hours)
    hours, acc = float(tm["hours"]), tm["accelerator"]
    res = resource_of(acc)
    if res == "TPU" and acc in TPU_REWRITTEN:
        raise SystemExit(f"machine_shape {acc} is stored by Kaggle as TpuV5E8: ask for TpuV5E8")
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
        q = fetch_quota(root)
        q = (q.get("tpu") if q and res == "TPU" else q) or cached_quota(root, res)
        avail = available_hours(root, q, res) if q else None
        if avail is None:
            print(f"warning: {res} quota unknown (kaggle quota failed); pushing without a budget check")
        elif avail < hours and not force:
            raise SystemExit(f"refusing to push: run is capped at {hours:.2f}h but only {avail:.2f}h of {res} quota "
                             f"is plannable before {q.get('refresh_at')}. Lower --hours, wait for the reset, "
                             "check the reservations (`kgkit gpu quota`), or pass --force.")
        elif avail is not None:
            print(f"{res} budget: run capped at {hours:.2f}h; {avail:.2f}h plannable -> "
                  f"{avail - hours:.2f}h left in the worst case")
    args = ["kernels", "push", "-p", str(kdir), "-t", str(int(hours * 3600))]
    if acc != "none":
        args += ["--accelerator", acc]
    r = run_kaggle(args, timeout=900)
    out = (r.stdout or "") + (r.stderr or "")
    print(out.strip())
    m = re.search(r"version\s+(\d+)\s+successfully pushed", out, re.IGNORECASE)
    ok = r.returncode == 0 and m is not None
    if not ok:
        hint = ""
        if "session count" in out.lower():
            hint = (" Kaggle allows 2 batch GPU sessions and 1 batch TPU session at a time: wait for a running "
                    "kernel to finish (`kgkit gpu status`), or free a queued one (`kgkit gpu cancel <kernel>`).")
        raise SystemExit("kernel push failed (see output above); nothing was started." + hint)
    rec = {"kernel": tm["kernel"], "version": int(m.group(1)), "name": tm["name"],
           "jobs": tm.get("jobs") or [tm["name"]], "accelerator": acc, "resource": res, "hours": hours,
           "folds": tm.get("folds"), "cmd": tm.get("cmd"), "build_id": tm.get("build_id"), "prune": tm.get("prune"),
           "resume_from": tm.get("resume_from"), "pushed_at": _iso(_now()), "status": "pushed"}
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


def wait(root: str | Path | None, kernel: str, interval: float = 60, max_hours: float = SESSION_LIMIT_H + 1) -> str:
    root = _root(root)
    t_end = time.time() + max_hours * 3600
    last = None
    while True:
        try:
            s, text = kernel_status(kernel)
        except (OSError, subprocess.SubprocessError) as e:
            s, text = "unknown", str(e)
        if s != last:
            print(f"{_iso(_now())} {kernel}: {s}", flush=True)
            note_status(root, kernel, s)
            last = s
        if s in FINISHED:
            if s != "complete":
                print(text)
                try:
                    lg = run_kaggle(["kernels", "logs", kernel], timeout=300)
                    print("\n".join(lg.stdout.splitlines()[-60:]))
                except (OSError, subprocess.SubprocessError):
                    pass
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
    hours_key = "tpu_hours" if report_resource(report) == "TPU" else "gpu_hours"
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
        for k in ("oof_path", "test_path"):
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
                   **{hours_key: gpu_h},
                   remote={"key": key, "kernel": kernel, "version": version, "remote_id": old_id,
                           "accelerator": report.get("accelerator"), "gpus": report.get("gpus"),
                           hours_key: gpu_h, "gpu_util": report.get("gpu_util")})
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
    """Quota-efficiency findings for a finished remote run."""
    out = []
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


def collect(root: str | Path | None, kernel: str, from_dir: str | Path | None = None, weights: bool = False,
            version: int | None = None) -> dict:
    root = _root(root)
    kslug = kernel.split("/")[-1]
    if from_dir is None:
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
        update_run(root, kernel, status="error", collected_at=_iso(_now()))
        logs = sorted(out.glob("*.log"))
        tail = logs[0].read_text(encoding="utf-8", errors="replace").splitlines()[-40:] if logs else []
        raise SystemExit("no kg_run.json in the kernel output: the runner did not finish. Last log lines:\n"
                         + "\n".join(tail))
    report = json.loads(rp.read_text(encoding="utf-8"))
    jobs = report.get("jobs", [])
    for j in jobs:
        if (out / "artifacts" / j["name"]).is_dir():
            _merge_tree(out / "artifacts" / j["name"], root / "artifacts" / j["name"])
    note_status(root, kernel, "complete")  # older still-open entries of this kernel ended before this output
    run = next((r for r in reversed(runs(root)) if r.get("kernel") == kernel), {})
    version = version or run.get("version")
    imported = import_ledger(root, out, report, kernel, version)
    gpu_h = round(report.get("elapsed_s", 0) / 3600, 3)
    res = report_resource(report)
    hours_key = "tpu_hours" if res == "TPU" else "gpu_hours"
    by_name = {r["name"]: r for r in imported}
    status = ("complete" if jobs and all(j["complete"] for j in jobs)
              else "partial" if any(j["folds_done"] for j in jobs) else "failed")
    fold_scores = {j["name"]: {k: s.get("best") for k, s in j["fold_status"].items() if s.get("best") is not None}
                   for j in jobs}
    fields = dict(status="collected" if status == "complete" else status, **{hours_key: gpu_h},
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
                         + (f" +/- {rec['cv_std']:.5f}" if rec.get("cv_std") is not None else "")
                         + (f", submission {rec['submission']}" if rec.get("submission") else ""))
            continue
        for k, v in sorted(fold_scores[j["name"]].items()):
            bid, base, std = _baseline_fold_score(root, prune.get("baseline"), int(k))
            extra = ""
            if base is not None:
                sd = f", baseline fold std {std:.5f}" if std is not None else ""
                extra = f" vs {bid} fold {k} {base:.5f} (delta {v - base:+.5f}{sd})"
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
        used_h = r.get("tpu_hours") if r.get("tpu_hours") is not None else r.get("gpu_hours")
        used = f"{used_h:.2f}" if used_h is not None else ""
        lines.append(f"| {r.get('kernel', '').split('/')[-1]} | {r.get('version') or ''} | "
                     f"{', '.join(r.get('jobs') or [r.get('name') or ''])} | "
                     f"{(r.get('accelerator') or '').replace('NvidiaTesla', '')} | {r.get('hours')} | {used} | "
                     f"{r.get('folds')} | {r.get('status')} | {res} |")
    every = runs(root)
    total = sum(r.get("gpu_hours") or 0 for r in every if run_resource(r) == "GPU")
    lines.append(f"\ntotal GPU-hours accounted: {total:.2f}")
    tpu = sum(r.get("tpu_hours") or 0 for r in every)
    if tpu:
        lines.append(f"total TPU-hours accounted: {tpu:.2f} (separate quota)")
    return "\n".join(lines)


def plan(root: str | Path | None = None, refresh: bool = True, deadline: str | None = None) -> str:
    """Budget the GPU hours left until the deadline (weekly resets, final-week reserve, measured costs)."""
    root = _root(root)
    st = S.load(root)
    if refresh:
        refresh_runs(root)
    q = (fetch_quota(root) if refresh else None) or cached_quota(root)
    lines = [quota_report(root, refresh=False) if q else "GPU quota unknown (run `kaggle quota`)."]
    deadline = deadline or (st.deadline if st else "")
    dl = _parse_time(deadline[:10] + "T23:59:00Z") if deadline else None
    now = _now()
    if q and dl:
        weekly = q.get("total_h") or 30.0
        avail_now = max(0.0, q["remaining_h"] - in_flight_hours(root, q))
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
    tq = cached_quota(root, "TPU")
    if tq is not None:
        tpu_left = max(0.0, tq["remaining_h"] - in_flight_hours(root, tq, "TPU"))
        gpu_left = max(0.0, q["remaining_h"] - in_flight_hours(root, q)) if q else None
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
