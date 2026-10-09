"""Training on Kaggle's GPUs: quota, remote training kernels, and GPU-hour accounting.

Kaggle gives each account a weekly GPU quota (``kaggle quota``; 30 h/week at the time of writing)
that does not roll over, with a per-session limit of 12 h. This module turns a local training
command into a kernel that runs it on Kaggle (both T4s busy, deadline-aware, resumable), and keeps
``.kaggle-gm/gpu_runs.jsonl`` so every GPU-hour can be traced to the experiment it bought::

    python -m kgkit gpu quota                                   # live quota + reset countdown
    python -m kgkit gpu build --name cnx384 --hours 6 \\
        --cmd "python src/train_image.py --name cnx384 --img-size 384 --folds {fold}"
    python -m kgkit gpu push kernels/cnx384-train               # asks nothing; check quota first
    python -m kgkit gpu wait <user>/cnx384-train                # poll until done (run in background)
    python -m kgkit gpu collect <user>/cnx384-train             # outputs -> ledger + GPU-hour log
    python -m kgkit gpu plan                                    # budget until the deadline

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

from . import state as S

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PLUGIN_ROOT / "templates" / "kaggle_gpu_runner.py"
RUNS_FILE = "gpu_runs.jsonl"
QUOTA_FILE = "gpu_quota.json"
SESSION_LIMIT_H = 12.0
ACCELERATORS = {
    "NvidiaTeslaT4": "2x T4 16 GB (two GPUs per quota hour: run two folds in parallel)",
    "NvidiaTeslaP100": "1x P100 16 GB (one GPU; no Triton/torch.compile, sm_60)",
}
WEIGHT_RE = r"\.(pt|pth|bin|ckpt|safetensors)$"
BUNDLE_EXCLUDE_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "wandb", ".venv", "venv"}
BUNDLE_EXCLUDE_SUFFIXES = {".pyc", ".pt", ".pth", ".bin", ".ckpt", ".safetensors", ".npy", ".npz", ".pkl",
                           ".parquet", ".feather", ".zip", ".png", ".jpg", ".jpeg", ".dcm"}
BUNDLE_WARN_MB, BUNDLE_MAX_MB = 2.0, 20.0
ACTIVE = ("pushed", "queued", "running")


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _slug(s: str, n: int = 50) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:n].strip("-") or "run"


def _root(root: str | Path | None) -> Path:
    if root is not None:
        return Path(root)
    return S.find_root() or Path.cwd()


# ----------------------------------------------------------------------------- kaggle CLI
def kaggle_cmd() -> list[str]:
    if os.environ.get("KGKIT_KAGGLE"):
        return shlex.split(os.environ["KGKIT_KAGGLE"], posix=os.name != "nt")
    exe = shutil.which("kaggle")
    return [exe] if exe else [sys.executable, "-m", "kaggle"]


def run_kaggle(args: list[str], timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run([*kaggle_cmd(), *args], capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")


def kaggle_username() -> str | None:
    """From KAGGLE_USERNAME or the username field of kaggle.json (the key is never read out)."""
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    cfg_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle")
    p = cfg_dir / "kaggle.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("username") or None
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------------- quota
def _hours(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    m = re.match(r"\s*([0-9.]+)\s*h?", str(v))
    return float(m.group(1)) if m else 0.0


def parse_quota(text: str) -> dict | None:
    """Parse ``kaggle quota --format json``."""
    try:
        rows = json.loads(text)
    except ValueError:
        return None
    for r in rows if isinstance(rows, list) else []:
        if str(r.get("resource", "")).upper() == "GPU":
            refresh = _parse_time(r.get("refreshAt") or r.get("refresh_at"))
            return {"used_h": _hours(r.get("used")), "remaining_h": _hours(r.get("remaining")),
                    "total_h": _hours(r.get("total")), "refresh_at": _iso(refresh) if refresh else None,
                    "fetched_at": _iso(_now())}
    return None


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


def cached_quota(root: str | Path) -> dict | None:
    p = quota_path(Path(root))
    try:
        q = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    refresh = _parse_time(q.get("refresh_at"))
    if refresh is not None and _now() >= refresh:  # the week rolled over since we looked
        q = {**q, "remaining_h": q.get("total_h", 0.0), "used_h": 0.0, "stale_reset": True}
    return q


def in_flight_hours(root: str | Path, quota: dict | None = None) -> float:
    """Upper bound of quota that kernels still running will consume after the quota snapshot."""
    fetched = _parse_time((quota or {}).get("fetched_at")) or _now()
    total = 0.0
    for r in runs(root):
        if r.get("status") not in ACTIVE:
            continue
        pushed = _parse_time(r.get("pushed_at"))
        cap = float(r.get("hours") or SESSION_LIMIT_H)
        if pushed is None:
            total += cap
            continue
        if (_now() - pushed).total_seconds() / 3600 > cap + 1:  # long over: stale entry
            continue
        already = max(0.0, (fetched - pushed).total_seconds() / 3600)
        total += max(0.0, cap - already)
    return total


def available_hours(root: str | Path, quota: dict | None = None) -> float | None:
    q = quota or cached_quota(root)
    if q is None:
        return None
    return q["remaining_h"] - in_flight_hours(root, q)


def quota_report(root: str | Path | None = None, refresh: bool = True) -> str:
    root = _root(root)
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
    refresh_at = _parse_time(q.get("refresh_at"))
    if refresh_at:
        h = (refresh_at - _now()).total_seconds() / 3600
        lines.append(f"  resets {_iso(refresh_at)} (in {h:.1f}h). Unused quota does NOT roll over.")
        left = q["remaining_h"] - inflight
        if 0 < h <= 36 and left >= 1.0:
            lines.append(f"  USE IT OR LOSE IT: {left:.1f}h expire in {h:.0f}h - queue long, useful jobs now "
                         "(extra seeds / full-fold runs of the current best, final-resolution models).")
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


def log_run(root: str | Path, rec: dict) -> dict:
    p = runs_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def update_run(root: str | Path, kernel: str, version: int | None = None, **fields) -> dict | None:
    """Update the latest run of ``kernel`` (optionally a given version)."""
    recs = runs(root)
    for r in reversed(recs):
        if r.get("kernel") == kernel and (version is None or r.get("version") in (None, version)):
            r.update(fields)
            p = runs_path(root)
            tmp = p.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(x) + "\n" for x in recs), encoding="utf-8")
            tmp.replace(p)
            return r
    return None


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


def _add_tree(z: zipfile.ZipFile, src: Path, arc: str) -> int:
    n = 0
    if src.is_file():
        z.write(src, arc)
        return 1
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if any(part in BUNDLE_EXCLUDE_DIRS for part in rel.parts) or not p.is_file():
            continue
        if p.suffix.lower() in BUNDLE_EXCLUDE_SUFFIXES:
            continue
        z.write(p, f"{arc}/{rel.as_posix()}")
        n += 1
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
            _add_tree(z, p, Path(inc).as_posix().strip("/"))
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
    if accelerator not in ACCELERATORS and accelerator != "none":
        raise SystemExit(f"--accelerator must be one of {sorted(ACCELERATORS)} or none")
    if not 0 < hours <= SESSION_LIMIT_H:
        raise SystemExit(f"--hours must be in (0, {SESSION_LIMIT_H}] (Kaggle session limit)")
    user = user or kaggle_username()
    if not user:
        raise SystemExit("cannot determine your Kaggle username: pass --user or set KAGGLE_USERNAME")
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
        "allow_cpu": accelerator == "none",
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
        "kernel_type": "script", "is_private": not public, "enable_gpu": accelerator != "none",
        "enable_tpu": False, "enable_internet": internet,
        "dataset_sources": [*datasets, *wheel_sources], "competition_sources": [st.slug],
        "kernel_sources": [*kernels, *resume_src], "model_sources": list(models),
    }
    if accelerator != "none":
        meta["machine_shape"] = accelerator
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    train_meta = {"kernel": meta["id"], "name": name, "jobs": [j["name"] for j in jobs], "hours": hours,
                  "accelerator": accelerator, "folds": folds, "cmd": cmd, "build_id": config["build_id"], "prune": prune,
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


def push(root: str | Path | None, kdir: str | Path, force: bool = False) -> dict:
    root = _root(root)
    kdir = Path(kdir)
    tm = json.loads((kdir / "kg-train.json").read_text(encoding="utf-8"))
    hours, acc = float(tm["hours"]), tm["accelerator"]
    if acc != "none":
        q = fetch_quota(root) or cached_quota(root)
        avail = available_hours(root, q) if q else None
        if avail is None:
            print("warning: GPU quota unknown (kaggle quota failed); pushing without a budget check")
        elif avail < hours and not force:
            raise SystemExit(f"refusing to push: run is capped at {hours:.2f}h but only {avail:.2f}h of GPU quota "
                             f"is plannable before {q.get('refresh_at')}. Lower --hours, wait for the reset, "
                             "or pass --force.")
        elif avail is not None:
            print(f"GPU budget: run capped at {hours:.2f}h; {avail:.2f}h plannable -> "
                  f"{avail - hours:.2f}h left in the worst case")
    args = ["kernels", "push", "-p", str(kdir), "-t", str(int(hours * 3600))]
    if acc != "none":
        args += ["--accelerator", acc]
    r = run_kaggle(args, timeout=900)
    out = (r.stdout or "") + (r.stderr or "")
    print(out.strip())
    m = re.search(r"version\s+(\d+)\s+successfully pushed", out, re.IGNORECASE)
    ok = r.returncode == 0 and m is not None
    rec = {"kernel": tm["kernel"], "version": int(m.group(1)) if m else None, "name": tm["name"],
           "jobs": tm.get("jobs") or [tm["name"]], "accelerator": acc, "hours": hours, "folds": tm.get("folds"), "cmd": tm.get("cmd"),
           "build_id": tm.get("build_id"), "prune": tm.get("prune"), "resume_from": tm.get("resume_from"),
           "pushed_at": _iso(_now()), "status": "pushed" if ok else "push-failed"}
    if ok:
        log_run(root, rec)
    if not ok:
        raise SystemExit("kernel push failed (see output above); nothing was started")
    return rec


def kernel_status(kernel: str) -> tuple[str, str]:
    r = run_kaggle(["kernels", "status", kernel], timeout=120)
    text = (r.stdout or "") + (r.stderr or "")
    return (normalize_status(text) if r.returncode == 0 else "unknown"), text.strip()


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
            if s in ("queued", "running"):
                update_run(root, kernel, status=s)
            last = s
        if s in ("complete", "error", "cancelled"):
            update_run(root, kernel, status=s, finished_at=_iso(_now()))
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
                   gpu_hours=gpu_h,
                   remote={"key": key, "kernel": kernel, "version": version, "remote_id": old_id,
                           "accelerator": report.get("accelerator"), "gpus": report.get("gpus"),
                           "gpu_hours": gpu_h, "gpu_util": report.get("gpu_util")})
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


def diagnose(report: dict) -> list[str]:
    """Quota-efficiency findings for a finished remote run."""
    out = []
    util = report.get("gpu_util") or {}
    gpus = report.get("gpus") or []
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
    run = next((r for r in reversed(runs(root)) if r.get("kernel") == kernel), {})
    version = version or run.get("version")
    imported = import_ledger(root, out, report, kernel, version)
    gpu_h = round(report.get("elapsed_s", 0) / 3600, 3)
    by_name = {r["name"]: r for r in imported}
    status = ("complete" if jobs and all(j["complete"] for j in jobs)
              else "partial" if any(j["folds_done"] for j in jobs) else "failed")
    fold_scores = {j["name"]: {k: s.get("best") for k, s in j["fold_status"].items() if s.get("best") is not None}
                   for j in jobs}
    fields = dict(status="collected" if status == "complete" else status, gpu_hours=gpu_h,
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
                       "hours": report.get("budget_h"), **fields})
    lines = [f"collected {kernel}: {status}, {gpu_h:.2f} GPU-hours (session wall time)"]
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
        used = f"{r['gpu_hours']:.2f}" if r.get("gpu_hours") is not None else ""
        lines.append(f"| {r.get('kernel', '').split('/')[-1]} | {r.get('version') or ''} | "
                     f"{', '.join(r.get('jobs') or [r.get('name') or ''])} | "
                     f"{(r.get('accelerator') or '').replace('NvidiaTesla', '')} | {r.get('hours')} | {used} | "
                     f"{r.get('folds')} | {r.get('status')} | {res} |")
    total = sum(r.get("gpu_hours") or 0 for r in runs(root))
    lines.append(f"\ntotal GPU-hours accounted: {total:.2f}")
    return "\n".join(lines)


def plan(root: str | Path | None = None, refresh: bool = True, deadline: str | None = None) -> str:
    """Budget the GPU hours left until the deadline (weekly resets, final-week reserve, measured costs)."""
    root = _root(root)
    st = S.load(root)
    q = (fetch_quota(root) if refresh else None) or cached_quota(root)
    lines = [quota_report(root, refresh=False) if q else "GPU quota unknown (run `kaggle quota`)."]
    deadline = deadline or (st.deadline if st else "")
    dl = _parse_time(deadline[:10] + "T23:59:00Z") if deadline else None
    now = _now()
    if q and dl:
        weekly = q.get("total_h") or 30.0
        refresh_at = _parse_time(q.get("refresh_at")) or now + dt.timedelta(days=7)
        avail_now = max(0.0, q["remaining_h"] - in_flight_hours(root, q))
        resets, t = 0, refresh_at
        while t < dl:
            resets += 1
            t += dt.timedelta(days=7)
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
