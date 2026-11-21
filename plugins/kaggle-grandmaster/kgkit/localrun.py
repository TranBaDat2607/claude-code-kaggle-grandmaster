"""Run a `gpu build` script on this machine's GPU(s) instead of Kaggle: `kgkit gpu push <dir> --backend local`.

The generated runner does what it does in a Kaggle session (the fold split checked before anything runs, one fold
per GPU, resume, KG_DEADLINE, assemble, kg_run.json). Here it reads its inputs from a folder laid out like
/kaggle/input whose entries link to the workspace's own files, and writes its outputs to a folder that
`gpu collect` reads as it reads a downloaded Kaggle output, so the ledger import and its folds_hash check are the
same. The run is a detached background process: it outlives the command and the conversation.

Nothing is spent from a Kaggle quota: local runs are logged with resource LOCAL, which the quota arithmetic
(`active_runs(root, "GPU")`, `unseen_spend`, `run_lengths`) never counts.

Layout: ``.kaggle-gm/local/<kernel slug>/v<N>/`` holds ``input/`` (the links), ``working/`` (the outputs),
``run/`` (the runner's scratch copy of the code) and ``session.log`` (everything the runner prints).

Standard library only (kgkit.gpu imports it).
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

PREFIX = "local/"
LOCAL_DIR = "local"
MAX_HOURS = 168.0  # a week: past that a run should checkpoint and be pushed again with --resume-from
SPLIT_NAME = "folds.csv"
CONFIG_RE = re.compile(r'^CONFIG_B64 = "([A-Za-z0-9+/=]+)"', re.M)


def is_local(kernel: str | None) -> bool:
    return str(kernel or "").startswith(PREFIX)


def kernel_id(kslug: str) -> str:
    return PREFIX + kslug


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def link_entry(src: Path, dst: Path) -> str:
    """Make ``dst`` point at ``src`` without copying where the OS allows: a symlink, else (Windows without the
    symlink privilege) a junction for a folder or a hard link for a file; a copy only when neither works (a file
    on another drive). Returns how it was made."""
    try:
        os.symlink(str(src), str(dst), target_is_directory=src.is_dir())
        return "symlink"
    except (OSError, NotImplementedError):
        pass
    if src.is_dir():
        if os.name == "nt":
            try:
                import _winapi
                _winapi.CreateJunction(str(src.resolve()), str(dst))
                return "junction"
            except (ImportError, OSError):
                pass
        shutil.copytree(str(src), str(dst))
        return "copy"
    try:
        os.link(str(src), str(dst))
        return "hardlink"
    except OSError:
        shutil.copy2(str(src), str(dst))
        return "copy"


def pid_alive(pid) -> bool:
    """Whether process ``pid`` is still running (a zombie child of this process counts as ended)."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)  # reaps it when it is our own child that has ended
        if done == pid:
            return False
    except ChildProcessError:
        pass
    except OSError:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def status(run: dict) -> str:
    """'running', 'complete' (the runner wrote kg_run.json and exited: how its jobs went is for `collect`),
    'cancelled' or 'error' (it ended without a report), or 'unknown' for a run the log has no folder for."""
    out = run.get("out")
    if not out:
        return "unknown"
    if pid_alive(run.get("pid")):
        return "running"
    if (Path(out) / "kg_run.json").is_file():
        return "complete"
    return "cancelled" if run.get("status") == "cancelled" else "error"


def latest(runs: list[dict], kernel: str) -> dict:
    return next((r for r in reversed(runs) if r.get("kernel") == kernel), {})


def log_tail(run: dict, n: int = 60) -> list[str]:
    p = Path(run.get("dir") or "") / "session.log"
    try:
        return p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def embedded_config(script: Path) -> dict:
    m = CONFIG_RE.search(script.read_text(encoding="utf-8"))
    if m is None:
        raise SystemExit(f"{script} has no embedded run config: rebuild it with `kgkit gpu build`")
    return json.loads(base64.b64decode(m.group(1)).decode("utf-8"))


def _owner_slug(ref: str) -> tuple[str, str]:
    parts = ref.strip("/").split("/")
    return (parts[0], parts[-1]) if len(parts) > 1 else ("", parts[-1])


def dataset_copy(root: Path, ref: str) -> Path | None:
    """A local copy of dataset ``ref``: data/ext/<slug>/ (where `gpu build --input-layout nested` mounts it) or
    data/<slug>/."""
    slug = _owner_slug(ref)[1]
    for p in (root / "data" / "ext" / slug, root / "data" / slug):
        if p.is_dir():
            return p
    return None


def kernel_copy(root: Path, ref: str, runs: list[dict], state_dir: str, output_dir: str) -> Path | None:
    """The output of kernel ``ref`` on this machine: the latest local run of it (``local/<slug>`` or a Kaggle
    kernel of the same slug run locally), else what `gpu collect` downloaded of it."""
    slug = _owner_slug(ref)[1]
    for kernel in dict.fromkeys([ref if is_local(ref) else kernel_id(slug), kernel_id(slug)]):
        out = latest(runs, kernel).get("out")
        if out and Path(out).is_dir():
            return Path(out)
    for p in (root / state_dir / "remote" / slug, root / output_dir):
        if p.is_dir():
            return p
    return None


def prepare_inputs(root: Path, cfg: dict, input_dir: Path, runs: list[dict], state_dir: str, output_dir_of,
                   competition: str) -> tuple[list[str], dict[str, str]]:
    """Lay out ``input_dir`` like /kaggle/input with links to the workspace's files. Returns (lines saying what
    was linked, {input with no local copy: 'datasets' or 'notebooks'})."""
    said, missing, copies = [], {}, {}
    for s in cfg.get("sources", []):
        if s.get("kind") == "dataset":
            copies[s["ref"]] = ("datasets", dataset_copy(root, s["ref"]))
        elif s.get("kind") == "kernel":
            copies[s["ref"]] = ("notebooks", kernel_copy(root, s["ref"], runs, state_dir, output_dir_of(s["ref"])))
    for ref in [*cfg.get("resume_sources", [])]:
        copies[ref] = ("notebooks", kernel_copy(root, ref, runs, state_dir, output_dir_of(ref)))
    for ref in [*cfg.get("wheel_sources", []), *([cfg["split"]["from"]] if (cfg.get("split") or {}).get("from")
                                                 else [])]:
        copies[ref] = ("datasets", dataset_copy(root, ref))
    # the competition's own files: everything in data/ but the fold split (the runner writes it from the bundle)
    # and the folders that are another input's local copy (each is linked as that input)
    data = root / "data"
    held = {p.resolve().relative_to(data.resolve()).parts[0] for _, p in copies.values()
            if p is not None and data.resolve() in p.resolve().parents}
    comp = input_dir / "competitions" / competition
    comp.mkdir(parents=True)
    how = {}
    if data.is_dir():
        for k in sorted(data.iterdir()):
            if k.name == SPLIT_NAME or k.name in held:
                continue
            m = link_entry(k, comp / k.name)
            how[m] = how.get(m, 0) + 1
    said.append(f"competition {competition}: {sum(how.values())} entries of data/ ("
                + ", ".join(f"{n} {m}" for m, n in sorted(how.items())) + ")" if how else
                f"competition {competition}: data/ is empty or missing")
    for ref, (kind, src) in copies.items():
        if src is None:
            missing[ref] = kind
            continue
        owner, slug = _owner_slug(ref)
        dst = input_dir / kind / (owner or "local") / slug
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        said.append(f"{ref}: {src} ({link_entry(src, dst)})")
    return said, missing


def _spawn(cmd: list[str], cwd: Path, env: dict, log: Path) -> int:
    """Start ``cmd`` detached from this process (it outlives the command and the conversation); returns its pid."""
    f = open(log, "ab")
    try:
        kw = dict(cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
        if os.name == "nt":
            base = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008 | subprocess.CREATE_NO_WINDOW  # DETACHED_PROCESS
            try:  # out of the job object of the shell that ran kgkit, so closing it does not end the run
                p = subprocess.Popen(cmd, creationflags=base | 0x01000000, **kw)  # CREATE_BREAKAWAY_FROM_JOB
            except OSError:
                p = subprocess.Popen(cmd, creationflags=base, **kw)
        else:
            p = subprocess.Popen(cmd, start_new_session=True, **kw)
        return p.pid
    finally:
        f.close()


def push(root: Path, kdir: Path, runs: list[dict], state_dir: str, output_dir_of, competition: str,
         hours: float | None = None, force: bool = False, gpu_note: str = "") -> dict:
    """Start the `gpu build` script in ``kdir`` on this machine. Returns the run-log entry (not yet logged)."""
    tm_path, meta_path = kdir / "kg-train.json", kdir / "kernel-metadata.json"
    if not tm_path.is_file() or not meta_path.is_file():
        raise SystemExit(f"{kdir} is not a `gpu build` output (no kg-train.json): --backend local runs only those. "
                         "A notebook of your own runs locally as it is.")
    tm = json.loads(tm_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    script = kdir / meta["code_file"]
    cfg = embedded_config(script)
    hours = float(hours if hours is not None else tm["hours"])
    if not 0 < hours <= MAX_HOURS:
        raise SystemExit(f"--hours must be in (0, {MAX_HOURS:g}] for a local run")
    kslug = tm["kernel"].split("/")[-1]
    kernel = kernel_id(kslug)
    busy = [r for r in runs if is_local(r.get("kernel")) and r.get("status") in ("pushed", "running")
            and status(r) == "running"]
    if busy and not force:
        raise SystemExit("refusing to start: local run(s) " + ", ".join(f"{r['kernel']} v{r.get('version')}"
                                                                       for r in busy)
                         + " still running on this machine's GPU(s), and two runs on one GPU share its memory. "
                         "Wait for it (`kgkit gpu wait <kernel>`), or pass --force.")
    if cfg.get("accelerator") != "none" and shutil.which("nvidia-smi") is None and not force:
        raise SystemExit("refusing to start: no NVIDIA GPU on this machine (nvidia-smi not found), so the run "
                         "would train on the CPU. Push it to Kaggle instead, or pass --force.")
    base = root / state_dir / LOCAL_DIR / kslug
    version = 1 + max([int(p.name[1:]) for p in base.glob("v*") if p.name[1:].isdigit()] or [0])
    vdir = base / f"v{version}"
    input_dir, out, scratch = vdir / "input", vdir / "working", vdir / "run"
    out.mkdir(parents=True)
    said, missing = prepare_inputs(root, cfg, input_dir, runs, state_dir, output_dir_of, competition)
    if missing and not force:
        shutil.rmtree(str(vdir))
        hint = "; ".join(f"`kaggle datasets download -d {m} -p data/ext/{_owner_slug(m)[1]} --unzip`"
                         if kind == "datasets" else f"`kgkit gpu collect {m}` (or run it locally first)"
                         for m, kind in missing.items())
        raise SystemExit("refusing to start: no local copy of " + ", ".join(missing) + f". Get it: {hint}. "
                         "Or pass --force to run without it.")
    env = dict(os.environ, KAGGLE_INPUT=str(input_dir), KAGGLE_WORKING=str(out), KG_RUN_DIR=str(scratch),
               KG_HOURS=str(hours), KG_EXPECT_GPUS="0", KG_MONITOR_S="0", PYTHONUNBUFFERED="1")
    log = vdir / "session.log"
    pid = _spawn([sys.executable, "-u", str(script)], vdir, env, log)
    now = _now()
    for line in said:
        print(f"input {line}")
    if missing:
        print(f"WARNING: started without {', '.join(missing)} (--force)")
    if gpu_note:
        print(f"WARNING: {gpu_note}")
    return {"kernel": kernel, "version": version, "name": tm["name"], "jobs": tm.get("jobs") or [tm["name"]],
            "backend": "local", "accelerator": "local", "resource": "LOCAL", "hours": hours,
            "folds": tm.get("folds"), "cmd": tm.get("cmd"), "build_id": tm.get("build_id"), "prune": tm.get("prune"),
            "resume_from": tm.get("resume_from"), "pushed_at": now, "started_at": now, "status": "running",
            "pid": pid, "dir": str(vdir), "out": str(out), "built_from": kdir.as_posix()}


def cancel(run: dict) -> str:
    """Stop a local run and the training processes it started. Its finished folds stay in its working/ folder."""
    pid = run.get("pid")
    if not pid_alive(pid):
        return f"{run.get('kernel')} v{run.get('version')} is not running ({status(run)})"
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=60)
    else:
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except OSError:
            pass
    return (f"stopped {run.get('kernel')} v{run.get('version')} (pid {pid}). Folds it finished are in "
            f"{run.get('out')}/artifacts: `kgkit gpu collect {run.get('kernel')}` brings them back, and a new build "
            f"with --resume-from {run.get('kernel')} skips them.")
