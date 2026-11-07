"""Where a training or inference loop spends its time, and the speed-up of a change.

Standard library only: the templates import it from a vendored copy, and it never imports torch.

In-process, as the templates' ``--bench`` does::

    it = forever(train_loader)
    rates = measure({"data": data_step, "compute": compute_step, "full": full_step}, n=30,
                    sync=torch.cuda.synchronize)
    report("cnx_384", rates, config=cfg, kind="train_image")

``data`` runs the loader alone, ``compute`` the training step on batches already on the device, ``full`` the real
loop. Each step function runs one batch and returns how many samples it processed.

Out-of-process, for any command: ``python -m kgkit bench --run "<cmd>" [--run "<cmd>" ...]`` prints the wall time,
GPU utilisation and memory sampled with nvidia-smi, a verdict, and the speed-up of each run against the first.

Both append to ``reports/bench.jsonl`` (in the competition workspace, else the current folder) and print the speed-up
against the previous bench with the same name, with the settings that changed. A remote Kaggle session saves its
``reports/`` with the output, and ``kgkit gpu collect`` merges those records into the local file (``merge``), with
the speed-up of each against the local history. Output is ASCII.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator

INPUT_BOUND, COMPUTE_BOUND, STALLS = "INPUT-BOUND", "COMPUTE-BOUND", "STALLS"
IDLE, UNDER_USED, BUSY = "GPU IDLE", "GPU UNDER-USED", "GPU BUSY"

ADVICE = {
    INPUT_BOUND: "the data loader alone is slower than the GPU step. Cache the inputs as uint8 at the training size, "
                 "raise num_workers up to CPU cores / processes, move augmentation to the GPU, tokenise once.",
    COMPUTE_BOUND: "the real loop runs at the GPU step's rate. Speed now comes from less compute: AMP, channels_last, "
                   "torch.compile, a smaller resolution / max_len / backbone, a larger batch.",
    STALLS: "the real loop is well below both the loader and the step. Look for per-step .item()/.cpu() syncs, "
            "logging or validation too often, tiny batches, metrics computed on the CPU.",
    IDLE: "the GPU did no work: the job ran on the CPU. Check the device flags (.to('cuda'), XGBoost "
          "device='cuda', CatBoost task_type='GPU'). CPU work belongs in a CPU session, which spends no GPU quota.",
    UNDER_USED: "the GPU waited most of the time: input-bound or CPU-bound (decoding, preprocessing, Python loops). "
                "A template's --bench separates the data loader from the GPU step.",
    BUSY: "the GPU was busy. Speed comes from less compute (AMP, smaller inputs, torch.compile) or from the "
          "second GPU.",
}


# ----------------------------------------------------------------------------- in-process timing
def forever(loader: Iterable) -> Iterator:
    """Batches from ``loader`` without end. ``itertools.cycle`` would keep every batch in memory."""
    while True:
        empty = True
        for batch in loader:
            empty = False
            yield batch
        if empty:
            raise ValueError("the loader yields no batches")


def measure(steps: dict, n: int = 30, warmup: int = 3, sync: Callable[[], None] | None = None,
            max_seconds: float = 120.0) -> dict:
    """Items per second of each step function, in order. Each call runs one step and returns the number of items it
    processed. ``sync`` (e.g. ``torch.cuda.synchronize``) runs at both ends of the timed region so queued GPU work is
    counted. A phase stops after ``max_seconds`` once it has at least two timed steps."""
    rates = {}
    for name, step in steps.items():
        for _ in range(max(0, warmup)):
            step()
        if sync:
            sync()
        t0, items, done = time.perf_counter(), 0, 0
        while done < max(1, n):
            items += int(step() or 0)
            done += 1
            if done >= 2 and time.perf_counter() - t0 > max_seconds:
                break
        if sync:
            sync()
        dt = time.perf_counter() - t0
        rates[name] = round(items / dt, 3) if dt > 0 else float("inf")
    return rates


def verdict(rates: dict) -> str:
    """INPUT-BOUND / COMPUTE-BOUND / STALLS from the data-only, compute-only and real-loop rates ('' without all three).
    The slower of the loader and the step limits the loop; a real loop far below that limit is a stall."""
    data, compute, full = (rates.get(k) for k in ("data", "compute", "full"))
    if not (data and compute and full):
        return ""
    if full < 0.7 * min(data, compute):
        return STALLS
    return INPUT_BOUND if data < compute else COMPUTE_BOUND


# ----------------------------------------------------------------------------- records
def bench_path(root: str | Path | None = None) -> Path:
    """reports/bench.jsonl in the competition workspace (``root`` or found upwards from the current folder), else in
    the current folder."""
    if root is None:
        try:
            from . import state as S

            root = S.find_root()
        except Exception:
            root = None
    return Path(root or ".") / "reports" / "bench.jsonl"


def records(path: str | Path) -> list:
    out = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def previous(path: str | Path, name: str, kind: str = "") -> tuple:
    """(record, how) of the last bench with this name, else the last one of this kind; (None, '') if none."""
    recs = records(path)
    for rec in reversed(recs):
        if rec.get("name") == name:
            return rec, "this name"
    if kind:
        for rec in reversed(recs):
            if rec.get("kind") == kind:
                return rec, f"this script ({rec.get('name')})"
    return None, ""


def _append(path: Path, rec: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        pass


def _fmt(v) -> str:
    s = json.dumps(v, default=str) if isinstance(v, (list, dict)) else str(v)
    return s if len(s) <= 40 else s[:37] + "..."


def changed(old: dict, new: dict) -> list:
    """'key old -> new' for each setting that differs."""
    keys = sorted(set(old) | set(new))
    return [f"{k} {_fmt(old.get(k))} -> {_fmt(new.get(k))}" for k in keys if old.get(k) != new.get(k)]


def speedup_line(prev: dict, cur: dict, how: str = "this name") -> str:
    """The speed-up of ``cur`` against ``prev``: the real-loop rate for loop benches, wall time for commands."""
    when = str(prev.get("time", ""))[:16]
    if cur.get("kind") == "command" or "wall_s" in cur:
        a, b = prev.get("wall_s"), cur.get("wall_s")
        ratio = (a / b) if a and b else None
        what = "wall time"
    else:
        a, b = (prev.get("rates") or {}).get("full"), (cur.get("rates") or {}).get("full")
        ratio = (b / a) if a and b else None
        what = "real loop"
    if ratio is None:
        return f"[bench] previous bench of {how} ({when}) has no comparable {what}"
    word = "faster" if ratio >= 1 else "slower"
    diff = changed(prev.get("config") or {}, cur.get("config") or {})
    tail = f"; changed: {', '.join(diff[:8])}" if diff else "; no setting changed (code or hardware did)"
    return f"[bench] vs previous bench of {how} ({when}): {what} x{ratio:.2f} {word}{tail}"


def report(name: str, rates: dict, config: dict | None = None, extra: dict | None = None,
           root: str | Path | None = None, kind: str = "loop", out: Callable[[str], None] = print) -> dict:
    """Print the rates, the verdict and the speed-up against the previous bench of ``name``; append the record."""
    v = verdict(rates)
    lines = []
    if all(k in rates for k in ("data", "compute", "full")):
        lines.append(f"[bench] {name}: data loader alone {rates['data']:.1f}/s | training step alone "
                     f"{rates['compute']:.1f}/s | real loop {rates['full']:.1f}/s (samples per second)")
    else:
        lines.append(f"[bench] {name}: " + " | ".join(f"{k} {r:.1f}/s" for k, r in rates.items()))
    if v:
        lines.append(f"[bench] verdict: {v} - {ADVICE[v]}")
    for k, val in (extra or {}).items():
        lines.append(f"[bench] {k}: {val}")
    rec = {"name": name, "kind": kind, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "rates": rates, "verdict": v,
           "config": config or {}}
    if extra:
        rec["extra"] = extra
    path = bench_path(root)
    prev, how = previous(path, name, kind)
    if prev is not None:
        lines.append(speedup_line(prev, rec, how))
    _append(path, rec)
    lines.append(f"[bench] recorded in {path.as_posix()}")
    for line in lines:
        out(line)
    return rec


def _key(rec: dict) -> str:
    return json.dumps([rec.get("name"), rec.get("time"), rec.get("rates"), rec.get("wall_s")], sort_keys=True,
                      default=str)


def _summary(rec: dict) -> str:
    r = rec.get("rates") or {}
    if all(k in r for k in ("data", "compute", "full")):
        what = f"data {r['data']:.1f}/s, step {r['compute']:.1f}/s, real loop {r['full']:.1f}/s"
    elif rec.get("wall_s") is not None:
        what = f"wall {rec['wall_s']:.1f}s"
    else:
        what = ", ".join(f"{k} {v:.1f}/s" for k, v in r.items())
    return f"{what}; {rec.get('verdict') or 'no verdict'}"


def merge(src: str | Path, dst: str | Path) -> list:
    """Append the records of ``src`` (a remote session's reports/bench.jsonl) that ``dst`` lacks, oldest first. Returns
    the lines to print: each record's rates and verdict, and its speed-up against the previous bench in ``dst`` of the
    same name (or, for a template's loop bench, the same script)."""
    dst = Path(dst)
    have = {_key(r) for r in records(dst)}
    lines = []
    for rec in records(src):
        if _key(rec) in have:
            continue
        name, kind = str(rec.get("name", "")), str(rec.get("kind", ""))
        prev, how = previous(dst, name, "" if kind == "command" else kind)
        lines.append(f"[bench] remote {name} ({str(rec.get('time', ''))[:16]}): {_summary(rec)}")
        if prev is not None and prev.get("rc", 0) == 0 and rec.get("rc", 0) == 0:
            lines.append("  " + speedup_line(prev, rec, "this command" if kind == "command" else how))
        _append(dst, rec)
        have.add(_key(rec))
    if lines:
        lines.append(f"[bench] merged into {dst.as_posix()}")
    return lines


# ----------------------------------------------------------------------------- GPU sampling
def smi_command() -> list | None:
    """nvidia-smi, or the script in KGKIT_NVIDIA_SMI (tests; a .py file runs with this interpreter)."""
    fake = os.environ.get("KGKIT_NVIDIA_SMI")
    if fake:
        return [sys.executable, fake] if fake.endswith(".py") else [fake]
    exe = shutil.which("nvidia-smi")
    return [exe] if exe else None


class GpuSampler(threading.Thread):
    """Samples utilisation and memory of every GPU every ``interval`` seconds until ``stop`` is set."""

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval, self.samples, self.stop = interval, {}, threading.Event()
        self.cmd = smi_command()

    def sample(self) -> None:
        try:
            out = subprocess.run([*self.cmd, "--query-gpu=index,utilization.gpu,memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                 timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            return
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and parts[1].isdigit():
                s = self.samples.setdefault(parts[0], {"util": [], "mem_mb": []})
                s["util"].append(int(parts[1]))
                try:
                    s["mem_mb"].append(int(float(parts[2])))
                except ValueError:
                    pass

    def run(self):
        if not self.cmd:
            return
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(self.interval)

    def summary(self) -> dict:
        out = {}
        for idx, s in sorted(self.samples.items()):
            if s["util"]:
                u = sorted(s["util"])
                out[idx] = {"util_mean": round(sum(u) / len(u), 1), "util_p10": u[len(u) // 10],
                            "mem_peak_mb": max(s["mem_mb"]) if s["mem_mb"] else None, "n": len(u)}
        return out


def gpu_verdict(gpus: dict) -> tuple:
    """(verdict, notes) from a GpuSampler summary; ('', notes) when no GPU was sampled."""
    if not gpus:
        return "", ["no GPU sampled (nvidia-smi not found, or the run ended before the first sample): wall time only"]
    means = {i: g["util_mean"] for i, g in gpus.items()}
    busy = [i for i, m in means.items() if m > 5]
    notes = []
    if busy and len(busy) < len(means):
        idle = ", ".join(f"gpu{i}" for i in means if i not in busy)
        notes.append(f"{idle} idle while another GPU worked: run a second fold or job on it")
    if not busy:
        return IDLE, notes
    top = max(means[i] for i in busy)
    return (BUSY if top >= 60 else UNDER_USED), notes


def run_command(cmd: str, sample_s: float = 0.5, cwd: str | None = None) -> dict:
    """Run a shell command, sampling the GPUs; wall time, return code and the GPU summary."""
    sampler = GpuSampler(sample_s)
    sampler.start()
    t0 = time.perf_counter()
    try:
        rc = subprocess.run(cmd, shell=True, cwd=cwd).returncode
    finally:
        wall = time.perf_counter() - t0
        sampler.stop.set()
        sampler.join(timeout=30)
    return {"command": cmd, "wall_s": round(wall, 2), "rc": rc, "gpus": sampler.summary()}


def main_bench(a) -> None:
    """``kgkit bench --run CMD [--run CMD ...]``: time each command with GPU sampling, compare with the first."""
    if not a.run:
        raise SystemExit("give at least one --run \"<command>\" (a reduced setting that runs in minutes)")
    path = bench_path()
    results = []
    for k, cmd in enumerate(a.run, 1):
        print(f"[bench] run {k}: {cmd}", flush=True)
        res = run_command(cmd, a.sample_s)
        v, notes = gpu_verdict(res["gpus"])
        gpu_txt = "; ".join(f"gpu{i} util mean {g['util_mean']:.0f}% (p10 {g['util_p10']}%), memory peak "
                            f"{g['mem_peak_mb']} MB" for i, g in res["gpus"].items())
        print(f"[bench] run {k}: wall {res['wall_s']:.1f}s, exit code {res['rc']}" + (f"; {gpu_txt}" if gpu_txt else ""))
        if res["gpus"] and min(g["n"] for g in res["gpus"].values()) < 5:
            notes.append("few samples: a short run is mostly start-up (imports, data loading); judge utilisation on "
                         "a run of a few minutes")
        if v:
            print(f"[bench] run {k} verdict: {v} - {ADVICE[v]}")
        for note in notes:
            print(f"[bench] run {k}: {note}")
        name = a.name if a.name and len(a.run) == 1 else (f"{a.name}#{k}" if a.name else cmd)
        rec = {"name": name, "kind": "command", "time": time.strftime("%Y-%m-%d %H:%M:%S"), "command": cmd,
               "wall_s": res["wall_s"], "rc": res["rc"], "gpus": res["gpus"], "verdict": v, "config": {"command": cmd}}
        prev, how = previous(path, name)
        if prev is not None and prev.get("rc") == 0 and res["rc"] == 0:
            print(speedup_line(prev, rec, "this command" if name == cmd else how))
        if not a.no_record:
            _append(path, rec)
        results.append(rec)
    base = results[0]
    for k, rec in enumerate(results[1:], 2):
        if base["rc"] == 0 and rec["rc"] == 0 and rec["wall_s"] > 0:
            ratio = base["wall_s"] / rec["wall_s"]
            print(f"[bench] run {k} vs run 1: x{ratio:.2f} {'faster' if ratio >= 1 else 'slower'} (wall time "
                  f"{rec['wall_s']:.1f}s vs {base['wall_s']:.1f}s)")
        else:
            print(f"[bench] run {k} vs run 1: not compared (a run failed)")
    if not a.no_record:
        print(f"[bench] recorded in {path.as_posix()}")
    if any(r["rc"] != 0 for r in results):
        sys.exit(1)
