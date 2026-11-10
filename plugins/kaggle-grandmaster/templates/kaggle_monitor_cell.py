"""Live resource readings for a Kaggle notebook or script that kgkit did not build.

Paste this as the first cell (or the top of the script). Every KG_MONITOR_S seconds (default 30) it prints one
`[kgmon] {...}` line to the session log: CPU, RAM, disk and each GPU of the session, what the code says it is
doing, and when its clock started (t0, Unix seconds: the pane counts the time to the session limit from it). The kaggle-monitor pane in Claude Code follows a running kernel's log (`kaggle kernels logs -f`) and
draws the latest line; Kaggle's API reports no session resources of its own. Kernels pushed with
`kgkit gpu push` print these lines already, and when Claude Code runs `kaggle kernels push` the plugin's hook adds
this block to the kernel's .py or .ipynb itself. A code file that mentions kgmon anywhere is left alone, so a
`# kgmon: off` comment keeps it out.

Say what the code is doing, and the pane shows it beside the readings:

    kgmon_doing("fold 2: epoch 3/10")

Standard library only. Prints ASCII only. Only batch runs (Save Version / `kaggle kernels push`) have a log the
API can stream; an interactive editor session has none.
"""

# Private names, so code that later rebinds `time`, `json` or `os` (from time import time) does not break it.
import json as _kgmon_json
import os as _kgmon_os
import shutil as _kgmon_shutil
import subprocess as _kgmon_subprocess
import sys as _kgmon_sys
import threading as _kgmon_threading
import time as _kgmon_time

KGMON_T0 = _kgmon_time.time()
_KGMON_DOING = []
_KGMON_STOP = _kgmon_threading.Event()


def kgmon_doing(text=None):
    """What the code is doing now, shown in the pane; None clears it."""
    _KGMON_DOING[:] = [] if text is None else [str(text).encode("ascii", "replace").decode("ascii")[:120]]


def _kgmon_read(path):
    try:
        with open(path, encoding="ascii", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def _kgmon_cpu_us():
    stat = _kgmon_read("/sys/fs/cgroup/cpu.stat")
    if stat:
        for line in stat.splitlines():
            if line.startswith("usage_usec "):
                return int(line.split()[1])
    v1 = _kgmon_read("/sys/fs/cgroup/cpuacct/cpuacct.usage") or _kgmon_read("/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage")
    return int(v1) // 1000 if v1 and v1.isdigit() else None


def _kgmon_cores():
    quota = (_kgmon_read("/sys/fs/cgroup/cpu.max") or "").split()
    if len(quota) == 2 and quota[0].isdigit() and int(quota[1] or 0) > 0:
        return max(1.0, int(quota[0]) / int(quota[1]))
    try:
        return float(len(_kgmon_os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return float(_kgmon_os.cpu_count() or 1)


def _kgmon_ram_mb():
    used = _kgmon_read("/sys/fs/cgroup/memory.current") or _kgmon_read("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    limit = _kgmon_read("/sys/fs/cgroup/memory.max") or _kgmon_read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    mem = {}
    for line in (_kgmon_read("/proc/meminfo") or "").splitlines():
        k, _, v = line.partition(":")
        if v.split():
            mem[k] = int(v.split()[0]) * 1024
    host = mem.get("MemTotal")
    if used and used.isdigit() and limit and limit.isdigit() and (host is None or int(limit) < host):
        return [int(used) // 2 ** 20, int(limit) // 2 ** 20]
    if host and "MemAvailable" in mem:
        return [(host - mem["MemAvailable"]) // 2 ** 20, host // 2 ** 20]
    return None


def _kgmon_disk_mb(path):
    try:
        du = _kgmon_shutil.disk_usage(path)
        return [du.used // 2 ** 20, du.total // 2 ** 20]
    except OSError:
        return None


def _kgmon_gpus():
    try:
        out = _kgmon_subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,name",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, _kgmon_subprocess.SubprocessError):
        return None
    rows = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5 and p[0].isdigit():
            try:
                rows.append([int(p[0]), int(float(p[1])), int(float(p[2])), int(float(p[3])),
                             p[4].encode("ascii", "replace").decode("ascii")])
            except ValueError:
                continue
    return rows


def _kgmon_snapshot(prev):
    now, cpu_us = _kgmon_time.time(), _kgmon_cpu_us()
    snap = {"v": 1, "t": int(now - KGMON_T0), "t0": int(KGMON_T0), "cores": round(_kgmon_cores(), 1)}
    if cpu_us is not None and prev and prev[1] is not None and now > prev[0]:
        snap["cpu"] = round(100.0 * (cpu_us - prev[1]) / 1e6 / (now - prev[0]) / snap["cores"], 1)
    ram = _kgmon_ram_mb()
    if ram:
        snap["ram"] = ram
    disk = _kgmon_disk_mb(_kgmon_os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
    if disk:
        snap["disk"] = disk
    tmp = _kgmon_disk_mb("/tmp")
    if tmp:
        snap["tmp"] = tmp
    gpus = _kgmon_gpus()
    if gpus is not None:
        snap["gpus"] = gpus
    if _KGMON_DOING:
        snap["doing"] = list(_KGMON_DOING)
    return snap, (now, cpu_us)


def _kgmon_print(snap):
    _kgmon_sys.stdout.write("[kgmon] " + _kgmon_json.dumps(snap, separators=(",", ":")) + "\n")
    _kgmon_sys.stdout.flush()


def start_kgmon(interval=None):
    """Prints a reading now, then every `interval` seconds from a background thread; 0 turns it off."""
    interval = float(_kgmon_os.environ.get("KG_MONITOR_S", "30") if interval is None else interval)
    if interval <= 0:
        return None
    snap, prev = _kgmon_snapshot(None)
    _kgmon_print(snap)

    def loop(prev=prev):
        while not _KGMON_STOP.wait(interval):
            try:
                snap, prev = _kgmon_snapshot(prev)
                _kgmon_print(snap)
            except Exception:  # a reading must never stop the code it reports on
                pass

    t = _kgmon_threading.Thread(target=loop, name="kgmon", daemon=True)
    t.start()
    return t


if __name__ == "__main__":  # not again in worker processes that re-import the script (spawn)
    start_kgmon()
