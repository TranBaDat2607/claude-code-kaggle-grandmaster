"""Live resource readings for a Kaggle notebook or script that kgkit did not build.

Paste this as the first cell (or the top of the script). Every KG_MONITOR_S seconds (default 30) it prints one
`[kgmon] {...}` line to the session log: CPU, RAM, disk and each GPU of the session, what the code says it is
doing, and when its clock started (t0, Unix seconds: the pane counts the time to the session limit from it). The kaggle-monitor pane in Claude Code follows a running kernel's log (`kaggle kernels logs -f`) and
draws the latest line; Kaggle's API reports no session resources of its own. Kernels built by `kgkit gpu build`
print these lines already; `kgkit gpu push <folder>`, and the plugin's hook when Claude Code runs
`kaggle kernels push`, add this block to the kernel's .py or .ipynb itself, between kaggle-grandmaster markers (kept
up to date on later pushes). A code file that mentions kgmon anywhere else is left alone, so a `# kgmon: off`
comment keeps it out.

It also checks the machine (#113): Kaggle sometimes starts a 2x T4 push on a machine with one GPU, or none. When
nvidia-smi lists fewer GPUs than KGMON_EXPECT_GPUS (filled in by kgkit from the pushed machine shape), it prints the
reading and a KG_BAD_MACHINE line and stops the session, before it spends the quota; pushing again usually gets a
healthy machine. KGMON_EXPECT_GPUS = 0, or the environment variable KG_EXPECT_GPUS=0, turns the check off.
The check never stops a code competition's scoring run (#121): a slower machine there still gives a valid submission,
while an exception fails it. kgkit writes KGMON_EXPECT_GPUS = 0 into a kernel that looks like a submission
(competition sources, internet off), and the check also stands down while KAGGLE_IS_COMPETITION_RERUN is set.

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
# The GPUs the pushed machine shape promises (2 for NvidiaTeslaT4, 1 for NvidiaTeslaP100); None or 0: no check.
KGMON_EXPECT_GPUS = None
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


def _kgmon_want():
    """The GPUs this session should have: KG_EXPECT_GPUS, else KGMON_EXPECT_GPUS; None when unknown or 0."""
    raw = _kgmon_os.environ.get("KG_EXPECT_GPUS")
    try:
        want = int(raw) if raw not in (None, "") else int(KGMON_EXPECT_GPUS or 0)
    except ValueError:
        return None
    return want or None


def _kgmon_gpus():
    fake = _kgmon_os.environ.get("KG_FAKE_GPUS")  # tests: this many GPUs, without nvidia-smi
    if fake is not None:
        return [[i, 0, 0, 0, "fake-gpu"] for i in range(int(fake))]
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
    want = _kgmon_want()
    if want:
        snap["want"] = want
    if _KGMON_DOING:
        snap["doing"] = list(_KGMON_DOING)
    return snap, (now, cpu_us)


def _kgmon_print(snap):
    _kgmon_sys.stdout.write("[kgmon] " + _kgmon_json.dumps(snap, separators=(",", ":")) + "\n")
    _kgmon_sys.stdout.flush()


class KgmonBadMachine(RuntimeError):
    """The session's machine has fewer GPUs than its shape promises."""


def kgmon_check():
    """Stops the session when nvidia-smi lists fewer GPUs than it should have (`_kgmon_want`), after printing the
    reading and a KG_BAD_MACHINE line: `kgkit gpu wait` and the pane read both. An exception, not sys.exit, so a
    notebook stops at this cell too."""
    want = _kgmon_want()
    if not want or _kgmon_os.environ.get("KAGGLE_IS_COMPETITION_RERUN"):
        return  # a scoring rerun on a slower machine still gives a valid submission; an exception fails it (#121)
    snap, _ = _kgmon_snapshot(None)
    found = len(snap.get("gpus") or [])
    if found >= want:
        return
    _kgmon_print(snap)
    msg = ("KG_BAD_MACHINE: this machine shape promises %d GPUs, nvidia-smi lists %d. Kaggle sometimes starts a "
           "session on a machine with fewer GPUs than its shape; stopping now, before it spends the quota. Push it "
           "again (`kgkit gpu repush <kernel> --yes` deletes the kernel and pushes it again: ask the user first), or "
           "set KGMON_EXPECT_GPUS = 0 to run on this machine anyway." % (want, found))
    _kgmon_sys.stdout.write(msg + "\n")
    _kgmon_sys.stdout.flush()
    raise KgmonBadMachine(msg)


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
    kgmon_check()
    start_kgmon()
