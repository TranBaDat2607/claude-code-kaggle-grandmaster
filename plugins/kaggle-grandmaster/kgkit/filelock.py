"""A lock file around the workspace's shared state files (``.kaggle-gm/ledger.jsonl``, ``backlog.json``).

Several processes write them at once: a training run logging its record, ``gpu collect`` importing remote records,
Claude running ``ledger decide`` or ``backlog set`` meanwhile, two experiments finishing together on two GPUs. A
record's id is the next number in the file, and a decision rewrites the whole file, so without a lock two runs get
the same number and a rewrite drops a record appended while it ran (#126).

    with locked(path):            # <path>.lock, created exclusively; re-entrant within one process
        ...read, change, write...

The lock is a file made with O_EXCL (atomic on every OS kgkit runs on). A lock older than ``STALE_S`` is taken to be
left by a process that died while holding it, and is taken over. Readers take no lock: an append adds whole lines,
and a rewrite replaces the file in one step (``replace``).

Standard library only: the hooks, the backlog and the remote runner import it.
"""

from __future__ import annotations

import contextlib
import os
import socket
import time
from pathlib import Path

STALE_S = 300.0     # a holder logs one record (git snapshot, arrays saved): seconds, never minutes
TIMEOUT_S = 360.0   # longer than STALE_S, so a waiter outlives a lock its dead holder left
_HELD: dict[str, int] = {}  # lock path -> depth, for this process


class LockTimeout(TimeoutError):
    """Another process held the lock for longer than the timeout."""


def lock_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_name(p.name + ".lock")


def _stale(lock: Path, stale_s: float) -> bool:
    try:
        return time.time() - lock.stat().st_mtime > stale_s
    except OSError:
        return False  # gone already: the next attempt creates it


@contextlib.contextmanager
def locked(path: str | Path, timeout: float = TIMEOUT_S, stale_s: float = STALE_S, poll: float = 0.05):
    """Hold ``<path>.lock`` while the block runs. Re-entrant within one process (a decision that updates a record
    takes it once). Raises ``LockTimeout`` after ``timeout`` seconds."""
    lock = lock_path(path)
    key = str(lock.resolve())
    if _HELD.get(key):
        _HELD[key] += 1
        try:
            yield lock
        finally:
            _HELD[key] -= 1
        return
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _stale(lock, stale_s):
                with contextlib.suppress(OSError):
                    lock.unlink()
                continue
            if time.time() > deadline:
                raise LockTimeout(f"{lock} is held by another process ({_holder(lock)}) for more than {timeout:g}s; "
                                  f"if no kgkit process is running, delete it")
            time.sleep(poll)
            continue
        except PermissionError:  # Windows: the file is being deleted by its holder right now
            if time.time() > deadline:
                raise LockTimeout(f"{lock} could not be created for {timeout:g}s")
            time.sleep(poll)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"pid {os.getpid()} on {socket.gethostname()} since {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        break
    _HELD[key] = 1
    try:
        yield lock
    finally:
        _HELD.pop(key, None)
        with contextlib.suppress(OSError):
            lock.unlink()


def _holder(lock: Path) -> str:
    try:
        return lock.read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        return "unknown"


def replace(tmp: str | Path, dest: str | Path, tries: int = 20, wait: float = 0.05) -> None:
    """``os.replace`` that retries: on Windows it fails while another process has ``dest`` open (a hook reading the
    ledger at that moment)."""
    for i in range(tries):
        try:
            os.replace(str(tmp), str(dest))
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(wait)
