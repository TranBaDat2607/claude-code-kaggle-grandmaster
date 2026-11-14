"""Per-fold GPU budget guard for training loops: session deadline, learning-curve logging and pruning.

Kaggle GPU time is a weekly quota, so a training loop should (1) never be killed mid-epoch by the
session limit with nothing saved, and (2) stop early when it is clearly losing to the baseline.
The remote runner (``kgkit/kaggle_gpu_runner.py``) passes the budget through env vars; locally
nothing is set and the guard only records the learning curve::

    from kgkit.budget import TrainBudget
    tb = TrainBudget(out_dir, fold, total_epochs=cfg["epochs"], greater_is_better=metric.greater_is_better)
    for epoch in range(start_epoch, cfg["epochs"]):
        tb.start_epoch()
        ...train, validate -> score...
        action = tb.end_epoch(epoch, score)     # "continue" | "timeout" | "prune"
        if action == "timeout": save a resume checkpoint, tb.write_status("timeout", ...); stop
        if action == "prune":   tb.write_status("pruned", ...); stop

Under DDP (``torchrun``, ``torch.distributed`` initialised with more than one rank) every rank builds its own
``TrainBudget`` and calls ``end_epoch`` each epoch; the ranks get the same answer, so none waits forever for another
in the next collective: "timeout" when any rank is short of time, "prune" as rank 0 decides from its own score
(validate on rank 0, or pass every rank the all-reduced score). Only rank 0 (``tb.is_main``) writes the curve and the
status file; gate your own checkpoints and OOF files on ``tb.is_main`` too.

Env vars (all optional)
  KG_DEADLINE        unix time by which the process must have saved its state and exited
  KG_RESERVE_S       seconds kept free after the last epoch for test inference + saving (default 600)
  KG_PRUNE_BASELINE  directory with the baseline's ``curve_fold<k>.json`` (or one curve file)
  KG_PRUNE_MARGIN    prune when the best-so-far score trails the baseline's by more than this
  KG_PRUNE_MIN_FRAC  never prune before this fraction of the schedule has run (default 0.3)

Standard library only: the remote runner and the hooks import this package without numpy. torch is used only when
the training script has already imported it and initialised ``torch.distributed``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _envf(env, key: str, default: float | None = None) -> float | None:
    try:
        v = env.get(key)
        return float(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


def load_curve(path: str | Path, fold: int) -> dict | None:
    """Baseline curve for ``fold`` from a directory of ``curve_fold<k>.json`` files or a single file."""
    p = Path(path)
    if p.is_dir():
        p = p / f"curve_fold{fold}.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if data.get("points") else None


def baseline_best_at(curve: dict, frac: float, greater_is_better: bool) -> float | None:
    """Best baseline score reached by fraction ``frac`` of its schedule (None if it had no eval yet)."""
    total = max(1, int(curve.get("epochs") or len(curve["points"])))
    seen = [pt["score"] for pt in curve["points"] if (pt["epoch"] + 1) / total <= frac + 1e-9]
    if not seen:
        return None
    return max(seen) if greater_is_better else min(seen)


class _Dist:
    """The running ``torch.distributed`` group, when the script started one with more than one rank."""

    def __init__(self, torch, rank: int, world: int):
        self.torch, self.rank, self.world = torch, rank, world

    def max(self, values: list[float]) -> list[float]:
        """Element-wise maximum over the ranks (a collective: every rank must call it)."""
        torch, dist = self.torch, self.torch.distributed
        dev = (torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl"
               else torch.device("cpu"))
        t = torch.tensor([float(v) for v in values], dtype=torch.float64, device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return [float(v) for v in t.tolist()]


def distributed() -> _Dist | None:
    """The DDP group this process is in, or None (no torch imported, not initialised, or a single rank). Never
    imports torch itself: a script that runs DDP has imported it already."""
    torch = sys.modules.get("torch")
    try:
        dist = torch.distributed if torch is not None else None
        if dist is None or not (dist.is_available() and dist.is_initialized()):
            return None
        world = int(dist.get_world_size())
        return _Dist(torch, int(dist.get_rank()), world) if world > 1 else None
    except Exception:
        return None


_LOCAL = object()


class TrainBudget:
    def __init__(self, out_dir: str | Path, fold: int, total_epochs: int, greater_is_better: bool = True,
                 env=None, dist=_LOCAL):
        env = os.environ if env is None else env
        self.dist = distributed() if dist is _LOCAL else dist
        self.rank = self.dist.rank if self.dist is not None else 0
        self.is_main = self.rank == 0
        self.out_dir = Path(out_dir)
        self.fold = int(fold)
        self.total_epochs = max(1, int(total_epochs))
        self.gib = bool(greater_is_better)
        self.deadline = _envf(env, "KG_DEADLINE")
        self.reserve_s = _envf(env, "KG_RESERVE_S", 600.0)
        self.margin = _envf(env, "KG_PRUNE_MARGIN")
        self.min_frac = _envf(env, "KG_PRUNE_MIN_FRAC", 0.3)
        base = env.get("KG_PRUNE_BASELINE")
        self.baseline = load_curve(base, self.fold) if base else None
        self.points: list[dict] = []
        self.best: float | None = None
        self._t: float | None = None
        self.curve_path = self.out_dir / f"curve_fold{self.fold}.json"
        if self.curve_path.is_file():  # resumed fold: keep the curve recorded by the previous session
            try:
                self.points = json.loads(self.curve_path.read_text(encoding="utf-8")).get("points", [])
                scores = [pt["score"] for pt in self.points]
                self.best = (max(scores) if self.gib else min(scores)) if scores else None
            except (OSError, ValueError):
                self.points = []

    # ------------------------------------------------------------------ timing
    def start_epoch(self) -> None:
        self._t = time.time()

    def epoch_seconds(self) -> float | None:
        secs = [pt["seconds"] for pt in self.points if pt.get("seconds") is not None]
        return sum(secs) / len(secs) if secs else None

    def seconds_left(self) -> float | None:
        return None if self.deadline is None else self.deadline - time.time()

    def _better(self, a: float, b: float) -> bool:
        return a > b if self.gib else a < b

    # ------------------------------------------------------------------ decisions
    def end_epoch(self, epoch: int, score: float) -> str:
        secs = time.time() - self._t if self._t is not None else None
        score = float(score)
        self.points = [pt for pt in self.points if pt["epoch"] < epoch]  # a resumed epoch overwrites its point
        self.points.append({"epoch": int(epoch), "score": score, "seconds": None if secs is None else round(secs, 3)})
        self.best = score if self.best is None or self._better(score, self.best) else self.best
        self._save_curve()
        if epoch + 1 >= self.total_epochs:
            return "continue"
        prune, timeout = self.should_prune(epoch), False
        left, per_epoch = self.seconds_left(), self.epoch_seconds()
        if left is not None and per_epoch is not None:
            last = epoch + 2 >= self.total_epochs  # the next epoch is the final one: test inference follows it
            need = per_epoch * 1.15 + (self.reserve_s if last else 60.0)
            timeout = left < need
        if self.dist is not None:  # every rank must take the same branch, or the others hang in the next collective
            timeout_any, prune_main = self.dist.max([float(timeout), float(prune and self.is_main)])
            timeout, prune = timeout_any > 0, prune_main > 0
        if prune:
            return "prune"
        return "timeout" if timeout else "continue"

    def should_prune(self, epoch: int) -> bool:
        if self.baseline is None or self.margin is None or self.best is None:
            return False
        frac = (epoch + 1) / self.total_epochs
        if frac < self.min_frac:
            return False
        ref = baseline_best_at(self.baseline, frac, self.gib)
        if ref is None:
            return False
        gap = (ref - self.best) if self.gib else (self.best - ref)
        return gap > self.margin

    # ------------------------------------------------------------------ files
    def _save_curve(self) -> None:
        if not self.is_main:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        data = {"fold": self.fold, "epochs": self.total_epochs, "greater_is_better": self.gib, "points": self.points}
        tmp = self.curve_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.curve_path)

    def write_status(self, status: str, **info) -> Path:
        """``status``: done | timeout | pruned | failed. Read by the remote runner and ``kgkit gpu collect``. Under
        DDP only rank 0 writes it (the path is returned on every rank)."""
        p = self.out_dir / f"fold{self.fold}_status.json"
        if not self.is_main:
            return p
        self.out_dir.mkdir(parents=True, exist_ok=True)
        rec = {"fold": self.fold, "status": status, "best": self.best, "epochs_run": len(self.points),
               "epoch_seconds": self.epoch_seconds(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **info}
        p.write_text(json.dumps(rec), encoding="utf-8")
        return p
