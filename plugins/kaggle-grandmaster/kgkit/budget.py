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

Env vars (all optional; a ``TrainBudget`` argument of the same name wins over its variable)
  KG_DEADLINE        unix time by which the process must have saved its state and exited
  KG_RESERVE_S       seconds kept free after the last epoch for test inference + saving (default 600)
  KG_PRUNE_BASELINE  directory with the baseline's ``curve_fold<k>.json`` (or one curve file)
  KG_PRUNE_MARGIN    how far a score may trail the baseline's before the run is stopped
  KG_PRUNE_MIN_FRAC  never prune before this fraction of the schedule has run (default 0.3)
  KG_PRUNE_MODE      ``last`` (default) or ``best``: which score of the run counts, see ``should_stop``
  KG_PRUNE_PATIENCE  ``last`` mode: evaluations below the run's own peak before a decline counts (default 2)

Pruning (``should_stop``) compares the run's curve with the baseline's at the same fraction of the schedule. In both
modes it stops a run whose best score so far trails the baseline's best by more than the margin: an idea that is bad
from the start. ``last`` mode is for runs that keep their final epoch (the templates' ``select="last"``, #109): the
score that counts is then the last one, so it also stops a run whose current score trails the baseline's score at that
point by more than the margin while it has fallen below its own peak for ``patience`` evaluations in a row (#119: a
run that ties the baseline early and then declines). ``best`` mode, for runs that keep their best epoch, judges the
best score so far only. Stopping discards the run (status ``pruned``): it never chooses an epoch for a model that is
kept, so the OOF stays honest.

A hand-written loop gets the same rule from ``TrainBudget`` with explicit arguments (no runner needed)::

    tb = TrainBudget(out_dir, fold, total_epochs, greater_is_better, baseline="control/", margin=0.002, mode="last")

or from the pure function on the curve files themselves: ``update_curve`` adds a point to ``curve_fold<k>.json`` and
``should_stop(curve, load_curve(control_dir, fold), margin=...)`` returns why to stop, or None. Keep
``curve_fold<k>.json`` with the run's outputs and copy it back when a run resumes in a new session, so a run that spans
several sessions is judged on its whole curve (``gpu build --resume-from`` does this).

Standard library only: the remote runner and the hooks import this package without numpy. torch is used only when
the training script has already imported it and initialised ``torch.distributed``.
"""

from __future__ import annotations

import json
import os
import re
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


MODES = ("last", "best")


def _seen(curve: dict, frac: float) -> list[dict]:
    """The points of ``curve`` evaluated by fraction ``frac`` of its schedule, in epoch order."""
    total = max(1, int(curve.get("epochs") or len(curve["points"])))
    return sorted((pt for pt in curve["points"] if (pt["epoch"] + 1) / total <= frac + 1e-9), key=lambda p: p["epoch"])


def baseline_best_at(curve: dict, frac: float, greater_is_better: bool) -> float | None:
    """Best baseline score reached by fraction ``frac`` of its schedule (None if it had no eval yet)."""
    seen = [pt["score"] for pt in _seen(curve, frac)]
    if not seen:
        return None
    return max(seen) if greater_is_better else min(seen)


def baseline_score_at(curve: dict, frac: float) -> float | None:
    """The baseline's score at fraction ``frac`` of its schedule: its latest evaluation by then (None before its
    first)."""
    seen = _seen(curve, frac)
    return seen[-1]["score"] if seen else None


def _check_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode not in MODES:
        raise ValueError(f"prune mode {mode!r}: expected one of {', '.join(MODES)}")
    return mode


def should_stop(curve: dict, baseline: dict | None, epoch: int | None = None, *, margin: float | None,
                mode: str = "last", min_frac: float = 0.3, patience: int = 2,
                greater_is_better: bool | None = None) -> str | None:
    """Why the run whose learning curve is ``curve`` should stop after ``epoch`` (default: its latest point), or None
    to go on. ``curve`` and ``baseline`` have the shape of ``curve_fold<k>.json`` (``epochs``, ``greater_is_better``,
    ``points``: ``{"epoch", "score"}``); the two are compared at the same fraction of their own schedules.

    - Both modes: the best score so far trails the baseline's best by then by more than ``margin``.
    - ``last`` mode also: the latest score trails the baseline's score at that point by more than ``margin``, and the
      last ``patience`` evaluations are all below the run's own peak. For runs that keep their final epoch.

    Never before ``min_frac`` of the schedule, never at its last epoch (it is finished), and never without a
    baseline evaluation to compare with."""
    mode = _check_mode(mode)
    if baseline is None or margin is None or not curve or not curve.get("points"):
        return None
    gib = bool(curve.get("greater_is_better", True) if greater_is_better is None else greater_is_better)
    total = max(1, int(curve.get("epochs") or len(curve["points"])))
    epoch = max(pt["epoch"] for pt in curve["points"]) if epoch is None else int(epoch)
    pts = sorted((pt for pt in curve["points"] if pt["epoch"] <= epoch), key=lambda p: p["epoch"])
    frac = (epoch + 1) / total
    if not pts or frac < min_frac or epoch + 1 >= total:
        return None

    def trail(ref: float, score: float) -> float:
        return (ref - score) if gib else (score - ref)

    scores = [pt["score"] for pt in pts]
    best = max(scores) if gib else min(scores)
    at = f"{frac:.0%} of the schedule"
    ref_best = baseline_best_at(baseline, frac, gib)
    if ref_best is not None and trail(ref_best, best) > margin:
        return (f"best so far {best:.5f} trails the baseline's best {ref_best:.5f} at {at} by more than the margin "
                f"{margin:g}")
    if mode == "best":
        return None
    ref_now, now = baseline_score_at(baseline, frac), scores[-1]
    peak = max(i for i, s in enumerate(scores) if s == best)  # the latest evaluation that reached the peak
    below = len(scores) - 1 - peak
    if ref_now is not None and below >= max(1, int(patience)) and trail(ref_now, now) > margin:
        return (f"score {now:.5f} trails the baseline's {ref_now:.5f} at {at} by more than the margin {margin:g} and "
                f"has been below its own peak {best:.5f} for {below} evaluations: the last epoch, which is the one "
                "kept, is unlikely to catch up")
    return None


def update_curve(path: str | Path, epoch: int, score: float, total_epochs: int, greater_is_better: bool = True,
                 seconds: float | None = None, fold: int | None = None) -> dict:
    """Add the evaluation after ``epoch`` to the learning curve at ``path`` (a ``curve_fold<k>.json``; a resumed
    epoch replaces its point and every later one) and return the curve. The file a hand-written loop keeps for
    ``should_stop`` and for later runs to be pruned against: the same one ``TrainBudget`` writes."""
    p = Path(path)
    data = None
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
    points = [pt for pt in (data or {}).get("points", []) if pt["epoch"] < int(epoch)]
    points.append({"epoch": int(epoch), "score": float(score), "seconds": None if seconds is None else round(seconds, 3)})
    m = re.match(r"curve_fold(\d+)\.json$", p.name)
    curve = {"fold": fold if fold is not None else (int(m.group(1)) if m else (data or {}).get("fold")),
             "epochs": max(1, int(total_epochs)), "greater_is_better": bool(greater_is_better), "points": points}
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(curve), encoding="utf-8")
    tmp.replace(p)
    return curve


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
    """Session deadline, learning curve and pruning for one fold (module docstring). ``baseline`` (a directory of
    ``curve_fold<k>.json`` or one curve file), ``margin``, ``min_frac``, ``mode`` (``last`` | ``best``) and
    ``patience`` override their ``KG_PRUNE_*`` variables, so a hand-written loop can prune without the remote
    runner. ``mode`` should say which epoch the loop keeps: ``last`` for the final one, ``best`` for the best-scoring
    one."""

    def __init__(self, out_dir: str | Path, fold: int, total_epochs: int, greater_is_better: bool = True,
                 env=None, dist=_LOCAL, *, baseline: str | Path | None = None, margin: float | None = None,
                 min_frac: float | None = None, mode: str | None = None, patience: int | None = None):
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
        self.margin = float(margin) if margin is not None else _envf(env, "KG_PRUNE_MARGIN")
        self.min_frac = float(min_frac) if min_frac is not None else _envf(env, "KG_PRUNE_MIN_FRAC", 0.3)
        self.mode = _check_mode(mode if mode is not None else (env.get("KG_PRUNE_MODE") or "last"))
        self.patience = int(patience if patience is not None else _envf(env, "KG_PRUNE_PATIENCE", 2.0))
        base = baseline if baseline is not None else env.get("KG_PRUNE_BASELINE")
        self.baseline = load_curve(base, self.fold) if base else None
        self.prune_reason: str | None = None
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

    def curve(self) -> dict:
        """The learning curve so far, in the shape of ``curve_fold<k>.json``."""
        return {"fold": self.fold, "epochs": self.total_epochs, "greater_is_better": self.gib, "points": self.points}

    def should_prune(self, epoch: int) -> bool:
        """Whether ``should_stop`` stops the run after ``epoch``; its reason is kept in ``prune_reason``."""
        self.prune_reason = should_stop(self.curve(), self.baseline, epoch, margin=self.margin, mode=self.mode,
                                        min_frac=self.min_frac, patience=self.patience, greater_is_better=self.gib)
        return self.prune_reason is not None

    # ------------------------------------------------------------------ files
    def _save_curve(self) -> None:
        if not self.is_main:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.curve_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.curve()), encoding="utf-8")
        tmp.replace(self.curve_path)

    def write_status(self, status: str, **info) -> Path:
        """``status``: done | timeout | pruned | failed. Read by the remote runner and ``kgkit gpu collect``. A
        pruned fold also records why (``reason``) and the prune mode. Under DDP only rank 0 writes it (the path is
        returned on every rank)."""
        p = self.out_dir / f"fold{self.fold}_status.json"
        if not self.is_main:
            return p
        self.out_dir.mkdir(parents=True, exist_ok=True)
        rec = {"fold": self.fold, "status": status, "best": self.best, "epochs_run": len(self.points),
               "epoch_seconds": self.epoch_seconds(), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if status == "pruned":
            rec.update(prune_mode=self.mode, reason=self.prune_reason)
        rec.update(info)
        p.write_text(json.dumps(rec), encoding="utf-8")
        return p
