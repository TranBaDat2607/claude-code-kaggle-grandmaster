"""Post-processing optimisers: decision thresholds and ordinal rounding.

Label metrics (F1, MCC, QWK, accuracy on imbalanced data) are not optimised by
argmax/0.5. Train on a smooth objective, predict a continuous score, then choose the
cut-points on OOF predictions. Always fit thresholds on OOF (never on the fold you
report) — or nest: fit on folds != k, evaluate on fold k — to get an honest estimate.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from . import metrics as M


def best_threshold(
    y_true: Sequence[int],
    y_score: Sequence[float],
    metric: str | Callable = "f1",
    grid: int = 201,
) -> tuple[float, float]:
    """Grid-search a single binary threshold on scores. Returns (threshold, score)."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score, dtype=float)
    m = M.get(metric) if isinstance(metric, str) else None
    fn = m.fn if m else metric
    gib = m.greater_is_better if m else True
    qs = np.unique(np.quantile(y_score, np.linspace(0, 1, grid)))
    best_t, best_s = 0.5, -np.inf if gib else np.inf
    for t in qs:
        s = fn(y_true, (y_score >= t).astype(int))
        if (s > best_s) if gib else (s < best_s):
            best_t, best_s = float(t), float(s)
    return best_t, best_s


def per_class_thresholds(
    Y_true: np.ndarray,
    Y_score: np.ndarray,
    metric: str | Callable = "f1",
    grid: int = 101,
) -> np.ndarray:
    """Independent per-column thresholds for multilabel problems."""
    Y_true, Y_score = np.asarray(Y_true), np.asarray(Y_score)
    return np.array([best_threshold(Y_true[:, j], Y_score[:, j], metric, grid)[0] for j in range(Y_true.shape[1])])


def apply_ordinal_thresholds(pred: Sequence[float], thresholds: Sequence[float], labels: Sequence[int] | None = None) -> np.ndarray:
    """Map continuous predictions to ordinal classes using sorted cut points."""
    t = np.sort(np.asarray(thresholds, dtype=float))
    idx = np.searchsorted(t, np.asarray(pred, dtype=float), side="right")
    if labels is None:
        return idx
    return np.asarray(labels)[idx]


class OptimizedRounder:
    """Find ordinal cut-points that maximise a label metric (QWK by default).

    The classic Kaggle trick for QWK competitions (PetFinder, PANDA, APTOS...):
    regress on the ordinal target with RMSE, then optimise the K-1 thresholds.
    Coordinate-descent refinement after Nelder–Mead makes it robust to the flat,
    piecewise-constant objective.

        r = OptimizedRounder(labels=[0, 1, 2, 3, 4]).fit(oof_pred, y)
        test_labels = r.predict(test_pred)
    """

    def __init__(self, labels: Sequence[int] | None = None, metric: str = "qwk", n_passes: int = 3):
        self.labels = None if labels is None else np.asarray(labels)
        self.metric = M.get(metric)
        self.n_passes = n_passes
        self.thresholds_: np.ndarray | None = None

    def _loss(self, t, x, y):
        pred = apply_ordinal_thresholds(x, t, self.labels)
        s = self.metric(y, pred)
        return -s if self.metric.greater_is_better else s

    def fit(self, x: Sequence[float], y: Sequence[int]) -> "OptimizedRounder":
        x, y = np.asarray(x, dtype=float), np.asarray(y)
        if self.labels is None:
            self.labels = np.unique(y)
        init = (self.labels[:-1] + self.labels[1:]) / 2.0
        res = minimize(self._loss, init, args=(x, y), method="Nelder-Mead", options={"xatol": 1e-4, "maxiter": 2000})
        t = np.sort(res.x)
        # coordinate descent over candidate cut points
        cands = np.unique(np.quantile(x, np.linspace(0, 1, 400)))
        best = self._loss(t, x, y)
        for _ in range(self.n_passes):
            improved = False
            for i in range(len(t)):
                lo = t[i - 1] if i > 0 else -np.inf
                hi = t[i + 1] if i < len(t) - 1 else np.inf
                for c in cands[(cands > lo) & (cands < hi)]:
                    trial = t.copy()
                    trial[i] = c
                    loss = self._loss(trial, x, y)
                    if loss < best - 1e-12:
                        best, t, improved = loss, trial, True
            if not improved:
                break
        self.thresholds_ = t
        self.score_ = -best if self.metric.greater_is_better else best
        return self

    def predict(self, x: Sequence[float]) -> np.ndarray:
        if self.thresholds_ is None:
            raise RuntimeError("call fit first")
        return apply_ordinal_thresholds(x, self.thresholds_, self.labels)


def class_prior_shift(proba: np.ndarray, train_prior: Sequence[float], test_prior: Sequence[float]) -> np.ndarray:
    """Re-weight multiclass probabilities when the test class prior differs from train
    (Saerens et al. 2002 adjustment): p'(c|x) ∝ p(c|x) * pi_test(c) / pi_train(c)."""
    p = np.asarray(proba, dtype=float) * (np.asarray(test_prior) / np.asarray(train_prior))
    return p / p.sum(axis=1, keepdims=True)
