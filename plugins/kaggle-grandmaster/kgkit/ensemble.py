"""Ensembling from out-of-fold predictions.

Rules of thumb that win competitions:
  * Only blend models whose OOF predictions were produced on the *same* folds.
  * Prefer diverse models (different algorithms, features, seeds) — check
    :func:`oof_correlation`; > 0.98 correlated models add little.
  * Hill climbing (Caruana ensemble selection) is the robust default. Constrained
    weight optimisation is fine with few models. Stacking needs nested CV discipline.
  * Large libraries (100s of OOFs): :func:`prune_library` first, then hill climbing or a
    :func:`multi_level_stack`; :func:`residual_stack` lets a stage-2 model fix one strong model's errors.
  * For AUC / ranking metrics blend ranks, not raw probabilities, when models are
    calibrated differently.
  * Weights fit on full OOF are slightly optimistic; use :func:`cv_blend_score` to
    estimate the honest blended score.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

from . import metrics as M


def _metric(metric: str | Callable, greater_is_better: bool | None):
    if isinstance(metric, str):
        m = M.get(metric)
        return m.fn, m.greater_is_better
    if greater_is_better is None:
        raise ValueError("pass greater_is_better when metric is a callable")
    return metric, greater_is_better


def _stack(preds: Mapping[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    names = list(preds)
    arrs = [np.asarray(preds[n], dtype=float) for n in names]
    return names, np.stack(arrs, axis=0)  # (n_models, n_rows[, n_classes])


def rank_normalize(p: np.ndarray) -> np.ndarray:
    """Map scores to (0, 1] ranks column-wise — use before blending for AUC-type metrics."""
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        return rankdata(p) / len(p)
    return np.column_stack([rankdata(p[:, j]) / len(p) for j in range(p.shape[1])])


def blend(preds: Mapping[str, np.ndarray], weights: Mapping[str, float], method: str = "mean") -> np.ndarray:
    """Weighted blend. method: mean | rank | geometric | power<k> (e.g. 'power2')."""
    names = [n for n in weights if weights[n] != 0]
    w = np.array([weights[n] for n in names], dtype=float)
    arrs = [np.asarray(preds[n], dtype=float) for n in names]
    if method == "rank":
        arrs = [rank_normalize(a) for a in arrs]
    if method == "geometric":
        logs = sum(wi * np.log(np.clip(a, 1e-15, None)) for wi, a in zip(w, arrs))
        return np.exp(logs / w.sum())
    if method.startswith("power"):
        k = float(method[5:] or 2)
        return (sum(wi * np.power(np.clip(a, 0, None), k) for wi, a in zip(w, arrs)) / w.sum()) ** (1 / k)
    return sum(wi * a for wi, a in zip(w, arrs)) / w.sum()


def hill_climb(
    oofs: Mapping[str, np.ndarray],
    y_true,
    metric: str | Callable,
    greater_is_better: bool | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    allow_negative: bool = False,
    neg_steps: tuple[float, ...] = (-0.5, -0.25, -0.1),
    init_k: int = 1,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    """Caruana-style greedy ensemble selection *with replacement*.

    Starts from the best ``init_k`` single models, then repeatedly adds the model whose
    inclusion most improves the metric (a model may be added many times — that is how
    weights emerge). With ``allow_negative`` it also tries subtracting models, which
    sometimes squeezes extra score from highly correlated models — use with care.

    ``transform`` maps the blended prediction before scoring (e.g. argmax for accuracy,
    an OptimizedRounder for QWK).

    Returns {"weights": {name: w}, "score": float, "history": [(step, name, score)]}.
    """
    fn, gib = _metric(metric, greater_is_better)
    y_true = np.asarray(y_true)
    names, P = _stack(oofs)
    tf = transform or (lambda x: x)

    def sc(pred):
        return fn(y_true, tf(pred))

    def better(a, b):
        return a > b + tol if gib else a < b - tol

    singles = [sc(P[i]) for i in range(len(names))]
    order = np.argsort(singles)[::-1] if gib else np.argsort(singles)
    counts = np.zeros(len(names))
    for i in order[:init_k]:
        counts[i] += 1
    current = np.tensordot(counts, P, axes=1) / counts.sum()
    best_score = sc(current)
    history = [(0, ",".join(names[i] for i in order[:init_k]), best_score)]

    steps = [1.0] + (list(neg_steps) if allow_negative else [])
    for it in range(1, max_iter + 1):
        cand_best, cand = best_score, None
        total = counts.sum()
        floor = -0.5 * total if allow_negative else 0.0  # most negative count a model may reach
        for i in range(len(names)):
            for s in steps:
                new_total = total + s
                if new_total <= 0 or counts[i] + s < floor:
                    continue
                trial = (current * total + s * P[i]) / new_total
                v = sc(trial)
                if better(v, cand_best):
                    cand_best, cand = v, (i, s)
        if cand is None:
            break
        i, s = cand
        current = (current * counts.sum() + s * P[i]) / (counts.sum() + s)
        counts[i] += s
        best_score = cand_best
        history.append((it, names[i] if s > 0 else f"-{names[i]}", best_score))

    w = counts / counts.sum()
    return {
        "weights": {n: float(wi) for n, wi in zip(names, w) if wi != 0},
        "score": float(best_score),
        "single_scores": {n: float(s) for n, s in zip(names, singles)},
        "history": history,
    }


def optimize_weights(
    oofs: Mapping[str, np.ndarray],
    y_true,
    metric: str | Callable,
    greater_is_better: bool | None = None,
    nonneg: bool = True,
    n_restarts: int = 5,
    seed: int = 0,
) -> dict:
    """Weights on the simplex (sum to 1, optionally >= 0) via SLSQP with random restarts.

    Works best with smooth metrics (logloss, RMSE). For AUC/QWK prefer hill_climb.
    """
    fn, gib = _metric(metric, greater_is_better)
    y_true = np.asarray(y_true)
    names, P = _stack(oofs)
    k = len(names)

    def loss(w):
        pred = np.tensordot(w, P, axes=1)
        v = fn(y_true, pred)
        return -v if gib else v

    rng = np.random.default_rng(seed)
    bounds = [(0, 1)] * k if nonneg else [(-1, 2)] * k
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    best = None
    for r in range(n_restarts):
        w0 = np.full(k, 1 / k) if r == 0 else rng.dirichlet(np.ones(k))
        res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons)
        if best is None or res.fun < best.fun:
            best = res
    w = best.x
    score = -best.fun if gib else best.fun
    return {"weights": {n: float(wi) for n, wi in zip(names, w)}, "score": float(score)}


def cv_blend_score(
    oofs: Mapping[str, np.ndarray],
    y_true,
    folds,
    metric: str,
    method: str = "hill_climb",
    **kw,
) -> dict:
    """Honest estimate of a blend: fit weights on folds != k, score on fold k."""
    y_true, folds = np.asarray(y_true), np.asarray(folds)
    m = M.get(metric)
    fitter = hill_climb if method == "hill_climb" else optimize_weights
    blended = np.zeros_like(np.asarray(next(iter(oofs.values())), dtype=float))
    per_fold = []
    for f in sorted(set(folds[folds >= 0].tolist())):
        tr, va = folds != f, folds == f
        res = fitter({n: np.asarray(p)[tr] for n, p in oofs.items()}, y_true[tr], metric, **kw)
        pred_va = blend({n: np.asarray(p)[va] for n, p in oofs.items()}, res["weights"])
        blended[va] = pred_va
        per_fold.append(m(y_true[va], pred_va))
    return {"oof_score": m(y_true[folds >= 0], blended[folds >= 0]), "fold_scores": per_fold, "oof": blended}


def oof_correlation(oofs: Mapping[str, np.ndarray], method: str = "pearson") -> pd.DataFrame:
    """Correlation matrix between model OOFs (flattened). Low correlation = useful diversity."""
    df = pd.DataFrame({n: np.asarray(p, dtype=float).ravel() for n, p in oofs.items()})
    return df.corr(method=method)


def stack(
    oofs: Mapping[str, np.ndarray],
    tests: Mapping[str, np.ndarray],
    y_true,
    folds,
    meta_model=None,
    task: str = "auto",
    use_ranks: bool = False,
    X_extra: np.ndarray | None = None,
    X_extra_test: np.ndarray | None = None,
) -> dict:
    """Level-2 stacking on OOF predictions using the *same* folds.

    ``X_extra`` / ``X_extra_test``: a few raw features appended to the meta inputs, so the
    meta-model can learn *where* each model is trustworthy.

    The meta-model is refit per fold (meta-OOF for honest scoring) and test meta
    predictions are averaged over folds. Defaults: Ridge for regression,
    LogisticRegression (C=1) for classification. Keep the meta-model simple.
    """
    from sklearn.linear_model import LogisticRegression, Ridge

    y_true, folds = np.asarray(y_true), np.asarray(folds)
    names = list(oofs)

    def mat(d):
        cols = []
        for n in names:
            a = np.asarray(d[n], dtype=float)
            a = a.reshape(len(a), -1)
            cols.append(rank_normalize(a) if use_ranks else a)
        return np.hstack(cols)

    X, Xt = mat(oofs), mat(tests)
    if X_extra is not None:
        X = np.hstack([X, np.asarray(X_extra, dtype=float).reshape(len(X), -1)])
        Xt = np.hstack([Xt, np.asarray(X_extra_test, dtype=float).reshape(len(Xt), -1)])
    if task == "auto":
        task = "classification" if len(np.unique(y_true)) <= 20 and np.allclose(y_true % 1, 0) else "regression"
    if meta_model is None:
        meta_model = LogisticRegression(C=1.0, max_iter=2000) if task == "classification" else Ridge(alpha=1.0)

    from sklearn.base import clone

    n_classes = len(np.unique(y_true)) if task == "classification" else 1
    shape = (len(y_true),) if n_classes <= 2 else (len(y_true), n_classes)
    meta_oof = np.zeros(shape)
    meta_test = np.zeros((len(Xt),) if n_classes <= 2 else (len(Xt), n_classes))
    fold_ids = sorted(set(folds[folds >= 0].tolist()))
    for f in fold_ids:
        tr, va = folds != f, folds == f
        mdl = clone(meta_model).fit(X[tr], y_true[tr])
        if task == "classification":
            pv, pt = mdl.predict_proba(X[va]), mdl.predict_proba(Xt)
            if n_classes <= 2:
                pv, pt = pv[:, 1], pt[:, 1]
        else:
            pv, pt = mdl.predict(X[va]), mdl.predict(Xt)
        meta_oof[va] = pv
        meta_test += pt / len(fold_ids)
    return {"oof": meta_oof, "test": meta_test, "task": task}


def prune_library(
    oofs: Mapping[str, np.ndarray],
    y_true,
    metric: str | Callable,
    greater_is_better: bool | None = None,
    max_models: int = 40,
    corr_threshold: float = 0.995,
) -> dict:
    """Shrink a library of hundreds of OOFs to a blendable set before hill climbing or stacking.

    Models are visited best-first; a model is dropped when its OOF correlates above
    ``corr_threshold`` with an already kept (better) model — a near-duplicate adds weight-fitting
    noise, not information. Stops at ``max_models``. Returns {"kept": [...], "dropped": {name: reason}}.
    """
    fn, gib = _metric(metric, greater_is_better)
    y_true = np.asarray(y_true)
    singles = {n: fn(y_true, np.asarray(p)) for n, p in oofs.items()}
    order = sorted(singles, key=singles.get, reverse=gib)
    flat = {n: np.asarray(oofs[n], dtype=float).ravel() for n in order}
    kept, dropped = [], {}
    for n in order:
        if len(kept) >= max_models:
            dropped[n] = f"beyond max_models={max_models}"
            continue
        twin = next((k for k in kept if np.corrcoef(flat[n], flat[k])[0, 1] > corr_threshold), None)
        if twin is not None:
            dropped[n] = f"corr > {corr_threshold} with better model {twin}"
            continue
        kept.append(n)
    return {"kept": kept, "dropped": dropped, "single_scores": {n: float(s) for n, s in singles.items()}}


def _default_meta_models(task: str) -> list[tuple[str, object]]:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    if task == "classification":
        return [("lin", LogisticRegression(C=1.0, max_iter=2000)),
                ("gbm", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                       l2_regularization=1.0, random_state=0))]
    return [("lin", Ridge(alpha=1.0)),
            ("gbm", HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                                  l2_regularization=1.0, random_state=0))]


def _infer_task(y_true) -> str:
    y = np.asarray(y_true)
    return "classification" if len(np.unique(y)) <= 20 and np.allclose(y % 1, 0) else "regression"


def multi_level_stack(
    oofs: Mapping[str, np.ndarray],
    tests: Mapping[str, np.ndarray],
    y_true,
    folds,
    metric: str,
    n_layers: int = 1,
    layers: list[list[tuple[str, object]]] | None = None,
    passthrough: bool = False,
    final: str = "hill_climb",
    task: str = "auto",
    use_ranks: bool = False,
) -> dict:
    """Stack of meta-model layers on the same folds, finished by a hill-climbed (or mean) blend.

    Layer 1 sees the level-1 OOFs; each further layer sees the previous layer's meta-OOFs (plus the
    level-1 OOFs with ``passthrough``). Default layer = [linear, shallow GBDT] — two views that
    disagree usefully. This is the Playground-style deep stack (e.g. 3-4 levels over 100+ models);
    every layer adds optimism, so accept a deeper stack only when its *honest* final score
    (``final_honest``: blend weights fit on k-1 folds) beats the shallower one by more than noise.
    """
    from sklearn.base import clone

    task = _infer_task(y_true) if task == "auto" else task
    m = M.get(metric)
    y_arr, f_arr = np.asarray(y_true), np.asarray(folds)
    layers = layers or [_default_meta_models(task) for _ in range(max(1, n_layers))]
    cur_oof, cur_test = dict(oofs), dict(tests)
    trace = []
    for li, layer in enumerate(layers, start=1):
        nxt_oof, nxt_test = {}, {}
        for name, est in layer:
            res = stack(cur_oof, cur_test, y_true, folds, meta_model=clone(est), task=task, use_ranks=use_ranks)
            key = f"L{li + 1}_{name}"
            nxt_oof[key], nxt_test[key] = res["oof"], res["test"]
        trace.append({k: float(m(y_arr[f_arr >= 0], v[f_arr >= 0])) for k, v in nxt_oof.items()})
        if passthrough:
            nxt_oof.update(oofs)
            nxt_test.update(tests)
        cur_oof, cur_test = nxt_oof, nxt_test
    if final == "hill_climb":
        weights = hill_climb(cur_oof, y_true, metric)["weights"]
        honest = cv_blend_score(cur_oof, y_true, folds, metric)["oof_score"]
    else:
        weights = {k: 1 / len(cur_oof) for k in cur_oof}
        honest = None
    oof = blend(cur_oof, weights)
    test = blend({k: cur_test[k] for k in weights}, weights)
    return {"oof": oof, "test": test, "weights": weights, "layer_scores": trace,
            "score": float(m(y_arr, oof)), "final_honest": honest, "task": task}


def residual_stack(
    oofs: Mapping[str, np.ndarray],
    tests: Mapping[str, np.ndarray],
    y_true,
    folds,
    base: str,
    model=None,
    X_extra: np.ndarray | None = None,
    X_extra_test: np.ndarray | None = None,
    shrink: float = 1.0,
) -> dict:
    """Residual stacking: a stage-2 model learns the errors of the ``base`` model from the other models'
    OOFs (and optional raw features), out-of-fold on the same folds; prediction = base + shrink * residual.

    For regression and binary probabilities (clipped to [0, 1]). It recovers structure the base model
    systematically misses (a slice where another model is right) without re-weighting everything.
    For GBDT bases, an equivalent alternative is boosting from the base margin (``init_score``).
    """
    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingRegressor

    y = np.asarray(y_true, dtype=float)
    folds = np.asarray(folds)
    b_oof, b_test = np.asarray(oofs[base], dtype=float), np.asarray(tests[base], dtype=float)
    if b_oof.ndim != 1:
        raise ValueError("residual stacking supports regression / binary (1-D predictions)")
    names = list(oofs)
    X = np.column_stack([np.asarray(oofs[n], dtype=float) for n in names])
    Xt = np.column_stack([np.asarray(tests[n], dtype=float) for n in names])
    if X_extra is not None:
        X = np.hstack([X, np.asarray(X_extra, dtype=float).reshape(len(X), -1)])
        Xt = np.hstack([Xt, np.asarray(X_extra_test, dtype=float).reshape(len(Xt), -1)])
    model = model or HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                                   l2_regularization=1.0, random_state=0)
    r = y - b_oof
    is_prob = set(np.unique(y)).issubset({0.0, 1.0}) and b_oof.min() >= 0 and b_oof.max() <= 1
    r_oof, r_test = np.zeros_like(y), np.zeros(len(Xt))
    fold_ids = sorted(set(folds[folds >= 0].tolist()))
    for f in fold_ids:
        tr, va = folds != f, folds == f
        mdl = clone(model).fit(X[tr], r[tr])
        r_oof[va] = mdl.predict(X[va])
        r_test += mdl.predict(Xt) / len(fold_ids)
    oof, test = b_oof + shrink * r_oof, b_test + shrink * r_test
    if is_prob:
        oof, test = np.clip(oof, 0, 1), np.clip(test, 0, 1)
    return {"oof": oof, "test": test, "base": base}
