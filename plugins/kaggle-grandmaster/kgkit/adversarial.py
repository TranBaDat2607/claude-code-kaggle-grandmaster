"""Adversarial validation: can a model tell train rows from test rows?

AUC ~0.5  -> train and test look alike; a random KFold CV is likely trustworthy.
AUC >0.7  -> real distribution shift. Find the drifting features (top importances),
             then: drop / transform them, build a validation fold out of the most
             test-like training rows, or weight training rows by p(test|x)/p(train|x).
AUC ~1.0  -> usually an ID / timestamp / row-order column; drop it and re-run.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold


def _prepare(train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str]) -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.concat([train[features], test[features]], axis=0, ignore_index=True)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):
            X[c] = pd.factorize(X[c])[0]
    y = np.r_[np.zeros(len(train)), np.ones(len(test))]
    return X, y


def adversarial_validation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str] | None = None,
    n_splits: int = 5,
    seed: int = 42,
    max_rows: int = 200_000,
    importance_rows: int = 20_000,
) -> dict:
    """Train a train-vs-test classifier with CV.

    Returns {"auc", "fold_aucs", "importance" (DataFrame, drop-in permutation AUC loss),
    "p_test" (OOF p(test) for every *training* row, aligned with ``train``)}.
    """
    from sklearn.metrics import roc_auc_score

    if features is None:
        features = [c for c in train.columns if c in test.columns]
    features = list(features)
    X, y = _prepare(train, test, features)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    if len(X) > max_rows:
        # keep every training row's prediction available: subsample only for fitting
        fit_idx = np.sort(rng.choice(idx, max_rows, replace=False))
    else:
        fit_idx = idx

    oof = np.full(len(X), np.nan)
    aucs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    last_model, last_va = None, None
    for tr, va in skf.split(X.iloc[fit_idx], y[fit_idx]):
        tr, va = fit_idx[tr], fit_idx[va]
        mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=seed)
        mdl.fit(X.iloc[tr], y[tr])
        oof[va] = mdl.predict_proba(X.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y[va], oof[va]))
        last_model, last_va = mdl, va
    if np.isnan(oof[: len(train)]).any():
        # rows left out of the fitting subsample: score them with the last model
        miss = np.flatnonzero(np.isnan(oof))
        oof[miss] = last_model.predict_proba(X.iloc[miss])[:, 1]

    sub = last_va if len(last_va) <= importance_rows else rng.choice(last_va, importance_rows, replace=False)
    pi = permutation_importance(last_model, X.iloc[sub], y[sub], scoring="roc_auc", n_repeats=3, random_state=seed)
    imp = (
        pd.DataFrame({"feature": features, "auc_drop": pi.importances_mean, "std": pi.importances_std})
        .sort_values("auc_drop", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "auc": float(np.mean(aucs)),
        "fold_aucs": [float(a) for a in aucs],
        "importance": imp,
        "p_test": oof[: len(train)],
    }


def iterative_drift_removal(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    target_auc: float = 0.6,
    max_drop: int = 10,
    **kw,
) -> dict:
    """Repeatedly drop the most drifting feature until AUC <= target_auc.

    Dropping is a *diagnostic*, not an automatic decision: a drifting feature can still
    be the strongest predictor. Check CV impact of each removal before committing.
    """
    feats = list(features)
    dropped, trace = [], []
    for _ in range(max_drop + 1):
        res = adversarial_validation(train, test, feats, **kw)
        trace.append((list(dropped), res["auc"]))
        if res["auc"] <= target_auc or len(feats) <= 1 or len(dropped) >= max_drop:
            break
        worst = res["importance"].iloc[0]["feature"]
        feats.remove(worst)
        dropped.append(worst)
    return {"kept": feats, "dropped": dropped, "trace": trace}


def testlike_holdout_mask(p_test: np.ndarray, frac: float = 0.2) -> np.ndarray:
    """Boolean mask selecting the ``frac`` most test-like training rows — use as a
    holdout that mimics the leaderboard distribution."""
    cut = np.quantile(p_test, 1 - frac)
    return np.asarray(p_test) >= cut


def importance_weights(p_test: np.ndarray, clip: tuple[float, float] = (0.01, 0.99), n_train: int | None = None,
                       n_test: int | None = None) -> np.ndarray:
    """Density-ratio sample weights w = p(test|x)/p(train|x), corrected for class sizes
    and normalised to mean 1."""
    p = np.clip(np.asarray(p_test, dtype=float), *clip)
    w = p / (1 - p)
    if n_train and n_test:
        w *= n_train / n_test
    return w / w.mean()
