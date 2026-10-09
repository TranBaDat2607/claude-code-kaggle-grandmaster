"""Is experiment A really better than B? Paired comparisons on the same folds.

The tempting rule "accept if the gain beats one fold std" is the wrong yardstick: the fold std
is dominated by how *hard* each fold is, and that difficulty is shared by both models, so it
cancels in a paired comparison. The right noise scale is the spread of the per-fold
*differences* (and, with OOF predictions, a paired bootstrap over rows). Using the fold std
throws away real gains, which in the late game are exactly the gains you are stacking.

    from kgkit.compare import compare_folds, paired_bootstrap, verdict
    f = compare_folds(new["fold_scores"], base["fold_scores"], greater_is_better=True)
    b = paired_bootstrap(y, oof_new, oof_base, "auc", groups=patient_ids)
    print(verdict(f, b))

CLI: ``python -m kgkit ledger compare <new> <baseline> [--truth data/train.csv:target]``.

What each test does *not* see: the fold-paired test sees training noise only through K
numbers; the bootstrap sees row-sampling noise for *fixed* trained models, not seed noise. A
small gain that passes only the bootstrap deserves a second seed before it becomes the baseline.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from . import metrics as M


def compare_folds(new_scores: Sequence[float], base_scores: Sequence[float], greater_is_better: bool = True) -> dict:
    """Paired per-fold comparison. ``gain`` is signed so that positive always means *new is better*."""
    a, b = np.asarray(new_scores, dtype=float), np.asarray(base_scores, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError(f"need per-fold scores of equal length, got {a.shape} vs {b.shape}")
    d = (a - b) if greater_is_better else (b - a)
    k = len(d)
    sd = float(np.std(d, ddof=1)) if k > 1 else float("nan")
    se = sd / math.sqrt(k) if k > 1 else float("nan")
    gain = float(d.mean())
    t = gain / se if k > 1 and se > 0 else (math.copysign(math.inf, gain) if gain else 0.0)
    return {
        "k": k,
        "gain": gain,
        "per_fold": d.tolist(),
        "folds_improved": int((d > 0).sum()),
        "diff_sd": sd,
        "se": se,
        "t": float(t),
        "base_fold_std": float(np.std(b)),
    }


def paired_bootstrap(
    y_true,
    pred_new,
    pred_base,
    metric: str | Callable,
    greater_is_better: bool | None = None,
    n_boot: int = 300,
    seed: int = 0,
    groups=None,
    mask=None,
) -> dict:
    """Paired bootstrap of metric(new) - metric(base) over rows (or over ``groups`` when rows are
    clustered, e.g. several images per patient — resample whole groups, like the CV does).

    ``mask`` restricts the comparison to some rows, e.g. one fold for a 1-fold screen.
    Both predictions are scored on the *same* resample each time, which is what makes it paired.
    """
    if isinstance(metric, str):
        m = M.get(metric)
        fn, gib = m.fn, m.greater_is_better
    else:
        if greater_is_better is None:
            raise ValueError("pass greater_is_better with a callable metric")
        fn, gib = metric, greater_is_better
    y, pa, pb = np.asarray(y_true), np.asarray(pred_new, dtype=float), np.asarray(pred_base, dtype=float)
    rows = np.arange(len(y)) if mask is None else np.flatnonzero(np.asarray(mask))
    sign = 1.0 if gib else -1.0
    full = sign * (fn(y[rows], pa[rows]) - fn(y[rows], pb[rows]))
    rng = np.random.default_rng(seed)
    if groups is not None:
        g = np.asarray(groups)[rows]
        uniq, inv = np.unique(g, return_inverse=True)
        members = [rows[inv == i] for i in range(len(uniq))]
    gains = []
    for _ in range(n_boot):
        if groups is None:
            idx = rows[rng.integers(0, len(rows), len(rows))]
        else:
            pick = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[i] for i in pick])
        try:
            gains.append(sign * (fn(y[idx], pa[idx]) - fn(y[idx], pb[idx])))
        except ValueError:  # e.g. a resample with one class for AUC
            continue
    gains = np.asarray(gains, dtype=float)
    se = float(gains.std(ddof=1)) if len(gains) > 1 else float("nan")
    return {
        "gain": float(full),
        "se": se,
        "z": float(full / se) if se and se > 0 else float("nan"),
        "p_better": float((gains > 0).mean()) if len(gains) else float("nan"),
        "ci95": [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))] if len(gains) else None,
        "n_boot": int(len(gains)),
        "n_rows": int(len(rows)),
    }


def verdict(folds: dict | None, boot: dict | None = None, z: float = 2.0) -> dict:
    """KEEP / DISCARD / INCONCLUSIVE from the paired fold test and/or the paired bootstrap.

    KEEP needs a positive gain that is >= ``z`` standard errors from zero in one test, with the
    other test (if available) not pointing the other way and most folds improving.
    A gain far above the baseline's fold std is flagged: audit it for leakage before accepting.
    """
    if folds is None and boot is None:
        raise ValueError("need fold scores or OOF predictions")
    gain = folds["gain"] if folds is not None else boot["gain"]
    reasons, flags = [], []
    fold_strong = folds is not None and folds["k"] > 1 and folds["t"] >= z
    majority = folds is None or folds["folds_improved"] * 2 > folds["k"]
    boot_strong = boot is not None and boot["z"] == boot["z"] and boot["z"] >= z
    boot_against = boot is not None and boot["gain"] < 0
    if folds is not None:
        reasons.append(f"folds: gain {folds['gain']:+.5f}, improved {folds['folds_improved']}/{folds['k']}, "
                       f"paired SE {folds['se']:.5f} (t={folds['t']:.2f}); baseline fold std "
                       f"{folds['base_fold_std']:.5f}")
        if folds["base_fold_std"] > 0 and gain > 3 * folds["base_fold_std"]:
            flags.append("gain > 3x the baseline fold std: audit for leakage (validation-auditor) before accepting")
    if boot is not None:
        reasons.append(f"bootstrap: gain {boot['gain']:+.5f} +/- {boot['se']:.5f} (z={boot['z']:.2f}, "
                       f"P(better)={boot['p_better']:.2f}, {boot['n_rows']} rows)")
    if gain <= 0:
        decision = "DISCARD"
    elif (fold_strong or boot_strong) and majority and not boot_against:
        decision = "KEEP"
    else:
        decision = "INCONCLUSIVE"
        reasons.append("positive but within noise: rerun with a second seed (or more folds) before "
                       "accepting, or keep only if it is free (no extra complexity / runtime)")
    return {"decision": decision, "gain": gain, "reasons": reasons, "flags": flags}


def format_comparison(new_id: str, base_id: str, res: dict) -> str:
    lines = [f"{new_id} vs {base_id}: {res['decision']} (gain {res['gain']:+.5f}, positive = better)"]
    lines += [f"  {r}" for r in res["reasons"]]
    lines += [f"  WARNING: {f}" for f in res["flags"]]
    return "\n".join(lines)
