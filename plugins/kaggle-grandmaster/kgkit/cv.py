"""Leak-free cross-validation fold assignment.

The single most important decision in a competition is the validation scheme: every
later decision (features, models, blend weights, final submission choice) is only as
good as the CV it is measured with. These helpers make the common schemes one call
and make the dangerous mistakes (group leakage, time leakage, unstable stratification)
loud.

Typical use::

    from kgkit.cv import assign_folds, fold_report
    train["fold"] = assign_folds(train, n_splits=5, target="label", group="patient_id")
    print(fold_report(train, "fold", target="label", group="patient_id"))
    train[["id", "fold"]].to_csv("folds.csv", index=False)   # freeze it, reuse forever
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

STRATEGIES = ("auto", "kfold", "stratified", "group", "stratified_group", "multilabel")


def _is_classification_target(y: pd.Series, max_classes: int = 50) -> bool:
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return True
    n_unique = y.nunique(dropna=True)
    if n_unique <= max_classes and np.allclose(y.dropna() % 1, 0):
        return True
    return False


def regression_bins(y: Sequence[float], n_bins: int | None = None) -> np.ndarray:
    """Quantile-bin a continuous target so it can be stratified on.

    Uses Sturges' rule (capped at 20) when ``n_bins`` is not given.
    """
    y = pd.Series(np.asarray(y, dtype=float))
    if n_bins is None:
        n_bins = int(min(20, max(2, np.floor(1 + np.log2(len(y))))))
    return pd.qcut(y.rank(method="first"), q=n_bins, labels=False, duplicates="drop").to_numpy()


def iterative_stratification(Y: np.ndarray, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Multilabel stratification (Sechidis et al., 2011).

    ``Y`` is an (n_samples, n_labels) 0/1 matrix. Returns a fold id per row such that
    every label's positives are spread as evenly as possible across folds, handling the
    rarest labels first.
    """
    Y = np.asarray(Y).astype(bool)
    n, _ = Y.shape
    rng = np.random.default_rng(seed)
    ratios = np.full(n_splits, 1.0 / n_splits)
    desired = n * ratios
    desired_label = np.outer(ratios, Y.sum(axis=0)).astype(float)
    folds = np.full(n, -1, dtype=int)
    remaining = np.ones(n, dtype=bool)

    while remaining.any():
        label_counts = Y[remaining].sum(axis=0)
        if label_counts.sum() == 0:
            idx = np.flatnonzero(remaining)
            rng.shuffle(idx)
            for i in idx:
                best = np.flatnonzero(desired == desired.max())
                f = rng.choice(best)
                folds[i] = f
                desired[f] -= 1
            break
        lc = np.where(label_counts > 0, label_counts, np.inf)
        label = int(np.argmin(lc))
        idx = np.flatnonzero(remaining & Y[:, label])
        rng.shuffle(idx)
        for i in idx:
            cand = desired_label[:, label]
            best = np.flatnonzero(cand == cand.max())
            if len(best) > 1:
                best = best[desired[best] == desired[best].max()]
            f = rng.choice(best)
            folds[i] = f
            remaining[i] = False
            desired_label[f] -= Y[i]
            desired[f] -= 1
    return folds


def assign_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    strategy: str = "auto",
    target: str | Sequence[str] | None = None,
    group: str | None = None,
    seed: int = 42,
    n_bins: int | None = None,
) -> np.ndarray:
    """Return an integer fold id (0..n_splits-1) for every row of ``df``.

    strategy:
        auto             group given -> stratified_group (if a target is given) else group;
                         list of targets -> multilabel; target given -> stratified
                         (continuous targets are quantile-binned); else kfold.
        kfold            shuffled KFold.
        stratified       StratifiedKFold on target (binned if continuous).
        group            GroupKFold — no group ever appears in two folds.
        stratified_group StratifiedGroupKFold — group-safe *and* balanced.
        multilabel       iterative stratification over several 0/1 target columns.

    For time-ordered data do NOT use this; use :func:`time_series_splits`.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    n = len(df)
    multi = isinstance(target, (list, tuple))

    if strategy == "auto":
        if multi:
            strategy = "multilabel"
        elif group is not None:
            strategy = "stratified_group" if target is not None else "group"
        elif target is not None:
            strategy = "stratified"
        else:
            strategy = "kfold"

    def strat_labels() -> np.ndarray:
        y = df[target]
        if _is_classification_target(y):
            return pd.factorize(y)[0]
        return regression_bins(y, n_bins)

    folds = np.full(n, -1, dtype=int)
    X_dummy = np.zeros((n, 1))

    if strategy == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(X_dummy)
    elif strategy == "stratified":
        if target is None:
            raise ValueError("stratified requires target")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(X_dummy, strat_labels())
    elif strategy == "group":
        if group is None:
            raise ValueError("group requires group")
        groups = df[group].to_numpy()
        # GroupKFold is deterministic; shuffle group identities for seed control.
        uniq = pd.unique(groups)
        perm = np.random.default_rng(seed).permutation(len(uniq))
        remap = dict(zip(uniq, perm))
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X_dummy, groups=np.array([remap[g] for g in groups]))
    elif strategy == "stratified_group":
        if group is None or target is None:
            raise ValueError("stratified_group requires target and group")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(X_dummy, strat_labels(), groups=df[group].to_numpy())
    else:  # multilabel
        if not multi:
            raise ValueError("multilabel requires a list of target columns")
        return iterative_stratification(df[list(target)].to_numpy(), n_splits, seed)

    for fold, (_, va) in enumerate(splits):
        folds[va] = fold
    return folds


def time_series_splits(
    time_values: Sequence,
    n_splits: int = 5,
    gap: int = 0,
    test_size: int | None = None,
    max_train_size: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits over *unique* time values (rows sharing a timestamp
    never straddle train/valid).

    ``gap`` / ``test_size`` / ``max_train_size`` are counted in unique time steps.
    ``gap`` should be at least the forecast horizon so lag features cannot see the
    validation period. Returns a list of (train_idx, valid_idx) arrays, oldest first;
    the last split is the one most like the leaderboard.
    """
    t = pd.Series(np.asarray(time_values))
    uniq = np.sort(t.unique())
    n_t = len(uniq)
    if test_size is None:
        test_size = n_t // (n_splits + 1)
    if test_size < 1 or n_t - gap - n_splits * test_size < 1:
        raise ValueError("not enough unique time steps for this n_splits/gap/test_size")
    codes = np.searchsorted(uniq, t.to_numpy())
    out = []
    for k in range(n_splits):
        va_start = n_t - (n_splits - k) * test_size
        va_end = va_start + test_size
        tr_end = va_start - gap
        tr_start = 0 if max_train_size is None else max(0, tr_end - max_train_size)
        tr = np.flatnonzero((codes >= tr_start) & (codes < tr_end))
        va = np.flatnonzero((codes >= va_start) & (codes < va_end))
        out.append((tr, va))
    return out


def purged_kfold_splits(
    time_values: Sequence,
    n_splits: int = 5,
    embargo: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged (blocked) K-fold for time-indexed data where using the future to train
    is acceptable but leakage across the boundary is not (e.g. financial targets
    computed over windows). Validation blocks are contiguous in time and ``embargo``
    unique time steps on *both* sides of each block are removed from training.
    """
    t = pd.Series(np.asarray(time_values))
    uniq = np.sort(t.unique())
    codes = np.searchsorted(uniq, t.to_numpy())
    blocks = np.array_split(np.arange(len(uniq)), n_splits)
    out = []
    for b in blocks:
        lo, hi = b[0], b[-1]
        va = np.flatnonzero((codes >= lo) & (codes <= hi))
        tr = np.flatnonzero((codes < lo - embargo) | (codes > hi + embargo))
        out.append((tr, va))
    return out


def split_indices(folds: Sequence[int]) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (fold, train_idx, valid_idx) from a fold-id column; rows with fold < 0
    (e.g. held out / excluded) are never used for validation but are still trained on."""
    folds = np.asarray(folds)
    for f in sorted(set(folds[folds >= 0].tolist())):
        yield f, np.flatnonzero(folds != f), np.flatnonzero(folds == f)


def fold_report(
    df: pd.DataFrame,
    fold_col: str = "fold",
    target: str | None = None,
    group: str | None = None,
) -> str:
    """Markdown sanity report: fold sizes, target balance per fold, group leakage."""
    lines = ["| fold | rows | share |" + (" target mean | target std |" if target else "")]
    lines.append("|---|---|---|" + ("---|---|" if target else ""))
    n = len(df)
    problems = []
    for f, part in df.groupby(fold_col):
        row = f"| {f} | {len(part)} | {len(part) / n:.3f} |"
        if target:
            y = part[target]
            if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
                y = pd.Series(pd.factorize(df[target])[0], index=df.index).loc[part.index]
            row += f" {y.mean():.4f} | {y.std():.4f} |"
        lines.append(row)
    sizes = df[fold_col].value_counts()
    if sizes.max() > 1.5 * sizes.min():
        problems.append(f"fold sizes unbalanced (min {sizes.min()}, max {sizes.max()})")
    if (df[fold_col] < 0).any():
        problems.append(f"{int((df[fold_col] < 0).sum())} rows have no fold (fold < 0)")
    if group:
        per_group = df.groupby(group)[fold_col].nunique()
        leaked = int((per_group > 1).sum())
        if leaked:
            problems.append(f"GROUP LEAKAGE: {leaked} groups of '{group}' appear in more than one fold")
        else:
            lines.append(f"\nGroup check: every '{group}' value is confined to a single fold. OK")
    if problems:
        lines.append("\n**Problems:**")
        lines += [f"- {p}" for p in problems]
    else:
        lines.append("\nNo fold problems detected.")
    return "\n".join(lines)
