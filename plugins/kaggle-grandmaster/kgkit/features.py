"""Feature engineering primitives that are leak-safe by construction.

The golden rule: any statistic that uses the *target* must be computed out-of-fold
on the training set (with the same folds you validate on) and on the full training set
for test. Statistics that do not touch the target (counts, group means of features) may
use train+test together — that is standard practice and often helps.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def reduce_mem_usage(df: pd.DataFrame, float16: bool = False, verbose: bool = False) -> pd.DataFrame:
    """Downcast numeric columns in place-ish (returns the frame). float16 is lossy — off by default."""
    start = df.memory_usage(deep=True).sum()
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            continue
        if pd.api.types.is_integer_dtype(s):
            df[c] = pd.to_numeric(s, downcast="integer")
        else:
            df[c] = s.astype(np.float16) if float16 else pd.to_numeric(s, downcast="float")
    if verbose:
        end = df.memory_usage(deep=True).sum()
        print(f"memory {start / 1e6:.1f}MB -> {end / 1e6:.1f}MB")
    return df


def _smoothed_mean(stats: pd.DataFrame, prior: float, smoothing: float) -> pd.Series:
    return (stats["sum"] + prior * smoothing) / (stats["count"] + smoothing)


def oof_target_encode(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    cols: str | Sequence[str],
    target: str,
    folds: Sequence[int],
    smoothing: float = 20.0,
    name: str | None = None,
) -> tuple[pd.Series, pd.Series | None]:
    """Smoothed mean target encoding, out-of-fold for train and full-train for test.

    ``cols`` may be several columns, encoded as their combination. Unseen categories
    get the prior. Returns (train_encoded, test_encoded).
    """
    cols = [cols] if isinstance(cols, str) else list(cols)
    name = name or "te_" + "_".join(cols)
    folds = np.asarray(folds)
    key_tr = train[cols].astype(str).agg("|".join, axis=1) if len(cols) > 1 else train[cols[0]]
    y = train[target].astype(float)
    out = pd.Series(np.nan, index=train.index, name=name, dtype=float)
    for f in sorted(set(folds[folds >= 0].tolist())):
        tr, va = folds != f, folds == f
        prior = y[tr].mean()
        stats = y[tr].groupby(key_tr[tr]).agg(["sum", "count"])
        enc = _smoothed_mean(stats, prior, smoothing)
        out[va] = key_tr[va].map(enc).fillna(prior).to_numpy()
    if (folds < 0).any():
        out[folds < 0] = y.mean()
    test_out = None
    if test is not None:
        key_te = test[cols].astype(str).agg("|".join, axis=1) if len(cols) > 1 else test[cols[0]]
        prior = y.mean()
        stats = y.groupby(key_tr).agg(["sum", "count"])
        test_out = key_te.map(_smoothed_mean(stats, prior, smoothing)).fillna(prior).astype(float).rename(name)
    return out, test_out


def count_encode(train: pd.DataFrame, test: pd.DataFrame | None, cols: Iterable[str], use_test: bool = True,
                 normalize: bool = False) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Frequency of each value (computed on train+test when ``use_test``)."""
    tr_out, te_out = pd.DataFrame(index=train.index), (pd.DataFrame(index=test.index) if test is not None else None)
    for c in cols:
        pool = pd.concat([train[c], test[c]]) if (use_test and test is not None) else train[c]
        vc = pool.value_counts(normalize=normalize, dropna=False)
        tr_out[f"cnt_{c}"] = train[c].map(vc).astype(float)
        if test is not None:
            te_out[f"cnt_{c}"] = test[c].map(vc).fillna(0).astype(float)
    return tr_out, te_out


def group_aggregates(df: pd.DataFrame, by: str | Sequence[str], cols: Iterable[str],
                     aggs: Sequence[str] = ("mean", "std", "min", "max"), add_diffs: bool = True) -> pd.DataFrame:
    """Group statistics of feature columns broadcast back to rows, plus value-minus-mean
    and value/mean deviations (often the most useful part). Concatenate train+test first
    if the groups span both."""
    by = [by] if isinstance(by, str) else list(by)
    tag = "_".join(by)
    out = pd.DataFrame(index=df.index)
    g = df.groupby(by)
    for c in cols:
        for a in aggs:
            out[f"{c}_{a}_by_{tag}"] = g[c].transform(a)
        if add_diffs and "mean" in aggs:
            m = out[f"{c}_mean_by_{tag}"]
            out[f"{c}_diff_mean_by_{tag}"] = df[c] - m
            out[f"{c}_ratio_mean_by_{tag}"] = df[c] / m.replace(0, np.nan)
    out[f"size_by_{tag}"] = g[by[0]].transform("size")
    return out


def lag_features(df: pd.DataFrame, group: str | Sequence[str] | None, time: str, cols: Iterable[str],
                 lags: Sequence[int] = (1, 7), windows: Sequence[int] = (7, 28), shift: int = 1) -> pd.DataFrame:
    """Lags and shifted rolling means/stds per group, ordered by ``time``.

    ``shift`` is applied before rolling so a row never sees its own target; set it to
    the forecast horizon when predicting h steps ahead (recursive features otherwise
    leak the future relative to the LB period).
    """
    order = df.sort_values(time).index
    d = df.loc[order]
    g = d.groupby(group) if group is not None else d
    out = pd.DataFrame(index=order)
    for c in cols:
        s = g[c] if group is not None else d[c]
        for lag in lags:
            out[f"{c}_lag{lag}"] = s.shift(lag)
        base = s.shift(shift)
        for w in windows:
            if group is not None:
                gb = base.groupby([d[k] for k in ([group] if isinstance(group, str) else group)])
                out[f"{c}_rmean{w}"] = gb.transform(lambda x: x.rolling(w, min_periods=1).mean())
                out[f"{c}_rstd{w}"] = gb.transform(lambda x: x.rolling(w, min_periods=2).std())
            else:
                out[f"{c}_rmean{w}"] = base.rolling(w, min_periods=1).mean()
                out[f"{c}_rstd{w}"] = base.rolling(w, min_periods=2).std()
    return out.loc[df.index]


def date_features(s: pd.Series, prefix: str | None = None, cyclical: bool = True) -> pd.DataFrame:
    s = pd.to_datetime(s)
    p = prefix or s.name or "dt"
    out = pd.DataFrame({
        f"{p}_year": s.dt.year,
        f"{p}_month": s.dt.month,
        f"{p}_day": s.dt.day,
        f"{p}_dow": s.dt.dayofweek,
        f"{p}_doy": s.dt.dayofyear,
        f"{p}_hour": s.dt.hour,
        f"{p}_is_weekend": (s.dt.dayofweek >= 5).astype(int),
        f"{p}_is_month_end": s.dt.is_month_end.astype(int),
        f"{p}_elapsed_days": (s - s.min()).dt.total_seconds() / 86400,
    }, index=s.index)
    if cyclical:
        for col, period in [("month", 12), ("dow", 7), ("hour", 24)]:
            v = out[f"{p}_{col}"]
            out[f"{p}_{col}_sin"] = np.sin(2 * np.pi * v / period)
            out[f"{p}_{col}_cos"] = np.cos(2 * np.pi * v / period)
    return out


def cross_categoricals(df: pd.DataFrame, pairs: Iterable[tuple[str, str]]) -> pd.DataFrame:
    """Concatenate pairs of categorical columns into interaction categories."""
    return pd.DataFrame({f"{a}__{b}": df[a].astype(str) + "_" + df[b].astype(str) for a, b in pairs}, index=df.index)
