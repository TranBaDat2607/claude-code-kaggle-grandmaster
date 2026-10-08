"""Fast, opinionated EDA that surfaces what matters for *winning*, not for pretty plots:
leaks, ID-like columns, train/test drift, unseen categories, target shape and anything
that will make a naive CV lie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _univariate_strength(x: pd.Series, y: pd.Series) -> float:
    """max(AUC, 1-AUC) for binary targets, |spearman| for numeric targets (1.0 = perfect)."""
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    if len(x) < 10 or x.nunique() < 2:
        return np.nan
    if y.nunique() == 2:
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(pd.factorize(y, sort=True)[0], x)
        return round(float(max(auc, 1 - auc)), 3)
    if _is_num(y):
        with np.errstate(all="ignore"):
            c = stats.spearmanr(x, y)[0]
        return round(float(abs(c)), 3) if pd.notna(c) else np.nan
    return np.nan


def column_profile(train: pd.DataFrame, test: pd.DataFrame | None = None, target: str | None = None,
                   id_col: str | None = None) -> pd.DataFrame:
    rows = []
    for c in train.columns:
        if c in (target, id_col):
            continue
        s = train[c]
        r = {
            "column": c,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "missing%": round(100 * s.isna().mean(), 2),
            "top_share": round(float(s.value_counts(normalize=True, dropna=False).iloc[0]), 3) if len(s) else np.nan,
        }
        if test is not None and c in test.columns:
            t = test[c]
            r["test_missing%"] = round(100 * t.isna().mean(), 2)
            if _is_num(s) and _is_num(t):
                a, b = s.dropna(), t.dropna()
                r["drift_ks"] = round(float(stats.ks_2samp(a, b).statistic), 3) if len(a) and len(b) else np.nan
                r["unseen_in_test%"] = np.nan
            else:
                seen = set(s.dropna().unique())
                r["drift_ks"] = np.nan
                r["unseen_in_test%"] = round(100 * float((~t.dropna().isin(seen)).mean()), 2) if t.notna().any() else 0.0
        if target is not None and _is_num(s):
            r["univariate"] = _univariate_strength(s, train[target])
        rows.append(r)
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    df = df.head(max_rows)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def quick_eda(train: pd.DataFrame, test: pd.DataFrame | None = None, target: str | None = None,
              id_col: str | None = None) -> str:
    """Return a markdown EDA report with an explicit list of red flags."""
    out = ["# Quick EDA", ""]
    out.append(f"- train: {train.shape[0]:,} rows x {train.shape[1]} cols "
               f"({train.memory_usage(deep=True).sum() / 1e6:.1f} MB)")
    if test is not None:
        out.append(f"- test: {test.shape[0]:,} rows x {test.shape[1]} cols (test/train ratio {len(test) / max(1, len(train)):.2f})")
        only_train = [c for c in train.columns if c not in test.columns and c != target]
        if only_train:
            out.append(f"- columns only in train (besides target): {only_train} — not usable as features unless reconstructable")
    flags: list[str] = []

    if target is not None and target in train.columns:
        y = train[target]
        out += ["", "## Target", ""]
        if _is_num(y) and y.nunique() > 20:
            out.append(f"- continuous: mean {y.mean():.4g}, std {y.std():.4g}, min {y.min():.4g}, "
                       f"median {y.median():.4g}, max {y.max():.4g}, skew {y.skew():.2f}")
            if y.min() >= 0 and y.skew() > 2:
                flags.append("target is heavily right-skewed and non-negative — try log1p transform / RMSLE-style training")
        else:
            vc = y.value_counts(normalize=True)
            out.append("- classes: " + ", ".join(f"{k}: {v:.3f}" for k, v in vc.head(20).items()))
            if vc.iloc[-1] < 0.05:
                flags.append(f"class imbalance: rarest class share {vc.iloc[-1]:.4f} — stratify folds, check metric sensitivity, consider thresholds")
        if y.isna().any():
            flags.append(f"target has {int(y.isna().sum())} missing values")

    prof = column_profile(train, test, target, id_col)
    out += ["", "## Columns", "", _md_table(prof)]

    n = len(train)
    for _, r in prof.iterrows():
        c = r["column"]
        if r["n_unique"] <= 1:
            flags.append(f"`{c}` is constant — drop")
        if r["n_unique"] == n and c != id_col:
            flags.append(f"`{c}` is unique per row (ID-like) — drop as a feature unless it encodes order/time (check vs target)")
        if r["missing%"] > 50:
            flags.append(f"`{c}` is {r['missing%']}% missing — missingness itself may be predictive")
        if "test_missing%" in r and pd.notna(r.get("test_missing%")) and abs(r["test_missing%"] - r["missing%"]) > 10:
            flags.append(f"`{c}` missing rate differs train {r['missing%']}% vs test {r['test_missing%']}%")
        if pd.notna(r.get("drift_ks", np.nan)) and r["drift_ks"] > 0.1:
            flags.append(f"`{c}` drifts between train and test (KS {r['drift_ks']}) — run adversarial validation")
        if pd.notna(r.get("unseen_in_test%", np.nan)) and r["unseen_in_test%"] > 5:
            flags.append(f"`{c}` has {r['unseen_in_test%']}% test values unseen in train — target encodings will fall back to prior")
        if pd.notna(r.get("univariate", np.nan)) and r["univariate"] > 0.97:
            flags.append(f"`{c}` alone predicts the target almost perfectly (univariate {r['univariate']}) — possible LEAK, investigate")

    feats = [c for c in train.columns if c not in (target, id_col)]
    dup = int(train.duplicated(subset=feats).sum()) if feats else 0
    if dup:
        msg = f"{dup} duplicated feature rows in train"
        if target is not None:
            conflicting = train[train.duplicated(subset=feats, keep=False)].groupby(feats, dropna=False)[target].nunique()
            msg += f" ({int((conflicting > 1).sum())} groups with conflicting targets — label noise ceiling)"
        flags.append(msg + " — duplicates across folds inflate CV; consider grouping them")
    if test is not None and feats:
        common = [c for c in feats if c in test.columns]
        if common:
            merged = test[common].merge(train[common].drop_duplicates(), on=common, how="inner")
            if len(merged):
                flags.append(f"{len(merged)} test rows exactly match a train row on all features — possible leak to exploit (lookup)")

    if target is not None and target in train.columns and _is_num(train[target]):
        with np.errstate(all="ignore"):
            rc = stats.spearmanr(np.arange(n), train[target].to_numpy(), nan_policy="omit")[0]
        if pd.notna(rc) and abs(rc) > 0.1:
            flags.append(f"row order correlates with target (spearman {rc:.2f}) — data is sorted/time-ordered; random KFold may leak")

    out += ["", "## Red flags", ""]
    out += [f"- {f}" for f in flags] if flags else ["- none detected (still run adversarial validation)"]
    return "\n".join(out)
