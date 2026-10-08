"""Validate a submission file against ``sample_submission`` before spending a daily slot.

A wasted submission (wrong id order, missing rows, NaN, labels instead of probabilities,
index column written by accident) costs a day of feedback near the deadline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def validate_submission(sub: str | Path | pd.DataFrame, sample: str | Path | pd.DataFrame,
                        id_col: str | None = None, prob_cols: list[str] | None = None) -> dict:
    """Return {"ok": bool, "errors": [...], "warnings": [...], "summary": str}."""
    s = pd.read_csv(sub) if not isinstance(sub, pd.DataFrame) else sub
    ref = pd.read_csv(sample) if not isinstance(sample, pd.DataFrame) else sample
    errors, warnings = [], []
    id_col = id_col or ref.columns[0]

    if list(s.columns) != list(ref.columns):
        missing = [c for c in ref.columns if c not in s.columns]
        extra = [c for c in s.columns if c not in ref.columns]
        if missing or extra:
            errors.append(f"column mismatch: missing {missing}, extra {extra}")
        else:
            warnings.append("columns are in a different order than sample_submission (usually fine, but match it)")
        if any(str(c).startswith("Unnamed") for c in s.columns):
            errors.append("an 'Unnamed' column is present — you wrote the DataFrame index (use index=False)")

    if len(s) != len(ref):
        errors.append(f"row count {len(s)} != sample {len(ref)}")

    if id_col in s.columns and id_col in ref.columns:
        if s[id_col].duplicated().any():
            errors.append(f"{int(s[id_col].duplicated().sum())} duplicated ids")
        sid, rid = set(s[id_col].astype(str)), set(ref[id_col].astype(str))
        if sid != rid:
            errors.append(f"id set differs from sample: {len(rid - sid)} missing, {len(sid - rid)} unexpected")
        elif not (s[id_col].astype(str).to_numpy() == ref[id_col].astype(str).to_numpy()).all():
            warnings.append("ids are in a different order than sample_submission (Kaggle usually joins by id; sort to be safe)")

    value_cols = [c for c in ref.columns if c != id_col and c in s.columns]
    for c in value_cols:
        col = s[c]
        if col.isna().any():
            errors.append(f"`{c}` has {int(col.isna().sum())} NaN values")
        ref_num = pd.api.types.is_numeric_dtype(ref[c])
        if ref_num and not pd.api.types.is_numeric_dtype(col):
            errors.append(f"`{c}` should be numeric like the sample but has dtype {col.dtype}")
            continue
        if pd.api.types.is_numeric_dtype(col):
            if np.isinf(col.to_numpy(dtype=float, na_value=np.nan)).any():
                errors.append(f"`{c}` has infinite values")
            if col.nunique() <= 1 and len(col) > 1:
                warnings.append(f"`{c}` is constant ({col.iloc[0]}) — is the model actually predicting?")
            is_prob = (prob_cols is not None and c in prob_cols)
            if is_prob and ((col < 0) | (col > 1)).any():
                errors.append(f"`{c}` has values outside [0, 1] but is declared a probability column")
            ref_int = pd.api.types.is_integer_dtype(ref[c])
            if ref_int and not np.allclose(col.dropna() % 1, 0):
                warnings.append(f"`{c}` is integer in the sample but your values are fractional — labels expected?")

    summary = f"{len(s)} rows, columns {list(s.columns)}"
    return {"ok": not errors, "errors": errors, "warnings": warnings, "summary": summary}


def format_report(res: dict) -> str:
    lines = [("OK — submission is valid" if res["ok"] else "INVALID submission"), res["summary"]]
    lines += [f"ERROR: {e}" for e in res["errors"]]
    lines += [f"warning: {w}" for w in res["warnings"]]
    return "\n".join(lines)
