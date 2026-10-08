import numpy as np
import pandas as pd

from kgkit import cv, eda, features as F


def test_oof_target_encoding_is_out_of_fold(binary_df):
    df = binary_df.copy()
    df["uid"] = np.arange(len(df))  # a unique-per-row key: in-fold TE would perfectly leak the target
    folds = cv.assign_folds(df, 5, "stratified", target="target")
    tr_enc, te_enc = F.oof_target_encode(df, df.head(10), "uid", "target", folds, smoothing=1)
    # OOF: every uid unseen in its training folds -> prior, no leakage
    assert np.corrcoef(tr_enc, df["target"])[0, 1] < 0.1
    assert te_enc is not None and len(te_enc) == 10


def test_target_encoding_signal_and_combo(binary_df):
    folds = cv.assign_folds(binary_df, 5, "stratified", target="target")
    enc, _ = F.oof_target_encode(binary_df, None, "cat", "target", folds)
    assert enc.groupby(binary_df["cat"]).mean().idxmax() == "a"
    enc2, _ = F.oof_target_encode(binary_df, None, ["cat", "grp"], "target", folds)
    assert enc2.notna().all()


def test_count_and_group_features(binary_df):
    tr, te = binary_df.iloc[:1500], binary_df.iloc[1500:]
    a, b = F.count_encode(tr, te, ["cat"])
    assert a["cnt_cat"].sum() > 0 and b.shape[0] == 500
    g = F.group_aggregates(binary_df, "cat", ["x1"])
    assert {"x1_mean_by_cat", "x1_diff_mean_by_cat", "size_by_cat"} <= set(g.columns)


def test_lag_features_never_see_current_row():
    df = pd.DataFrame({"g": [0] * 5 + [1] * 5, "t": list(range(5)) * 2, "y": np.arange(10.0)})
    out = F.lag_features(df, "g", "t", ["y"], lags=(1,), windows=(2,))
    assert np.isnan(out.loc[0, "y_lag1"]) and out.loc[1, "y_lag1"] == 0
    assert out.loc[2, "y_rmean2"] == 0.5  # mean of rows 0,1 — excludes row 2
    assert np.isnan(out.loc[5, "y_lag1"])  # group boundary respected


def test_date_and_mem():
    s = pd.Series(pd.date_range("2024-01-01", periods=50, freq="D"), name="d")
    out = F.date_features(s)
    assert "d_dow_sin" in out and out["d_is_weekend"].sum() > 0
    df = pd.DataFrame({"a": np.arange(100, dtype=np.int64), "b": np.random.rand(100)})
    F.reduce_mem_usage(df)
    assert df["a"].dtype == np.int8


def test_quick_eda_flags(binary_df):
    df = binary_df.copy()
    df["leak"] = df["target"] + np.random.default_rng(0).normal(0, 1e-3, len(df))
    df["const"] = 1
    test = df.drop(columns="target").copy()
    test["x2"] = test["x2"] + 2.0
    rep = eda.quick_eda(df, test, target="target", id_col="id")
    assert "Red flags" in rep
    assert "`const` is constant" in rep
    assert "`leak`" in rep and "LEAK" in rep
    assert "`x2` drifts" in rep


def test_quick_eda_ignores_declared_id(binary_df):
    test = binary_df.drop(columns="target").copy()
    test["id"] = test["id"] + 100_000  # ids never overlap — must not be reported as drift
    rep = eda.quick_eda(binary_df, test, target="target", id_col="id")
    assert "`id` drifts" not in rep
