import numpy as np
import pandas as pd
import pytest

from kgkit import cv


@pytest.mark.parametrize("strategy", ["kfold", "stratified", "group", "stratified_group"])
def test_assign_folds_covers_all_rows(binary_df, strategy):
    folds = cv.assign_folds(binary_df, 5, strategy, target="target", group="grp")
    assert folds.min() == 0 and folds.max() == 4
    assert (folds >= 0).all()


@pytest.mark.parametrize("strategy", ["group", "stratified_group"])
def test_group_strategies_never_leak(binary_df, strategy):
    df = binary_df.copy()
    df["fold"] = cv.assign_folds(df, 5, strategy, target="target", group="grp")
    assert (df.groupby("grp")["fold"].nunique() == 1).all()
    assert "GROUP LEAKAGE" not in cv.fold_report(df, "fold", "target", "grp")


def test_report_flags_leakage(binary_df):
    df = binary_df.copy()
    df["fold"] = cv.assign_folds(df, 5, "kfold")
    assert "GROUP LEAKAGE" in cv.fold_report(df, "fold", "target", "grp")


def test_stratified_balances_target(binary_df):
    folds = cv.assign_folds(binary_df, 5, "stratified", target="target")
    means = binary_df.groupby(folds)["target"].mean()
    assert means.max() - means.min() < 0.01


def test_auto_picks_regression_bins():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.lognormal(size=1000)})
    folds = cv.assign_folds(df, 5, "auto", target="y")
    med = df.groupby(folds)["y"].median()
    assert med.max() / med.min() < 1.3


def test_seed_changes_assignment(binary_df):
    a = cv.assign_folds(binary_df, 5, "group", group="grp", seed=1)
    b = cv.assign_folds(binary_df, 5, "group", group="grp", seed=2)
    assert (a != b).any()


def test_multilabel_iterative_stratification():
    rng = np.random.default_rng(0)
    Y = (rng.random((1500, 6)) < [0.5, 0.2, 0.05, 0.02, 0.1, 0.01]).astype(int)
    df = pd.DataFrame(Y, columns=list("abcdef"))
    folds = cv.assign_folds(df, 5, target=list("abcdef"))
    assert set(folds) == {0, 1, 2, 3, 4}
    for col in "abcdef":
        per_fold = df.groupby(folds)[col].sum()
        assert per_fold.max() - per_fold.min() <= max(2, 0.1 * per_fold.mean())


def test_time_series_splits_respect_order_and_gap():
    t = np.repeat(np.arange(100), 3)
    splits = cv.time_series_splits(t, n_splits=4, gap=5, test_size=10)
    assert len(splits) == 4
    for tr, va in splits:
        assert t[tr].max() + 5 < t[va].min()
    assert t[splits[-1][1]].max() == 99


def test_purged_kfold_embargo():
    t = np.arange(100)
    for tr, va in cv.purged_kfold_splits(t, n_splits=5, embargo=3):
        lo, hi = t[va].min(), t[va].max()
        near = (t[tr] >= lo - 3) & (t[tr] <= hi + 3)
        assert not near.any()


def test_split_indices_skips_negative_folds():
    folds = np.array([0, 1, 2, -1, 0, 1, 2])
    out = list(cv.split_indices(folds))
    assert [f for f, _, _ in out] == [0, 1, 2]
    assert all(3 not in va for _, _, va in out)
