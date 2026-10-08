import numpy as np
import pytest

from kgkit import ensemble as E
from kgkit import metrics as M


def _sig(z):
    return 1 / (1 + np.exp(-z))


@pytest.fixture
def oofs():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 3000)
    signal = y + rng.normal(0, 1, 3000)
    a = signal + rng.normal(0, 1.0, 3000)
    b = signal + rng.normal(0, 1.0, 3000)
    noise = rng.normal(0, 1, 3000)
    return y, {"a": _sig(a), "b": _sig(b), "noise": _sig(noise)}


def test_hill_climb_beats_singles_and_ignores_noise(oofs):
    y, P = oofs
    res = E.hill_climb(P, y, "auc")
    assert res["score"] > max(res["single_scores"].values())
    assert res["weights"].get("noise", 0) < 0.1
    assert sum(res["weights"].values()) == pytest.approx(1)


def test_optimize_weights_logloss(oofs):
    y, P = oofs
    res = E.optimize_weights(P, y, "logloss")
    # for logloss a near-0.5 "noise" model can legitimately act as calibration shrinkage
    assert sum(res["weights"].values()) == pytest.approx(1, abs=1e-6)
    assert res["score"] <= min(M.score("logloss", y, p) for p in P.values()) + 1e-9


def test_cv_blend_score_is_honest(oofs):
    y, P = oofs
    folds = np.arange(len(y)) % 5
    honest = E.cv_blend_score(P, y, folds, "auc")
    full = E.hill_climb(P, y, "auc")
    assert honest["oof_score"] <= full["score"] + 1e-3
    assert len(honest["fold_scores"]) == 5


def test_blend_methods(oofs):
    y, P = oofs
    for method in ["mean", "rank", "geometric", "power2"]:
        out = E.blend(P, {"a": 0.5, "b": 0.5}, method)
        assert out.shape == y.shape and np.isfinite(out).all()


def test_stack_and_corr(oofs):
    y, P = oofs
    folds = np.arange(len(y)) % 5
    res = E.stack(P, P, y, folds)
    assert res["task"] == "classification"
    assert M.score("auc", y, res["oof"]) > 0.7
    corr = E.oof_correlation(P)
    assert corr.loc["a", "noise"] < 0.2
