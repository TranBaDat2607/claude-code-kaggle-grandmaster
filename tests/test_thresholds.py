import numpy as np

from kgkit import metrics as M
from kgkit.thresholds import OptimizedRounder, apply_ordinal_thresholds, best_threshold, class_prior_shift


def test_best_threshold_beats_half():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.1).astype(int)
    score = np.clip(y * 0.3 + rng.random(3000) * 0.6, 0, 1)
    _, s = best_threshold(y, score, "f1")
    assert s >= M.score("f1", y, (score >= 0.5).astype(int))


def test_optimized_rounder_improves_qwk():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 2000)
    pred = y * 0.6 + 0.8 + rng.normal(0, 0.4, 2000)  # biased + squeezed scale
    naive = M.score("qwk", y, np.clip(np.round(pred), 0, 4))
    r = OptimizedRounder(labels=[0, 1, 2, 3, 4]).fit(pred, y)
    assert r.score_ > naive
    assert set(np.unique(r.predict(pred))) <= {0, 1, 2, 3, 4}


def test_apply_ordinal_thresholds():
    assert apply_ordinal_thresholds([0.2, 1.4, 3.9], [0.5, 1.5, 2.5]).tolist() == [0, 1, 3]


def test_prior_shift_rows_sum_to_one():
    p = np.array([[0.7, 0.3], [0.4, 0.6]])
    q = class_prior_shift(p, [0.5, 0.5], [0.2, 0.8])
    assert np.allclose(q.sum(1), 1) and (q[:, 1] > p[:, 1]).all()
