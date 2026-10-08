import numpy as np
import pytest

from kgkit import metrics as M


def test_registry_and_aliases():
    assert M.get("roc_auc").name == "auc"
    assert M.get("Quadratic-Weighted-Kappa").name == "qwk"
    with pytest.raises(KeyError):
        M.get("nope")
    assert len(M.list_metrics()) > 25


def test_directions():
    assert M.get("auc").greater_is_better
    assert not M.get("rmse").greater_is_better
    assert M.get("rmse").is_better(0.1, 0.2)
    assert M.get("auc").is_better(0.9, 0.8)


def test_values():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.8, 0.3])
    assert M.score("auc", y, p) == 1.0
    assert M.score("gini", y, p) == pytest.approx(1.0)
    assert M.score("rmse", [1, 2, 3], [1, 2, 4]) == pytest.approx(np.sqrt(1 / 3))
    assert M.score("rmsle", [0, np.e - 1], [0, np.e - 1]) == pytest.approx(0)
    assert M.score("smape", [0, 1], [0, 1]) == 0
    assert M.score("qwk", [0, 1, 2], [0, 1, 2]) == pytest.approx(1)


def test_logloss_matches_sklearn():
    from sklearn.metrics import log_loss
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    assert M.score("logloss", y, p) == pytest.approx(log_loss(y, p), rel=1e-6)
    P = rng.dirichlet(np.ones(3), 200)
    y3 = rng.integers(0, 3, 200)
    assert M.score("logloss", y3, P) == pytest.approx(log_loss(y3, P), rel=1e-6)


def test_balanced_logloss_weights_classes_equally():
    y = np.array([0] * 90 + [1] * 10)
    p = np.full(100, 0.1)
    assert M.score("balanced_logloss", y, p) > M.score("logloss", y, p)


def test_mapk():
    actual = [[1, 2], [3]]
    pred = [[1, 9, 2], [4, 3]]
    assert M.score("map@5", actual, pred) == pytest.approx(((1 + 2 / 3) / 2 + 0.5) / 2)
    assert M.ndcgk([[1]], [[1, 2]], 10) == 1.0


def test_masks():
    a = np.zeros((4, 4))
    a[:2] = 1
    assert M.score("dice", a, a) == pytest.approx(1)
    assert M.score("iou", np.zeros((2, 2)), np.zeros((2, 2))) == 1.0
