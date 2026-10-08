import numpy as np
import pandas as pd

from kgkit import adversarial as A


def test_detects_shift_and_culprit():
    rng = np.random.default_rng(0)
    tr = pd.DataFrame({"same": rng.normal(size=1500), "shift": rng.normal(0, 1, 1500)})
    te = pd.DataFrame({"same": rng.normal(size=1500), "shift": rng.normal(1.5, 1, 1500)})
    res = A.adversarial_validation(tr, te, n_splits=3)
    assert res["auc"] > 0.75
    assert res["importance"].iloc[0]["feature"] == "shift"
    assert len(res["p_test"]) == len(tr)
    w = A.importance_weights(res["p_test"])
    assert abs(w.mean() - 1) < 1e-9
    assert A.testlike_holdout_mask(res["p_test"], 0.2).mean() > 0.15


def test_no_shift_is_near_half():
    rng = np.random.default_rng(1)
    tr = pd.DataFrame({"a": rng.normal(size=1500), "c": rng.choice(list("xyz"), 1500)})
    te = pd.DataFrame({"a": rng.normal(size=1500), "c": rng.choice(list("xyz"), 1500)})
    res = A.adversarial_validation(tr, te, n_splits=3)
    assert res["auc"] < 0.6
