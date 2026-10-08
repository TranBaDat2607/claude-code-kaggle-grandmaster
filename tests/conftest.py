import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1] / "plugins" / "kaggle-grandmaster"
sys.path.insert(0, str(PLUGIN_ROOT))


@pytest.fixture
def plugin_root() -> Path:
    return PLUGIN_ROOT


@pytest.fixture
def binary_df():
    rng = np.random.default_rng(0)
    n = 2000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(list("abcde"), size=n)
    logit = 1.5 * x1 - x2 + (cat == "a") * 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "id": np.arange(n),
        "x1": x1,
        "x2": x2,
        "cat": cat,
        "grp": rng.integers(0, 300, size=n),
        "target": y,
    })
