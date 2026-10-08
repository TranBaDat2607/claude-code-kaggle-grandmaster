"""Offline inference kernel template for Kaggle code competitions.

Push as a Kaggle script (kernel_type "script") with internet disabled. It is written to survive
the hidden-test rerun:

  * discovers test files at runtime (never hardcodes row counts or ids)
  * installs extra wheels offline from an attached dataset
  * runs models in priority order under a wall-clock budget, skipping optional members when
    time is short, and always writes *some* valid submission (fallback = best single model)
  * frees memory between models and validates the output before writing it

Fill in the three hooks: load_test(), MODELS (one predict function per ensemble member) and
make_submission(). Local dry run (outside Kaggle):
    KAGGLE_INPUT=./data_like_kaggle python kernels/infer/inference_kernel.py
"""

from __future__ import annotations

import gc
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

START = time.time()
RUNTIME_LIMIT_H = 9.0          # competition limit
SAFETY = 0.80                  # use at most 80% of it
BUDGET_S = RUNTIME_LIMIT_H * 3600 * SAFETY

INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
COMP = INPUT / "COMPETITION-SLUG"          # competition data
WEIGHTS = INPUT / "my-weights"             # dataset with fold checkpoints
CODE = INPUT / "my-code"                   # dataset with src/ + kgkit/
WHEELS = INPUT / "my-wheels"               # dataset with pip wheels


def log(msg: str) -> None:
    print(f"[{time.time() - START:7.0f}s] {msg}", flush=True)


def elapsed() -> float:
    return time.time() - START


def install_offline_wheels() -> None:
    if WHEELS.exists():
        wheels = glob.glob(str(WHEELS / "*.whl"))
        if wheels:
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--quiet", *wheels],
                           check=False)
    if CODE.exists():
        sys.path.insert(0, str(CODE))


# ----------------------------------------------------------------------------- hooks to fill in
def load_test() -> pd.DataFrame:
    """Read whatever test data exists at runtime (the hidden test replaces the public stub)."""
    files = sorted(COMP.glob("test*.csv")) or sorted(COMP.glob("test*.parquet"))
    parts = [pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f) for f in files]
    return pd.concat(parts, ignore_index=True)


def predict_model_a(test: pd.DataFrame) -> np.ndarray:
    """Example member: average the fold models' predictions."""
    preds = []
    for ckpt in sorted(WEIGHTS.glob("model_a/fold*.pt")):
        # model = build_model(); model.load_state_dict(torch.load(ckpt, map_location="cuda")); ...
        preds.append(np.zeros(len(test)))
    return np.mean(preds, axis=0) if preds else np.zeros(len(test))


# (name, predict_fn, blend weight, estimated seconds per 1k test rows, required?)
MODELS = [
    ("model_a", predict_model_a, 1.0, 5.0, True),
]


def make_submission(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    sample = pd.read_csv(COMP / "sample_submission.csv") if (COMP / "sample_submission.csv").exists() else None
    id_col = sample.columns[0] if sample is not None else "id"
    target_col = sample.columns[1] if sample is not None else "target"
    return pd.DataFrame({id_col: test[id_col].to_numpy(), target_col: pred})


# ----------------------------------------------------------------------------- driver
def validate(sub: pd.DataFrame, test: pd.DataFrame) -> None:
    assert len(sub) == len(test), f"row count {len(sub)} != test {len(test)}"
    assert not sub.isna().any().any(), "NaN in submission"
    num = sub.select_dtypes("number")
    assert np.isfinite(num.to_numpy()).all(), "inf in submission"
    assert not sub.iloc[:, 0].duplicated().any(), "duplicated ids"


def main() -> None:
    install_offline_wheels()
    test = load_test()
    log(f"test rows: {len(test):,}")
    blended, total_w = None, 0.0
    for name, fn, w, sec_per_1k, required in MODELS:
        est = sec_per_1k * len(test) / 1000
        if not required and elapsed() + est > BUDGET_S:
            log(f"skip {name}: estimated {est:.0f}s would exceed budget")
            continue
        t = time.time()
        try:
            p = np.asarray(fn(test), dtype=np.float64)
        except Exception as e:  # an optional member failing must not kill the submission
            if required:
                raise
            log(f"{name} failed ({e!r}); continuing without it")
            continue
        np.save(WORK / f"pred_{name}.npy", p)  # survive a later crash
        blended = p * w if blended is None else blended + p * w
        total_w += w
        log(f"{name}: {time.time() - t:.0f}s")
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    pred = blended / total_w
    sub = make_submission(test, pred)
    validate(sub, test)
    sub.to_csv(WORK / "submission.csv", index=False)
    log(f"wrote submission.csv {sub.shape}; total {elapsed() / 60:.1f} min")


if __name__ == "__main__":
    main()
