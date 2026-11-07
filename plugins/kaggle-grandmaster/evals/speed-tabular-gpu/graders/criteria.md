---
type: llm
---

PASS when the response:

1. Moves the feature code to GPU dataframes, cuDF or `cudf.pandas` (or makes it faster with polars), and says RAPIDS
   needs a GPU of compute capability 7.0 or newer: the T4 works, the P100 does not.
2. Trains GBDTs on the GPU: XGBoost with `device="cuda"` (hist) and/or CatBoost with `task_type="GPU"`. LightGBM needs
   its GPU build for that, or stays on the CPU with faster settings (threads, max_bin).
3. Uses both T4s, one fold or model per GPU in parallel.
4. Measures before moving long jobs (one fold on CPU vs GPU, or profiling the pandas code) and keeps memory in check
   (float32, categories, only the needed columns).

FAIL if it recommends the P100 for cuDF/cuML, or if it gives only generic advice (subsample, fewer features) with no
GPU path.
