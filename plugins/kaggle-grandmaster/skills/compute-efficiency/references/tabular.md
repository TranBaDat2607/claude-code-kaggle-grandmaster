# Tabular and large dataframes

Contents: [Where the time goes](#where-the-time-goes) · [GPU dataframes](#gpu-dataframes) · [GBDTs](#gbdts) ·
[cuML and GPU tabular models](#cuml-and-gpu-tabular-models) ·
[Feature search and hill climbing](#feature-search-and-hill-climbing) ·
[Recsys and time-series pipelines](#recsys-and-time-series-pipelines) · [Memory](#memory)

## Where the time goes

Split feature building from model fitting before optimising either:

```bash
python -m kgkit bench --run "python src/features.py"
python -m kgkit bench --run "python src/train_gbdt.py --smoke" --run "python src/train_gbdt.py --smoke --device cuda"
python -m cudf.pandas --profile src/features.py    # which pandas calls ran on the GPU, which fell back to the CPU
```

The profiler needs RAPIDS (Linux or WSL2, a GPU with compute capability 7.0+).

## GPU dataframes

- **`cudf.pandas`**: no code change. In a notebook, run `%load_ext cudf.pandas` before importing pandas. For a script,
  run `python -m cudf.pandas src/features.py`. Operations cuDF supports run on the GPU, and the rest fall back to
  pandas on the CPU (with copies), so check the profiler output.
- It pays on groupby-aggregate, merge, sort, value_counts and string operations over millions of rows. Insurance
  Playground (1st) evaluated about 145,000 column combinations as groupby features with cuDF-pandas.
- It does not pay on small frames (below about 100k rows) or on Python UDFs and row loops (`apply(axis=1)`,
  `iterrows`). Vectorise those first: they are slow on any hardware.
- Requirements: Linux or WSL2 and a GPU with compute capability 7.0+ (T4 yes, P100 no). Recent Kaggle images ship
  RAPIDS; check with `import cudf; print(cudf.__version__)`. A T4 has 16 GB: process larger frames in chunks.
- **polars** is fast on the CPU (multi-threaded, lazy), which makes it the best choice on Kaggle's 4 cores and on
  Windows. `lazy_frame.collect(engine="gpu")` runs on the GPU with `polars[gpu]` (Linux).
- **cupy** for custom array maths. USPTO (1st) intersected sorted id sets with `cp.intersect1d` on the GPU, and Ariel
  2025 (1st) did its signal processing in cupy.

## GBDTs

| Library | GPU | CPU speed settings |
|---|---|---|
| XGBoost | `tree_method="hist", device="cuda"` (`gpu_hist` is deprecated) | `tree_method="hist"`, `nthread` = cores, `max_bin` 64-256 |
| CatBoost | `task_type="GPU", devices="0"` | `thread_count=-1`, `border_count` 128-254 |
| LightGBM | `device_type="gpu"` (OpenCL, in the PyPI wheel on Linux and Windows; needs an OpenCL driver) or `device_type="cuda"` (source build, Linux), with `max_bin=63` | `num_threads` = physical cores, `max_bin` 63-127, `force_col_wise=True` (many columns) or `force_row_wise=True` (many rows), one `lgb.Dataset` built once and reused across seeds |

- Measured in a Kaggle T4 x2 session (November 2026): XGBoost, 5 folds x 300 rounds on 600k rows x 23 features,
  took 94 s on the session's 4 CPU cores and 32 s on one T4 (2.9x faster, with the GPU only 29 % busy). LightGBM's
  PyPI wheel trained on the T4 through OpenCL (`device_type="gpu"`), and CatBoost's `task_type="GPU"` worked.
- Mushroom Playground (1st): setting LightGBM's `num_threads` to the number of CPU cores sped it up. The default
  uses every hardware thread, and LightGBM's docs recommend physical cores.
- GPU and CPU trees differ slightly, and a few options exist on one device only (CatBoost names them in its error).
- After a GPU fit, predict on the device the data is on. `train_gbdt.py` switches XGBoost to CPU prediction for
  pandas input, which avoids a copy per call.
- One GPU per GBDT process: on 2xT4, run two folds or seeds at once (`CUDA_VISIBLE_DEVICES=0` and `1`). The runner
  does this per fold.
- LightGBM's GPU build gains little below a few million rows. Bench one fold both ways.
- On a slow metric, evaluate early stopping every N rounds (XGBoost and CatBoost `metric_period`, LightGBM
  `log_evaluation` with a custom callback).

## cuML and GPU tabular models

- `python -m cuml.accel src/train_svr.py`, or `%load_ext cuml.accel` before importing sklearn, runs the sklearn
  estimators cuML supports on the GPU without code changes. The others stay on the CPU.
- Direct imports: `cuml.svm.SVR`, `cuml.neighbors.KNeighborsRegressor`, `cuml.ensemble.RandomForestClassifier`,
  `cuml.linear_model`, `cuml.manifold.UMAP`.
- Playground winners: Podcast 2025 (1st) fitted cuML Lasso and SVR on 6,000 engineered features. Fertilizer 2025 (1st)
  passed a cuML linear model's logits to XGBoost as `base_margin`. Churn 2026 (1st) added cuML random forests and a
  cuML kNN graph for a GraphSAGE member.
- TabPFN, TabM and RealMLP run on the GPU. Podcast 2025 (1st) ran TabPFN 20 times on 10,000-row subsamples.

## Feature search and hill climbing

- `kgkit features search --model xgb --device cuda` screens candidate features with GPU XGBoost. On the CPU, keep
  `--model lgbm` and screen on fewer folds (`--screen-folds 0,1`). Generate candidates with cuDF when the frame is
  large.
- Hill climbing over hundreds of OOFs: evaluate the metric for all candidates at once with cupy or torch on the GPU.
  Calorie 2025 (1st) fed hundreds of GBDT, NN and cuML models into GPU hill climbing. `kgkit blend --prune` cuts
  near-duplicates first.

## Recsys and time-series pipelines

- Columnar, lazy and typed: polars LazyFrames or cuDF, int32 ids, float32 values, categorical strings, and parquet
  partitioned by week, day or user shard.
- Co-visitation and candidate counts: self-join sessions in chunks of users on the GPU (cuDF), as the widely used
  OTTO candidate notebooks do.
- Time series: build features per day or per series and write them into a preallocated float32 array instead of
  concatenating DataFrames, which doubles memory. Optiver Trading at the Close (1st) stored one h5 file per day and
  kept the top 300 features by importance so that memory allowed more online-training rounds.
- Online retraining under an API time limit: the refit cadence and the feature count set the runtime, so measure
  the time per step against the limit.

## Memory

- float64 to float32, int64 to int32/int16, object to category: frames 2-8x smaller, and every operation faster.
  Check value ranges before downcasting to float16.
- Read only the columns you need (`usecols=`, `columns=`), from parquet rather than CSV.
- Delete intermediate frames after joins (`del df; gc.collect()`). `inplace=True` does not save memory.
