---
name: code-competitions
description: Shipping solutions to Kaggle code competitions — offline notebooks with internet disabled, packaging models/wheels/code as Kaggle datasets, hidden test set re-runs, runtime and memory budgets on T4x2/P100/L4/TPU/CPU, evaluation API patterns, and debugging submission errors (Scoring Error, timeout, OOM). Use when a competition requires notebook submission or when preparing an inference kernel.
---

# Code Competitions

In a code competition you submit a notebook; Kaggle re-runs it on the hidden test set (often much
larger than the visible sample) with internet **off**, a hard runtime limit (commonly 9h GPU / 9h
CPU, sometimes much less), fixed hardware, and an output file (`submission.csv` / parquet) or an
evaluation-API loop.

## 1. Day-one checklist

- Make a trivial end-to-end submission (sample submission copy through the full kernel) to learn
  the mechanics: where input lands (`/kaggle/input/<slug>/`), the output path, API usage.
- Note the limits in `.kaggle-gm/competition.json` (`code_competition`, `runtime_limit_hours`,
  `internet_allowed`, `gpu`).
- Decide the inference budget per model early; it constrains backbone size, folds and TTA.

## 2. Packaging (everything offline)

| Asset | How |
|---|---|
| Model weights | `kaggle datasets create -p kernels/<name>-weights` (dataset-metadata.json), version with `kaggle datasets version -p ... -m "msg"` |
| Your code (`src/`, `kgkit`) | upload as a dataset, `sys.path.append("/kaggle/input/<code-ds>")` |
| Python packages not in the image | `pip download <pkg> -d wheels/ --no-deps` (matching Python/CUDA), upload as dataset, install with `pip install --no-index --find-links /kaggle/input/<wheels-ds> <pkg>` |
| HF models/tokenizers | `save_pretrained` to a folder → dataset; load with `local_files_only=True` |
| Pretrained timm weights | save state_dict; create model with `pretrained=False` and load |

Pin the exact Kaggle docker image version used when developing (notebook settings → environment
"pin to original") so packages don't change under you.

Kernel push via CLI: `kaggle kernels init -p kernels/infer`, edit `kernel-metadata.json`
(`"enable_gpu"`, `"enable_internet": false`, `"competition_sources"`, `"dataset_sources"`,
`"model_sources"`), then `kaggle kernels push -p kernels/infer` and poll
`kaggle kernels status <user>/<kernel>`. Template: `templates/inference_kernel.py`.

## 3. Hidden test pitfalls

- Never hardcode the number of test rows/ids; read whatever test files exist at runtime.
- The visible test may be a tiny stub: make the "submit" path differ from the "interactive" path
  only by data size. Test locally with a synthetic test set of the hidden size to measure runtime.
- Categories/values unseen in train will appear — handle unknowns everywhere.
- Some competitions hide the test until rerun: guard EDA-only code so it doesn't crash.

## 4. Runtime engineering

- Time every stage on Kaggle hardware with a full-size dummy test; keep ≥20% safety margin.
- Use both T4 GPUs: split test into halves and run two processes, or one model per GPU.
- FP16/bf16 inference, larger batches, `torch.inference_mode`, sort by length (NLP), resize once.
- Free memory between models (`del model; gc.collect(); torch.cuda.empty_cache()`), stream data
  in chunks; RAM is ~29–30 GB on GPU kernels — watch pandas copies.
- vLLM/ONNX/TensorRT/OpenVINO when they give a measured gain.
- Write intermediate predictions to `/kaggle/working` so a late crash doesn't lose everything; add
  a fallback (e.g. simpler model) if the time budget is about to be exceeded (track elapsed time).

## 5. Evaluation-API competitions

Some competitions stream test batches via a provided API/gateway (`kaggle_evaluation`, inference
server pattern, or iterator yielding (test_batch, sample_prediction)). Rules: predict each batch
before getting the next; state must be updated incrementally (time series); total time limits
apply; the local gateway can be run on the provided mock data — test with it.

## 6. Debugging failed submissions

| Error | Usual cause |
|---|---|
| Notebook Threw Exception | path assumptions, missing package offline, unseen category, different test schema |
| Notebook Timeout | hidden test bigger than expected; TTA/folds too heavy; CPU-bound preprocessing |
| Out of Memory | loading all test at once, several models resident, pandas copies |
| Submission Scoring Error | wrong columns/ids/row count, NaN, wrong dtype, file name/location wrong |
Reproduce locally with a scaled-up synthetic test; add defensive logging; validate the output
with `python -m kgkit validate` logic inside the kernel (assert row counts/ids/no NaNs).
