---
name: compute-efficiency
description: Speed defaults for Kaggle training and inference by domain (images, text and LLMs, tabular with cuDF/cuML/GPU GBDTs, audio, recsys, time series) and hardware (T4x2, P100, TPU, CPU-only kernels, Windows): measuring idle, input-bound or compute-bound runs, mixed precision, data loading, torch.compile, ONNX/OpenVINO/TensorRT/vLLM and quantisation. Use whenever training or inference code is written or changed, and when a run is slow, a GPU idles or runs out of memory, or a kernel must fit a time limit.
---

# Compute efficiency: more experiments per GPU-hour

Quota hours and runtime limits are fixed. A loop that runs twice as fast buys twice the experiments, and an
inference kernel that runs twice as fast can carry twice the ensemble. Treat speed work like any other experiment:

1. **Measure where the time goes** (section 2) before changing anything. Most slow Kaggle runs are input-bound
   (4 CPU cores feed two T4s) or leave the GPU idle, and GPU-side tricks do nothing for either.
2. **Write the domain defaults in from the start** (section 1). They do not cost score.
3. **Change one thing, measure the speed-up, and check that the score holds** with a paired comparison on one fold.
   fp16 overflow, a lower resolution, a shorter `max_len`, int4 weights or a different resize can all cost score.

## 1. Defaults by domain

Write these into every new training or inference script. The reference file for each domain has the how-to, the
conditions, and evidence from winning solutions.

| Domain | Defaults (always) | When measured, or when the data is large | Reference |
|---|---|---|---|
| Images, 3D volumes, video | AMP (fp16 + GradScaler on T4/P100, bf16 on Ampere+). `channels_last`. `cudnn.benchmark = True` for fixed input sizes. Inputs cached once as uint8 at the training resolution (npy/PNG/JPEG), never decoded from 2-4K images or DICOM every epoch. `num_workers` = CPU cores / processes, with `persistent_workers`. Fused AdamW. One fold per GPU on 2xT4 | augmentation and normalisation on the GPU; GPU JPEG decode (nvJPEG, DALI on Linux); `torch.compile` for long runs (Linux, sm_70+); progressive resizing | [vision.md](references/vision.md) |
| Text encoders (DeBERTa-style) | AMP. Tokenise once. Length-bucketed batches with dynamic padding. `max_len` from the token-length distribution. Fused AdamW. SDPA where the architecture supports it | gradient checkpointing only to fit a better model; a token-id cache on disk for large corpora | [text-and-llm.md](references/text-and-llm.md) |
| LLM fine-tuning | LoRA/QLoRA on all linear layers, fp16 on T4 (bf16 on Ampere+), gradient checkpointing, packed or length-bucketed sequences | Unsloth to fit 14B on 16 GB; train off Kaggle when the rules allow and ship the adapters | [text-and-llm.md](references/text-and-llm.md) |
| Tabular, GBDTs | float32 and categorical dtypes. XGBoost `tree_method="hist"`. LightGBM `num_threads` = physical cores. On a GPU: XGBoost `device="cuda"`, CatBoost `task_type="GPU"` | `cudf.pandas` for groupby/merge-heavy feature code on millions of rows; cuML for kNN/SVR/RF/linear models on big data; GPU feature search and hill climbing; LightGBM's GPU build | [tabular.md](references/tabular.md) |
| Recsys, time series, large logs | polars or cuDF columnar code, int32 ids, parquet, work per shard or per day into a preallocated float32 array | co-visitation counts on the GPU (cuDF/cupy); feature selection to fit in RAM | [tabular.md](references/tabular.md) |
| Audio and signals | spectrograms on the GPU inside the model (torchaudio); waveforms decoded once at the model's sample rate | CPU-only inference: ONNX, then OpenVINO fp16 with a fixed input shape, one spectrogram per window shared by all models, multiprocessing | [audio-and-signal.md](references/audio-and-signal.md) |
| Inference kernels (code competitions) | `torch.inference_mode()`, fp16, large batches sorted by length, both T4s (one process per GPU), a timing run on a hidden-test-sized input with >= 20 % margin | TensorRT / ONNX Runtime / `torch.compile` / OpenVINO exports; vLLM or LMDeploy with AWQ, GPTQ or W8A8 for LLMs; prefix caching; early stopping on consensus | [inference.md](references/inference.md) |

The general PyTorch loop (precision, memory, input pipeline, fold-parallel vs DDP, schedules that save epochs) is in
[training-loop.md](references/training-loop.md). Which winners used what, with sources, is in
[evidence.md](references/evidence.md).

## 2. Measure: idle, input-bound or compute-bound

- **DL templates:** `python src/train_image.py --bench` (or `train_transformer.py --bench`) times three loops on the
  first fold: the data loader alone, the training step alone on batches already on the device, and the real
  loop. It prints samples/s for each, the verdict and the peak GPU memory, then the speed-up against the previous
  bench of the same run name and the settings that changed (kept in `reports/bench.jsonl`). It trains nothing and
  logs nothing to the ledger.
- **Any command** (GBDT, cuDF features, an inference script): `python -m kgkit bench --run "<command>"` runs it,
  samples GPU utilisation and memory with `nvidia-smi` and prints the wall time and a verdict. With a second
  `--run` it prints the speed-up of each run against the first. Use reduced settings (one fold, fewer rounds, a
  slice of the test set) so a run takes minutes.
- **In your own script:** `kgkit.bench.measure()` and `kgkit.bench.report()` do what `--bench` does
  ([training-loop.md](references/training-loop.md#measuring)).
- **Remote Kaggle runs:** `kgkit gpu collect` reports utilisation per GPU (`kaggle-gpu` section 5). With no local GPU,
  a short bench session costs minutes of quota instead of a screen's hours:
  `gpu build --name bench --cmd "python src/train_image.py --bench" --hours 0.3`, then `gpu push` and
  `gpu wait <kernel> --collect`. Without `{fold}` the command runs once, with both GPUs visible. The session saves
  `reports/`, and `gpu collect` merges its bench records into the local `reports/bench.jsonl`, printing each one's
  speed-up against the previous bench there.

| What you see | Verdict | Fix first |
|---|---|---|
| GPU utilisation near 0, or the run is no faster with a GPU | **idle**: the work runs on the CPU | device flags (`.to("cuda")`, XGBoost `device="cuda"`), `torch.cuda.is_available()`; if the job is CPU work, run it in a CPU session, which spends no GPU quota |
| data-only rate below the compute-only rate; utilisation < 60 % | **input-bound** | inputs cached as uint8 at training size, more workers (up to cores / processes), augmentation on the GPU, tokenise once |
| real-loop rate close to the compute-only rate; utilisation > 85 % | **compute-bound** | AMP, `channels_last`, `torch.compile`, a smaller resolution / `max_len` / backbone, a larger batch |
| real loop well below both rates | **stalls** in the step | `.item()` / `.cpu()` syncs every step, logging or validation too often, tiny batches, metrics computed on the CPU |
| second T4 at 0 % | half the session wasted | two folds or two screens in parallel (`kaggle-gpu`) |
| out of memory | memory-bound | micro-batch + gradient accumulation, AMP, gradient checkpointing, an 8-bit optimiser, shorter sequences |

## 3. What runs where

| | Kaggle T4 x2 | Kaggle P100 | Kaggle TPU v5e-8 | CPU-only kernel | Local Windows | Local Linux / WSL2 |
|---|---|---|---|---|---|---|
| fp16 AMP | yes (tensor cores) | saves memory, little speed | use bf16 | no | yes (NVIDIA GPU) | yes |
| bf16 | no (emulated, slow) | no | native | - | Ampere+ GPUs | Ampere+ GPUs |
| `torch.compile` | yes | no (Triton needs sm_70+) | XLA instead | possible, rarely worth it | no official Triton: skip it or use WSL2 | yes |
| RAPIDS cuDF / cuML | yes (check with `import cudf`) | no (needs compute capability 7.0+) | no | no | WSL2 only | yes, Volta or newer |
| XGBoost / CatBoost GPU | yes | yes | no | - | yes | yes |
| LightGBM GPU | OpenCL build in the pip wheel when an OpenCL driver is visible; CUDA build from source | as T4 | no | - | OpenCL (pip wheel) | OpenCL or CUDA build |
| vLLM | yes, `dtype="half"` | no (needs sm_70+) | separate TPU build | no | WSL2 only | yes |
| TensorRT, ONNX Runtime GPU | yes (build engines on a T4) | use the T4 | no | - | yes | yes |
| OpenVINO, ONNX Runtime CPU | - | - | - | yes | yes | yes |

- A Kaggle GPU session has 4 CPU cores and about 30 GB of RAM for both GPUs. CPU decoding and augmentation are the
  usual bottleneck, and every pandas copy counts.
- P100: the Kaggle image of November 2026 (PyTorch 2.11, CUDA 13) has dropped `sm_60` (its arch list starts at
  `sm_75`), so PyTorch cannot use a P100 there. Check `torch.cuda.get_arch_list()` before choosing one. With fp16 bringing little speed and no compile, RAPIDS or vLLM, the T4 x2 is the default
  choice.
- CPU sessions spend no GPU quota. Work that measures as CPU-bound (pandas features, LightGBM on CPU, tokenising)
  belongs there.

## 4. When the GPU is worth the quota

- **GBDTs:** below about 100k rows the CPU finishes in minutes. GPU XGBoost and CatBoost pay from about a million
  rows, sooner against a Kaggle session's 4 cores (2.9x on 600k rows, [tabular.md](references/tabular.md#gbdts)),
  and for many seeds, deep trees, or feature search that refits hundreds of times. Bench one fold both
  ways (`kgkit bench --run ... --run "... --device cuda"`) before moving a long job.
- **Dataframes:** cuDF pays on groupby, merge and sort over millions of rows. Small frames lose to transfer overhead,
  and Python row loops (`apply(axis=1)`, `iterrows`) are slow everywhere: vectorise them first.
- **sklearn models:** SVR, SVC, kNN and random forests on 100k+ rows go from hours to minutes with cuML.
  `python -m cuml.accel script.py` runs unchanged sklearn code on the GPU where cuML supports the estimator.
- **Deep learning:** always on the GPU. The only question is whether the GPU is busy (section 2).

## 5. Windows

- Native Windows supports CUDA PyTorch with AMP, XGBoost and CatBoost on the GPU, LightGBM's OpenCL GPU wheel, ONNX
  Runtime, OpenVINO and bitsandbytes. RAPIDS, vLLM, DALI and Triton-backed `torch.compile` need Linux: use WSL2, or
  run the job on Kaggle.
- DataLoader workers start with `spawn`. Keep the training code under `if __name__ == "__main__":`, keep datasets
  picklable, and set `persistent_workers=True` so the workers start only once.
- Environment variables in PowerShell: `$env:CUDA_VISIBLE_DEVICES = "0"; python src/train_image.py --folds 0`.

## 6. What the templates do

| Template | On by default | Opt-in | Why it is not the default |
|---|---|---|---|
| `train_image.py` | AMP (bf16 when supported, else fp16 + GradScaler), `channels_last` model and inputs, `cudnn.benchmark`, fused AdamW on CUDA, pinned memory and persistent workers, EMA, `--bench` | `--compile` | compiling costs 1-3 min per process and needs Linux with an sm_70+ GPU. The template decodes images with PIL every epoch, so pre-resize them to the training size first ([vision.md](references/vision.md)) |
| `train_transformer.py` | AMP with fp32 master weights (the backbone is loaded with `.float()`: transformers 5 keeps a checkpoint's fp16/bf16), tokenising once, length-bucketed training batches with dynamic padding, length-sorted inference, fused AdamW on CUDA, `--bench` | `--no-bucket` (plain shuffled batches) | - |
| `train_gbdt.py` | LightGBM on all CPU threads, XGBoost `hist` | `--device cuda`: XGBoost and CatBoost on the GPU, LightGBM's GPU build when one is installed (otherwise CPU, with a note) | GPU trees differ slightly from CPU trees, and LightGBM's GPU build is not on every machine. Threads: the standard library cannot count physical cores, so `n_jobs=-1` stays; on a hyper-threaded CPU try `os.cpu_count() // 2` and keep it if `kgkit bench` says it is faster |
| `inference_kernel.py` | a time budget, members in priority order, a fallback submission | exports (TensorRT, ONNX, OpenVINO) in the predict functions | an export is worth its risk only with a measured gain ([inference.md](references/inference.md)) |

When a script departs from these defaults on purpose (for example fp32 for a model that diverges in fp16), say why
in a comment, so the next experiment does not "fix" it.
