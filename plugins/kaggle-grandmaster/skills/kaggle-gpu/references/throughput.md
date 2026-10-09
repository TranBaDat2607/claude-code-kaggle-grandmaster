# Throughput on Kaggle T4 / P100: more epochs per quota hour

Measure before tuning: steps/s with real data vs with a fixed in-memory fake batch. If fake data is
much faster, the input pipeline is the bottleneck, and no GPU-side trick will help.

## Precision and kernels

- **T4: fp16 autocast + GradScaler** (no native bf16). Tensor cores need fp16 and dims that are multiples of 8.
  The templates pick bf16 only when `torch.cuda.is_bf16_supported()`, which is False on T4/P100.
- P100: fp16 helps memory but has no tensor cores. The speed-up is smaller and fp32 is often similar.
- `channels_last` for CNNs (+10-30% on T4 with AMP). `torch.backends.cudnn.benchmark = True` for
  fixed input sizes.
- `torch.compile`: works on T4 (sm_75), not on P100 (Triton needs sm_70+). Compile time (1-3 min) is
  paid per process, so it only pays off for runs longer than ~20 min.
- SDPA / flash attention: PyTorch picks the memory-efficient kernel on T4. For HF models use
  `attn_implementation="sdpa"`.
- Fused AdamW (`torch.optim.AdamW(..., fused=True)`) saves a few % on many-parameter models.

## Memory → bigger batch or bigger model

- Gradient checkpointing trades ~25-30% speed for ~40-60% activation memory. Use it only if it unlocks a
  materially better model, not to raise the batch size.
- Gradient accumulation keeps the effective batch when memory limits the micro-batch.
- 8-bit optimisers (bitsandbytes) / LoRA for LLM fine-tunes on 16 GB.

## Input pipeline (the usual culprit: 4 CPU cores for 2 GPUs)

- Pre-resize offline to the training resolution. Never decode 2-4K images or DICOMs on the fly.
- Store uint8 (PNG, or npy/memmap for speed). Convert to float **on the GPU**:
  `x = x.cuda(non_blocking=True).float().div_(255)`, then normalise there.
- Move augmentation to the GPU (kornia / torchvision v2 on tensors / custom flips and crops) and keep
  only cheap ops on the CPU.
- `num_workers` = CPU cores ÷ processes (2 folds in parallel → 2 workers each), plus `pin_memory=True`
  and `persistent_workers=True`.
- If the whole preprocessed dataset fits in RAM (~29 GB shared by both processes), cache it once
  (shared memmap) and drop the per-epoch file I/O.
- NLP: tokenise once (save token ids) and sort by length with dynamic padding (the template does this
  for inference; for training use bucketed batches).

## Fold-parallel vs DDP on 2×T4

- **Fold-parallel (default in the runner):** one fold per GPU, no communication, nearly 2× throughput,
  and each process can fail or resume on its own. Best when you train ≥ 2 folds or 2 screens.
- **DDP** (`torchrun --nproc_per_node 2`, command without `{fold}`): use it for a single big model
  whose batch doesn't fit one T4, or a single full-data fit. Expect 1.6-1.8× (PCIe, no NVLink).
  The command then owns both GPUs, and the runner gives it the whole session.

## Schedules that save epochs

- Progressive resizing (low res → full res for the last 20-30%).
- Shorter cosine schedules with EMA often match longer ones. Screen the epoch count on one fold.
- Validate every epoch only when it is cheap. Large validation sets can be subsampled for the
  per-epoch curve (pruning), with a full evaluation of the best checkpoint at the end.
- Stop at the plateau: if fold 0's best epoch is consistently ≤ 70% of the schedule, shorten the
  schedule for the other folds.
