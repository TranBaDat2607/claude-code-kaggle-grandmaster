---
name: deep-learning-training
description: Competition-grade deep learning training practice in PyTorch — schedules, optimisers, mixed precision, EMA/SWA, gradient accumulation and checkpointing, multi-GPU on Kaggle (2xT4), data loading speed, reproducibility, debugging loss curves, and making the most of limited GPU hours. Use when writing or debugging any neural network training loop for a competition.
---

# Deep Learning Training for Competitions

## Default recipe

- Optimiser AdamW (β 0.9/0.999, wd 1e-2; no wd on norms/biases), cosine decay with linear warmup
  (3–10% of steps), peak LR found by a short LR range test or known defaults (CNN fine-tune
  1e-4–5e-4, transformers 1e-5–3e-5, heads ×10).
- Mixed precision: `torch.autocast` bf16 on Ampere+ (A100/L4/H100), fp16 + GradScaler on T4/P100.
- Gradient clipping (max norm 1.0) for transformers/RNNs.
- **EMA of weights** (decay 0.999–0.9999) — cheap, frequently +, evaluate the EMA model.
- Checkpoint the best-by-metric (not by loss) per fold; also keep last for SWA if used.
- Evaluate more often than once per epoch for short schedules.

## Speed (more experiments per GPU-hour)

- Profile first: is the GPU waiting on data? (`nvidia-smi` utilisation, step time with fake data).
- Pre-resize/cache inputs (images to target res as JPEG/PNG/npy; tokenise once), `num_workers` =
  CPU cores, `pin_memory`, `persistent_workers`, larger batches.
- `torch.compile`, `channels_last` for CNNs, fused AdamW, SDPA/flash attention.
- Gradient accumulation for large effective batch; gradient checkpointing to fit bigger models.
- Kaggle 2×T4: use DDP (`torchrun --nproc_per_node 2` via `accelerate`) or run two folds in
  parallel, one per GPU — the latter is simpler and nearly 2× throughput. No bf16 on T4/P100, and no
  `torch.compile` on P100. Remote runs, quota and resume: skill `kaggle-gpu`.
- Lower resolution / shorter max_len / subset of data for exploration; scale for finals.

## Debugging checklist (when the score is bad)

1. Overfit a single batch to ~0 loss. If impossible → bug in labels, loss, or model output shape.
2. Visualise inputs *after* augmentation and normalisation; check label alignment.
3. Check train vs val transforms, `model.eval()` + `torch.inference_mode()` at validation,
   BatchNorm with tiny batches (use GroupNorm/freeze BN), dropout off at eval.
4. Learning rate too high (loss spikes/NaN) or too low (flat). Re-check warmup and scheduler
   stepping (per step vs per epoch).
5. Loss/metric mismatch: e.g. sigmoid applied twice, softmax + CrossEntropyLoss, wrong
   `pos_weight`, logits vs probabilities in metric.
6. Data leakage between train/val (too-good val) or distribution mismatch (val ≫ LB).
7. NaNs under fp16: use bf16, scale loss, clamp logs, check for division by zero.

## Reproducibility & bookkeeping

- `kgkit.experiment.seed_everything(seed)`; record seed, config, git hash in the ledger.
- Config as a dict/dataclass dumped next to checkpoints; one script = one experiment family with
  flags, not copy-pasted notebooks.
- Save OOF predictions (per fold, then concatenated) and test predictions for ensembling.
- Weights & Biases / CSV logs for curves; the ledger for decisions.

## Regularisation menu

Augmentation (first), dropout / drop-path (stochastic depth 0.1–0.3 for ViTs/ConvNeXts), label
smoothing 0.05–0.1, mixup/cutmix, weight decay, early stopping, smaller LR for backbone,
layer-wise LR decay, AWP/FGM adversarial perturbations (NLP), SWA over last epochs.
