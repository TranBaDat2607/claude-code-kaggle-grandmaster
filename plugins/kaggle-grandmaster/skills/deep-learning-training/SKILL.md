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

Every training loop starts with these defaults. Load the `compute-efficiency` skill when writing or changing a
loop: it has the measurement, the per-domain how-to and what runs on which GPU.

- **Measure first**: the templates' `--bench` times the data loader alone, the training step alone and the real
  loop (`python -m kgkit bench --run "<cmd>"` for any script). Input-bound: fix the pipeline. Compute-bound: less
  compute (AMP, compile, smaller inputs).
- AMP (above), `channels_last` for CNNs, `cudnn.benchmark = True` for fixed input sizes, fused AdamW, SDPA attention.
- Inputs cached once at training size (images as uint8 npy/PNG, text as token ids). `num_workers` = CPU cores ÷
  processes (two folds on Kaggle's 4 cores: 2 each), `pin_memory`, `persistent_workers`; augmentation on the GPU when
  the loader is the bottleneck.
- `torch.compile` for runs over ~20 min on Linux with a T4 or newer (not P100, not native Windows).
- Gradient accumulation for a large effective batch; gradient checkpointing only to fit a better model.
- Kaggle 2×T4: run two folds in parallel, one per GPU (simpler, nearly 2× throughput, two results per session), or
  DDP (`torchrun --nproc_per_node 2`) for one model on both GPUs: a big model, or a result 1.6-1.9× sooner when
  wall time rather than quota binds (runs of ~6 h+, quota to spare, deadline close). Same GPU-hours either way;
  DDP needs a loop whose ranks agree on every stop (compute-efficiency `training-loop.md`, DDP checklist). No bf16
  on T4/P100. Remote runs, quota, the two regimes and resume: skill `kaggle-gpu`.
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
7. NaNs under fp16: switch to bf16 only on GPUs that have it (Ampere+, not T4/P100); otherwise keep fp16 with
   GradScaler, clamp logs, check for division by zero.

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
