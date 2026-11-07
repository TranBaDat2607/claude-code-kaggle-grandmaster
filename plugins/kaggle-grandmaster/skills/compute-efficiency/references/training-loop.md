# The PyTorch training loop: more epochs per GPU-hour

Contents: [Measuring](#measuring) · [Precision and kernels](#precision-and-kernels) ·
[Memory](#memory-bigger-batch-or-bigger-model) · [Input pipeline](#input-pipeline) ·
[Two GPUs](#two-gpus-fold-parallel-vs-ddp-on-2xt4) · [Schedules that save epochs](#schedules-that-save-epochs) ·
[Stalls in the step](#stalls-in-the-step)

## Measuring

Measure before tuning: samples/s of the data loader alone, of the training step alone on batches already on the
device, and of the real loop. If the loader alone is slower than the step, the input pipeline is the bottleneck
and no GPU-side trick will help.

The templates do this with `--bench`. In your own script:

```python
from kgkit import bench as B

it = B.forever(train_loader)                      # endless batches, without caching them in memory
fixed = [t.to(device, non_blocking=True) for t in next(it)]

def data_step():
    x, y = next(it)
    return len(x)

def compute_step():
    train_step(*fixed)                            # forward + backward + optimiser step
    return len(fixed[0])

def full_step():
    x, y = next(it)
    train_step(x.to(device, non_blocking=True), y.to(device, non_blocking=True))
    return len(x)

rates = B.measure({"data": data_step, "compute": compute_step, "full": full_step}, n=30,
                  sync=torch.cuda.synchronize if device.type == "cuda" else None)
B.report("my_run", rates, config={"img_size": 384, "batch_size": 32, "num_workers": 4})
```

Each step function runs one batch and returns the number of samples it processed. `report` prints the verdict and
the speed-up against the previous bench of the same name, and lists the config keys that changed.

## Precision and kernels

- **T4: fp16 autocast + GradScaler** (no native bf16). Tensor cores need fp16 and dimensions that are multiples of 8
  (channels, hidden sizes, batch x sequence). The templates pick bf16 only on compute capability 8.0+
  (`torch.cuda.get_device_capability()[0] >= 8`). Do not choose the dtype with `torch.cuda.is_bf16_supported()`:
  recent PyTorch returns True on T4 and P100 too, where bf16 is emulated and slow, unless it is called with
  `including_emulation=False`. `kgkit speedcheck` flags that call. Measured in a Kaggle T4 session (November 2026,
  PyTorch 2.11), resnet50 training at 224 px ran 343 images/s in fp16 + GradScaler, 99 in fp32 and 43 in bf16.
- **fp32 master weights.** Autocast runs the math in fp16 or bf16, but the weights the optimizer updates stay fp32.
  transformers 5 loads a model in its checkpoint's dtype: fp16 for DeBERTa-v3, bf16 for most LLMs. For full
  fine-tuning, load with `from_pretrained(...).float()` (or `dtype=torch.float32`). Otherwise GradScaler fails with
  "Attempting to unscale FP16 gradients", and bf16 weights silently drop updates smaller than their precision.
  LoRA and QLoRA bases stay fp16 or 4-bit, because they are frozen. `kgkit speedcheck` flags a model loaded
  without a dtype in a training script.
- P100: fp16 saves memory but there are no tensor cores, so the speed-up is small and fp32 is often similar.
- Ampere and newer (A100, L4, H100, RTX 30xx+): bf16 autocast without a GradScaler.
  `torch.set_float32_matmul_precision("high")` lets fp32 matmuls use TF32.
- `channels_last` for CNNs (+10-30 % on T4 with AMP): convert both the model and each input batch.
- `torch.backends.cudnn.benchmark = True` when input sizes are fixed. It re-tunes for every new shape, so it hurts
  when sizes vary.
- `torch.compile(model)`: T4 (sm_75) and newer, on Linux. Not on P100 (Triton needs sm_70+) and not on native
  Windows. Compiling takes 1-3 min per process, so it pays for runs longer than about 20 min, and variable shapes
  trigger recompiles (bucket them, or pass `dynamic=True`). It gave about 1.5x on an RL agent (Lux AI S3 1st) and
  about 2x on 3D segmentation inference (SenNet 1st); nnU-Net compiles by default on CUDA.
- SDPA / flash attention: `F.scaled_dot_product_attention` uses the memory-efficient kernel on T4
  (FlashAttention-2 needs Ampere+). Hugging Face models use SDPA by default where the architecture supports it;
  DeBERTa-v2/v3 has its own attention and no SDPA path.
- Fused AdamW (`torch.optim.AdamW(..., fused=True)` on CUDA) saves a few % on models with many parameters.

## Memory: bigger batch or bigger model

- AMP halves activation memory.
- Gradient checkpointing trades about 25-30 % speed for 40-60 % of activation memory. Use it when it unlocks a
  materially better model, not to raise the batch size.
- Gradient accumulation keeps the effective batch when memory limits the micro-batch.
- 8-bit optimisers (bitsandbytes) and LoRA for LLM fine-tunes on 16 GB.
- `--bench` prints `torch.cuda.max_memory_allocated()`. Fill the headroom with a larger batch only if the score holds
  (scale the learning rate with the batch).

## Input pipeline

The usual culprit: a Kaggle GPU session has 4 CPU cores for two GPUs.

- Pre-resize offline to the training resolution. Never decode 2-4K images or DICOMs on the fly.
- Store uint8 (PNG, or npy/memmap for speed) and convert to float **on the GPU**:
  `x = x.cuda(non_blocking=True).float().div_(255)`, then normalise there.
- Move augmentation to the GPU (flips, crops, MixUp/CutMix on the batch, kornia or torchvision v2 on tensors) and keep
  only cheap ops on the CPU.
- `num_workers` = CPU cores / processes (two folds in parallel on 4 cores: 2 workers each), with `pin_memory=True`,
  `persistent_workers=True` and a `prefetch_factor` of 2-4.
- If the preprocessed dataset fits in RAM (about 30 GB, shared by both processes on Kaggle), load it once (a shared
  memmap) and drop the per-epoch file reads.
- NLP: tokenise once (save the token ids) and use length-bucketed batches with dynamic padding. The transformer
  template does both.

## Two GPUs: fold-parallel vs DDP on 2xT4

- **Fold-parallel (the runner's default):** one fold per GPU, no communication, nearly 2x throughput, and each
  process can fail or resume on its own. Best when you train >= 2 folds or 2 screens.
- **DDP** (`torchrun --nproc_per_node 2`, a command without `{fold}`): for a single big model whose batch does not fit
  one T4, or a single full-data fit. Expect 1.6-1.8x (PCIe, no NVLink). The command then owns both GPUs, and the
  runner gives it the whole session.
- TPU v5e-8 when the GPU quota is gone: `kaggle-gpu` section 6 (bf16, XLA, the queue).

## Schedules that save epochs

- Progressive resizing: low resolution first, full resolution for the last 20-30 %.
- Shorter cosine schedules with EMA often match longer ones. Screen the epoch count on one fold.
- Validate every epoch only when it is cheap. Subsample a large validation set for the per-epoch curve (pruning) and
  evaluate the best checkpoint in full at the end.
- Stop at the plateau: if fold 0's best epoch is consistently <= 70 % of the schedule, shorten the schedule for the
  other folds.
- Exploration at lower resolution, shorter `max_len` or a data subset; scale up for the finals.

## Stalls in the step

When the real loop is slower than both the loader alone and the step alone:

- `.item()`, `.cpu()` or `print(loss)` every step forces a GPU sync. Accumulate on the GPU and read every N steps.
- Validation or checkpointing too often: each is a full pass or a disk write.
- Metrics computed on the CPU over the whole OOF array every step.
- Tiny batches, where Python overhead per step dominates. Raise the batch, or use `torch.compile`.
