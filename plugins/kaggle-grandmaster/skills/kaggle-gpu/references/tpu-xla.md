# TPU v5e-8: when to use it, the queue, and the torch_xla port

## When the GPU quota is gone

The TPU quota (20 h/week) is separate and expires at the same reset, so it is free hours once the GPU
quota is spent. Measured on a bandwidth-bound ConvNeXt-small (Oct 2026): one v5e chip ≈ **5 T4s** per
step, bf16 losses matching the T4 run to 3-4 decimals. The catch is the queue:

- Batch TPU jobs wait **10-24 h** (12 h 49 min measured) before they start, plus ~20 min of start-up. One
  batch TPU session per account; a queued job holds that slot for its whole wait.
- Expected finish = now + ~13 h queue + 20 min + (T4 time / 5). Use the TPU when that beats the GPU
  finish: for training longer than a GPU session, or when the GPU quota is gone. For runs under ~12 h
  with GPU hours left, the GPU usually wins on wall time. Queue about half a day before you need it,
  and keep a GPU fallback scheduled for after the reset.
- `machine_shape`: ask for `TpuV5E8`. `TpuV5E1`/`TpuV5E4`/`TpuV6E1` are silently stored as `TpuV5E8`;
  `TpuV6E8` (often no capacity) and `Tpu1VmV38` (v3-8) are separate pools, queued too. `gpu build` targets
  v5e-8 and v6e-8.
- A queued job cannot be cancelled through the API (it has no session id yet). The only way to free the
  slot is `kaggle kernels delete` (irreversible; ask the user), then a re-push, which starts the queue
  again from zero. So **validate before queueing**: run the XLA path locally on PJRT CPU, then the same
  code on a ~5 min 2×T4 dev run. A bug found after the queue costs a whole queue cycle.
- `kgkit gpu build --accelerator TpuV5E8 --hours <= 9` runs one process per chip for `{fold}` tasks (PJRT
  `TPU_VISIBLE_CHIPS=<k>` with single-chip process bounds), so each fold/seed keeps the per-process batch
  it had on one GPU. A command without `{fold}` gets the whole TPU (`PJRT_DEVICE=TPU`). Hours are
  accounted as TPU-hours, not GPU-hours.
- Porting a CUDA trainer to torch_xla (bf16, static shapes, `mark_step`, `xm.save`, timm checkpointing,
  resuming CUDA checkpoints): the port below.


## The port

Measured on a 2.5-D ConvNeXt-small MIL trainer (4,348 MRI studies, batch 2, Oct 2026), resumed from a
CUDA checkpoint: ≈ 0.43 s per optimiser step on one v5e chip vs ≈ 2.1 s on a T4; losses matched the T4
run to 3-4 decimals. Keep the port as patches behind a flag (`MYRUN_XLA=1`) so the CUDA path stays the
reference.

## Source changes

| CUDA trainer | TPU version | why |
|---|---|---|
| `torch.device("cuda")` | `xm.xla_device()` | |
| fp16 autocast + `GradScaler` | `torch.autocast("xla", dtype=torch.bfloat16)`, scaler `enabled=False` | TPUs are bf16-native |
| boolean indexing (`x[mask]`, `t[mask] = f`) | compute on every padded element and mask the result; pad variable lengths to a fixed size in the collate | dynamic shapes recompile the graph every step |
| random crops on the device with Python-int offsets | crop on the CPU before `.to(dev)` | every offset pair is a new XLA graph |
| `loss[ok]` (boolean select) | `(loss * ok).sum() / ok.sum()` | boolean indexing forces a host sync |
| `total += float(loss)` every step | accumulate on the device, `float()` once per epoch | one host sync per step |
| — | `xm.mark_step()` after `opt.step()` | cuts the lazy graph |
| `torch.save(ckpt)` | `xm.save(ckpt)` | moves XLA tensors to CPU; the file loads on CUDA again |
| `torch.cuda.set/get_rng_state_all` | guard with `torch.cuda.is_available()` | resume across devices |
| timm `checkpoint_seq` (`torch.utils.checkpoint(..., use_reentrant=False)`) | redirect to `torch_xla.utils.checkpoint.checkpoint(..., use_reentrant=True)` | the non-reentrant path fails on XLA (`module 'torch' has no attribute 'xla'`) |
| `.to(memory_format=torch.channels_last)` | remove | |

Check numerical equivalence of the masked rewrite on CPU first (e.g. max |Δlogit| ~1e-7). It holds when
padded elements are excluded downstream (−inf attention, masked loss) and the model has no BatchNorm
over the padded batch.

## One process per chip

To keep a recipe's per-process batch (and so its schedule), run one seed or fold per chip instead of
data-parallel over 8 chips. Per process:

```
PJRT_DEVICE=TPU TPU_VISIBLE_CHIPS=<k> TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1
TPU_PROCESS_PORT=847<k> TPU_PROCESS_ADDRESSES=localhost:847<k>
```

`kgkit gpu build --accelerator TpuV5E8` sets exactly this for each `{fold}` task.

## Validate off-quota before queueing

1. Local CPU XLA: `pip install torch==<v> torch_xla==<v>` (CPU wheels, Linux/WSL) in a fresh venv, run the
   patched trainer with `PJRT_DEVICE=CPU` on a tiny cache: train → `xm.save` → resume → final model. Pin
   `torch` again after `pip install timm`, which can pull a newer torch and break `_XLAC` (`undefined
   symbol ...`).
2. Load the real CUDA checkpoint on `xla:0` (model, optimizer state and scheduler step).
3. Run the same code on a ~5 min 2×T4 dev session. The TPU queue is ~12 h; a bug found after it costs
   a whole cycle, and fixing a queued job means deleting the kernel.

## Kernel setup

- `kernel-metadata.json`: `"enable_tpu": true, "enable_gpu": false, "machine_shape": "TpuV5E8"`, push with
  `-t` ≤ 32400 (9 h).
- Every `kernel_sources` run must have finished before the push: inputs are bound at push time, and a
  TPU job waits for hours.
- `kaggle kernels logs` is empty while a TPU session runs; log progress to a file in `/kaggle/working`
  and read it after the run.
