# Porting a CUDA PyTorch trainer to Kaggle TPU (torch_xla, PJRT)

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
