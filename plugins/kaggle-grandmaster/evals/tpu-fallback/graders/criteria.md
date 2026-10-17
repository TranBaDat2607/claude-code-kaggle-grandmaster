---
type: llm
weight: 1
---

The response weighs the TPU as a separate, unused quota against waiting for the GPU reset, with the queue
in mind. Required elements: (1) the TPU quota is separate and also expires at the reset, so it is usable
now; (2) batch TPU jobs queue for many hours (roughly half a day or more) before they start, and only one
batch TPU session runs at a time, so the job should be queued soon and the finish time estimated as
queue + run time, compared with simply waiting ~17 h for the GPU reset (a fallback GPU run after the reset
is a sensible hedge); (3) the code must be ported to torch_xla: bf16 instead of fp16 + GradScaler, static
shapes (no boolean indexing / data-dependent shapes), `xm.mark_step`, saving with `xm.save`, and loading
the CUDA checkpoint on the XLA device; (4) validate before queueing (a local CPU/PJRT XLA run and/or a short
GPU dev run of the same code), because a bug found after the queue costs a whole queue cycle and a queued
job can only be removed by deleting the kernel; (5) keep each seed on its own chip (or otherwise keep the
per-process batch) and stay within the 9 h TPU session limit. Fail if it says the TPU uses the GPU quota,
ignores the queue, or tells the user to queue the unported CUDA code as is.
