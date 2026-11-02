---
type: llm
---

PASS when the response says the CUDA code must be ported to torch_xla before it runs on the TPU and validated
before queueing:

1. The port: bf16 instead of fp16 + GradScaler, static shapes (no boolean indexing or data-dependent shapes),
   `xm.mark_step`, saving with `xm.save`, and loading the CUDA checkpoint onto the XLA device. Naming most of
   these is enough.
2. Validate first (a local CPU/PJRT XLA run and/or a short GPU dev run of the same code), because a bug found
   after the queue costs a whole queue cycle and a queued job can only be removed by deleting the kernel.

FAIL if it tells the user to queue the unported CUDA code as is, or never mentions validating before queueing.
