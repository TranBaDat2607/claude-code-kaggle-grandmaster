---
type: llm
---

PASS when the response weighs the TPU against waiting for the GPU reset with the queue in mind:

1. The TPU quota is separate from the GPU quota and also expires at the reset, so it is usable now.
2. Batch TPU jobs queue for many hours (roughly half a day or more) before they start, and only one batch TPU
   session runs at a time, so the job should be queued soon and its finish time estimated as queue + run time,
   compared with simply waiting ~17 h for the GPU reset. A fallback GPU run after the reset is a sensible hedge.
3. It keeps each seed on its own chip (or otherwise keeps the per-process batch) and stays within the 9 h TPU
   session limit.

FAIL if it says the TPU uses the GPU quota, or ignores the TPU queue.
