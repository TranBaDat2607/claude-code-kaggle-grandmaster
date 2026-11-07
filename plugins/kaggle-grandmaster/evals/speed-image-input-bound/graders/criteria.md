---
type: llm
---

PASS when the response:

1. Diagnoses the run as input-bound: the CPU decodes and resizes ~2500px PNGs every epoch on the few CPU cores of a
   Kaggle GPU session (4), so the GPU waits. torch.compile or a bigger GPU would not fix that first.
2. Recommends converting the images once to (or near) the training resolution and caching them (uint8 npy, PNG or
   JPEG), and making the per-epoch CPU work cheap (augmentation or normalisation on the GPU, lighter CPU transforms).
3. Uses the idle second T4 (a second fold in parallel, one per GPU, or DDP) and matches num_workers to the CPU cores
   shared by the processes (8 workers oversubscribe 4 cores).
4. Says to measure before and after (data loading alone vs the training step, or GPU utilisation) and to check that
   the score holds after the change.

FAIL if it leads with torch.compile, bf16 or a bigger GPU as the main fix, or recommends bf16 on the T4.
