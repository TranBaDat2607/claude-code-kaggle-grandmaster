---
type: llm
---

PASS when the response:

1. Avoids generation for a multiple-choice answer: one forward pass that reads the logits of the answer tokens (A-E),
   or a classification head.
2. Batches the prompts (sorted by length, dynamic padding), or uses a fast engine such as vLLM, with fp16 on the T4
   (`dtype="half"`: the T4 has no bf16).
3. Fits the 14B model in 2x16 GB: quantisation (8-bit GPTQ, W8A8 or 4-bit AWQ/GPTQ) checked on a validation set so
   accuracy holds, and/or the model split across both T4s or one process per GPU.
4. Measures on a sample on Kaggle hardware and extrapolates to the full test with a safety margin.

FAIL if it suggests bf16 on the T4, or anything that needs internet access during the run.
