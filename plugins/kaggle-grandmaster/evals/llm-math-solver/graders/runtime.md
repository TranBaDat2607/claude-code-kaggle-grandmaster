---
type: llm
---

PASS when the response makes the plan fit 5 hours on 2x T4:

- An efficient inference engine (vLLM or similar) using both GPUs (tensor parallel or one process per GPU),
  fp16 or quantised weights (AWQ/GPTQ) since the T4 has no bf16, batched generation.
- Explicit time management: a per-problem time or token budget, hard timeouts, stopping early when the sampled
  answers already agree, and a fallback answer so every problem gets one.

FAIL if it ignores the 5-hour / 2x T4 budget or gives no per-problem time control.
