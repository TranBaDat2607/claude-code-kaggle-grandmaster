---
type: llm
---

PASS when the response gives concrete ways to fit the CPU-only 90-minute limit: export to ONNX or OpenVINO (or
otherwise optimise CPU inference), a smaller or cheaper model/input setting, batching the mel computation and
the windows, using all CPU cores (multiprocessing), and measuring throughput on a hidden-test-sized input early
with a safety margin. Naming at least three of these is enough.

FAIL if it ignores the CPU time limit or only suggests a GPU.
