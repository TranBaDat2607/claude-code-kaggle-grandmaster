---
type: llm
---

PASS when the response keeps the faster screens comparable with the earlier runs, with at least one of:

- keep the effective batch the same (micro-batch x accumulation x GPUs: e.g. halve the accumulation to 2 on two
  GPUs) and the learning rate and schedule;
- compare each one-copy screen against a one-copy control (the baseline run the same way), not against the
  average of two copies;
- name the trade-off for the user: one copy is noisier than an averaged pair.

FAIL if it switches to two GPUs and doubles the effective batch without saying so, or compares single-copy
screens with the two-copy average as if they were equal.
