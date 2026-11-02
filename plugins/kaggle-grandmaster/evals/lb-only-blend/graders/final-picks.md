---
type: llm
---

PASS when the final-selection advice keeps a CV-backed pick:

1. The blend weights are a bet on the public LB (a fraction of the test set), and a +0.00004 public gain is
   within LB noise.
2. Keep the honest nested-CV stack as one final pick; use a public-file blend, if any, only as the hedge on a
   different risk axis.

FAIL if it recommends that both final picks be the public-file blends, or picks a blend over the stack because
of its public score.
