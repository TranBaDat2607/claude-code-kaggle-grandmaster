---
type: llm
weight: 1
---

The response rejects the correlation and keeps a CV-backed final. Required elements: (1) the pearson 1.00
is an artefact: the blends were never cross-validated (they carry the stack's CV, so there are only ~2
distinct CV values over 3 points), so it says nothing about whether CV tracks the LB; (2) such blends
should be recorded as LB-only (no CV of their own) and left out of any CV-LB correlation; (3) the blend
weights are a bet on the public LB (a fraction of the test set), and a +0.00004 public gain is within LB
noise; (4) final selection: keep the honest CV stack as one pick and use a public-file blend, if any,
only as the hedge on a different risk axis, never both picks as LB-tuned blends. Fail if it treats the
1.00 correlation as evidence, or recommends both final picks be the public-file blends.
