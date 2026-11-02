---
type: llm
---

PASS when the response rejects the "pearson 1.00" as evidence:

1. The 1.00 is an artefact: the two blends were never cross-validated (they carry the stack's CV), so there are
   only about 2 distinct CV values over 3 points, and the number says nothing about whether CV tracks the LB.
2. Such blends should be recorded as LB-only (no CV of their own) and kept out of any CV-LB correlation.

FAIL if it treats the 1.00 correlation as evidence that CV and LB agree.
