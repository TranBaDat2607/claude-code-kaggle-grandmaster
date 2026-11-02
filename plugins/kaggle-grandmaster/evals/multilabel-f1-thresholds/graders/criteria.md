---
type: llm
---

PASS when the response covers all three:

1. Tune the threshold(s) on out-of-fold predictions (global and/or per label) for micro-F1 instead of a fixed
   0.5, and apply the OOF-fitted thresholds to test.
2. Never output an empty row, since every row has at least one label: fall back to the top-1 (or top-k) label.
3. Estimate the gain honestly: thresholds fitted and scored on the same OOF overstate it, so fit on k-1 folds'
   OOF and score the held-out fold (nested), or similar.

FAIL if it keeps a fixed 0.5 threshold, ignores the empty rows, or reports the score of thresholds fitted on the
same OOF as the expected gain without any honesty check.
