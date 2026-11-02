---
type: llm
---

PASS when the response names the time leakage and fixes the validation:

- Shuffled KFold puts neighbouring dates in train and validation, and with 20-day forward-return targets the
  windows overlap, so training rows share most of their target with validation rows (and same-date rows sit on
  both sides). The CV is leaky, which explains 0.09 vs 0.01.
- The fix is time-ordered validation (walk-forward / forward-chaining splits) with purging of the overlapping
  20-day target window and an embargo gap, for example purged K-fold with embargo.

FAIL if it attributes the gap mainly to overfitting or hyperparameters without naming the overlapping-target /
temporal leakage, or keeps shuffled KFold.
