---
type: llm
---

The workspace ledger holds the facts: 0003_lgbm-f3 (CV 0.74244) vs the accepted baseline 0002_lgbm-f2 (CV 0.73887)
is +0.0034, better on 5 of 5 folds with a paired SE of about 0.0002 (t about 20): the comparison kgkit stored when
0003 was logged says KEEP. Backlog idea #4 ("Add feature f3") is the idea 0003 tests.

PASS when the reply says 0003 was recorded as keep (it is now the accepted baseline) on the paired evidence (folds
improved, the paired SE / t, or the KEEP verdict), and that backlog idea #4 was closed with the result.

FAIL if it records discard or inconclusive, leaves the decision or the backlog idea unrecorded without saying why
(for example, kgkit could not run), or judges the gain against the fold std.
