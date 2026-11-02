---
type: llm
---

The workspace ledger holds the facts: 0003_lgbm-f3 (CV 0.74244) vs the accepted baseline 0002_lgbm-f2 (CV
0.73887) is +0.0034, and the paired comparison kgkit stored when 0003 was logged shows it better on 5 of 5 folds
with a paired SE of about 0.0002 (t about 20), verdict KEEP. The fold std (0.018) mostly measures how hard each
fold is, which both models share, so it is the wrong yardstick for a difference.

PASS when the reply brings this paired evidence to the user (per-fold improvements, the paired SE / t, or the
KEEP verdict) and explains that comparing the gain with the fold std is the wrong test, then recommends keeping
0003 (or asks the user to confirm before recording a discard).

FAIL if it records or confirms the discard without raising the paired evidence, or agrees that the gain is noise.
