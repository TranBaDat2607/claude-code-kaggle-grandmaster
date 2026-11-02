---
type: llm
---

PASS when all three hold:

1. It says the 0.039 / 0.077 figure is the wrong yardstick for differences: it describes the noise of one
   absolute score, while two models scored on the same 58 studies are strongly correlated, so the noise of their
   *difference* is much smaller (the shared difficulty of the studies cancels).
2. It recommends measuring that noise directly with a paired comparison on the per-study predictions of both
   models: a paired bootstrap that resamples studies and scores both models with the exact metric on each
   resample (for example `kgkit ledger compare`, or `kgkit ledger add ... --pred` on stored holdout predictions),
   and judging the +0.0008 gain against that paired SE.
3. It keeps the holdout as a decision tool (for example a regression guard next to the public LB) rather than
   dropping it for the public LB alone.

A plus, not required: noting that the 0.039 comes from a binomial sqrt(p(1-p)/n) formula that describes an
accuracy-like metric, not a macro AUC; that +0.0008 may still be within paired noise; that the bootstrap leaves
out seed (training) noise, so a second seed helps.

FAIL if it accepts 0.077 as the threshold for differences, recommends dropping the holdout because of it, or
replaces it with another formula of only n and the score.
