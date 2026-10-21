---
type: llm
weight: 1
---

The response explains that the 0.039 figure is the wrong yardstick for differences: it is the SE of one
absolute score (a binomial sqrt(p(1-p)/n) that only describes an accuracy-like metric, not a macro AUC), while
two models scored on the same 58 studies are strongly correlated, so the noise of their *difference* is much
smaller (the shared difficulty of the studies cancels). It recommends measuring that noise directly with a
paired comparison on per-study predictions of both models: a paired bootstrap that resamples studies and
scores both models with the exact metric on each resample (e.g. `kgkit ledger compare`, or `kgkit ledger add
... --pred` on stored holdout predictions), and judging the +0.0008 gain against that paired SE, so the holdout
remains useful as a regression guard next to the public LB. Noting that +0.0008 may still be within paired
noise, or that the bootstrap does not include seed (training) noise and a second seed would help, is a plus.
Fail if it accepts 0.077 as the threshold for differences, recommends dropping the holdout because of it, or
replaces it with another formula of only n and the score.
