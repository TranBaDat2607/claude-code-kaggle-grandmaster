---
name: cv-lb-debugging
description: Systematic diagnosis when cross-validation and leaderboard disagree, when a score is suspiciously good or bad, or when an improvement in CV does not transfer to LB — leakage hunting, distribution shift analysis, inference/training mismatch, metric bugs, and public LB noise estimation. Use when CV and LB diverge or a result looks wrong.
---

# CV ↔ LB Debugging

## Symptom → likely causes

| Symptom | Likely causes (check in order) |
|---|---|
| CV ≫ LB (CV too optimistic) | group leakage (same entity in train/val), temporal leakage, target-derived features fit outside folds, duplicates across folds, preprocessing fit on all data, post-processing tuned on reported OOF |
| CV ≪ LB | validation harder than test (e.g. grouped CV but test shares groups with train → test has "seen" entities; could exploit), noisy small public LB, val includes noisy labels absent from test |
| CV improves, LB flat/down | small/noisy public split, change exploits a train-only artefact, distribution shift on that feature, inference pipeline differs |
| LB much worse than any CV fold | submission bug: id misalignment, sorting, wrong column, probabilities vs labels, wrong transform inverse (expm1), different preprocessing at inference |
| Huge sudden CV jump | leak — assume guilty until proven otherwise |
| High fold variance | small data / noisy metric; use more folds, repeated CV, seeds |

## Procedure

1. **Re-score the submitted file offline**: run the inference code on *training* data (or a
   validation fold) and check it reproduces OOF scores — catches inference/training mismatch.
2. **Sanity-check the submission**: `python -m kgkit validate`; compare prediction distribution
   (mean, quantiles) of test vs OOF; correlation of the new submission with the previous best.
3. **Leakage audit** (dispatch `validation-auditor` agent): list every feature's construction and
   when it is fit; check fold assignment against groups/time; search for duplicates.
4. **Shift analysis**: `python -m kgkit adv` — drifting features; compare per-feature importance
   with drift ranking; try dropping top drifting features and see whether CV↔LB agreement improves.
5. **Public LB noise estimate**: public set size n and metric — bootstrap OOF on n-sized samples
   to get the std of the metric at that size. If the LB difference is within ~2 std, it's noise.
6. **CV-LB correlation across subs**: `python -m kgkit status`. Low correlation with several diverse
   submissions → the validation doesn't mirror the test; rebuild it (time split? grouping?
   test-like holdout via adversarial p(test)).
7. Record the conclusion in the ledger notes and CLAUDE.md — this knowledge drives final picks.

## Bootstrap snippet (public LB noise)

```python
import numpy as np
from kgkit import metrics as M
rng = np.random.default_rng(0)
n_public = 5000  # rows in public split (overview/data page: e.g. 20% of test)
scores = [M.score("auc", y[idx], oof[idx]) for idx in (rng.choice(len(y), n_public) for _ in range(500))]
print(np.std(scores))  # LB differences below ~2x this are noise
```
