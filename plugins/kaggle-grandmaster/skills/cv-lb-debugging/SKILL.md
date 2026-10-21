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
5. **Public LB noise estimate**: ask the right question. Two submissions are scored on the *same*
   public rows, so the noise of their LB *gap* is the paired SE, often several times smaller than the
   SE of one score: `python -m kgkit ledger compare <a> <b> --public-n <rows>` (paired bootstrap of
   their OOF/holdout predictions at the public-set size). The SE of one *absolute* score at that size,
   `python -m kgkit ledger noise <id> --public-n <rows>`, is only the yardstick for a single
   submission's CV-LB gap. Both assume the public rows look like your evaluation rows.
6. **CV-LB correlation across subs**: `python -m kgkit status`. Low correlation with several diverse
   submissions → the validation doesn't mirror the test; rebuild it (time split? grouping?
   test-like holdout via adversarial p(test)).
7. Record the conclusion in the ledger notes and CLAUDE.md — this knowledge drives final picks.

## Snippet (public LB noise, outside the CLI)

```python
from kgkit.compare import describe, estimate
n_public = 5000  # rows in public split (overview/data page: e.g. 20% of test)
gap = estimate(y, oof_a, oof_b, "auc", size=n_public, groups=None)  # paired: LB gap between a and b
one = estimate(y, oof_a, None, "auc", size=n_public)                # absolute: one score's CV-LB gap
print(describe(gap))  # LB gaps between a and b below ~2x this SE are noise
print(describe(one))  # a CV-LB gap of a's within ~2x this SE is sampling noise, not a validation problem
```
