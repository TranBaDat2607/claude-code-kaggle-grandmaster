---
name: validation-strategy
description: Designing a cross-validation scheme that tracks the private leaderboard — choosing KFold/Stratified/Group/StratifiedGroup/multilabel/time-series/purged splits, adversarial validation, CV-LB correlation, noise estimation, and leakage prevention. Use when setting up CV, when CV and LB disagree, or before trusting any score.
---

# Validation Strategy

**Rule:** the validation split must be generated the same way the test split was generated.
Find out how train/test were separated (random rows? by patient/user/site? by time? by
geography?) and imitate it.

## 1. Decision table

| How test differs from train | Scheme | kgkit |
|---|---|---|
| Random rows, iid | StratifiedKFold (classification) / KFold or binned-stratified (regression) | `assign_folds(strategy="stratified")` |
| Unseen groups (patients, users, images of same object, sessions, sites) | GroupKFold / StratifiedGroupKFold | `strategy="stratified_group", group=...` |
| Multilabel targets | iterative stratification | `target=[cols...]` |
| Future period | forward-chaining time split with gap ≥ horizon | `time_series_splits(t, gap=h)` |
| Overlapping-window targets (finance) | purged K-fold with embargo | `purged_kfold_splits(t, embargo=k)` |
| Unseen groups *and* future | time split, with groups in validation unseen in train | combine both |
| Strong covariate shift | adversarial validation → test-like holdout or weights | `kgkit.adversarial` |

Hidden group structure is the #1 source of CV/LB gaps: near-duplicate images, same customer
across rows, same text template, augmentations of one source, multiple rows per
patient-visit. Look for them (hash images, cluster texts, group by obvious IDs) and group.

## 2. Mechanics

- `python -m kgkit folds data/train.csv --target y --group g --id-col id --out data/folds.csv`
  then **commit folds.csv** (or the seed + script). Every model uses it.
- 5 folds is the default; use 10 for small data (<5k rows) or very noisy metrics, 3 for
  huge data or slow DL in exploration (but keep the final folds consistent across models).
- Report mean ± std across folds; look at per-fold scores for outliers.
- Repeated CV (several fold seeds) reduces variance for small data — but blending requires
  OOFs per seed to be combined consistently.
- Do the *whole* pipeline inside the fold: imputation, scaling, target encoding, feature
  selection, oversampling (SMOTE), PCA, pseudo-label selection, threshold tuning.
- Early stopping on the validation fold slightly inflates CV. For the final model either
  (a) use the per-fold models (average test preds over folds — the standard) or
  (b) retrain on full data with ~1.1–1.2× the mean best iteration.

## 3. Trust calibration: CV ↔ LB

1. Submit 3–5 *different kinds* of models early; record LB with `kgkit ledger lb`.
2. `kgkit status` shows CV-LB Pearson/Spearman. Good: strong positive monotone relation.
3. If they disagree, diagnose (see `cv-lb-debugging`): leakage in CV, distribution shift,
   a tiny/noisy public split, different preprocessing at inference, or a metric bug.
4. Public LB noise: with n public samples, the binomial std of accuracy ≈ sqrt(p(1-p)/n).
   E.g. accuracy 0.9 on 2,000 rows → ±0.0067; differences below that are noise.

## 4. Adversarial validation

`python -m kgkit adv data/train.csv data/test.csv --drop id,target`
- AUC ≈ 0.5: iid; random folds are fine.
- 0.6–0.8: find the drifting features; test whether dropping/transforming them hurts CV;
  consider normalising per group/time.
- >0.8: build validation from the most test-like training rows
  (`testlike_holdout_mask`) or weight training rows (`importance_weights`), and expect
  shake-up. AUC ≈ 1 usually means an ID/time column — drop and re-run.

## 5. Leakage checklist (run before trusting a jump in CV)

- Any feature computed using the target outside the fold? (target/mean encoding, aggregations
  that include the row's own target, "future" lags)
- Duplicates / near-duplicates spread across folds?
- Preprocessing fit on train+valid (scalers, PCA, TF-IDF vocab is OK only if unsupervised and
  also available at test time)?
- Threshold or post-processing tuned on the same predictions you report?
- Features unavailable at prediction time in the real test (e.g. post-event info)?
- A CV jump > 2× what any similar change gave before is guilty until proven innocent —
  dispatch the `validation-auditor` agent.

## 6. Metric hygiene

Implement the exact competition metric in `src/metric.py`, unit-test it on a tiny example
with a hand-computed value (or the host's code), and use it for every CV number. Check
whether the metric is computed globally or per group then averaged (e.g. per-image IoU,
per-user MAP) and replicate that.
