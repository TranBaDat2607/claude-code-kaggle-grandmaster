---
name: metric-optimization
description: Optimising directly for the Kaggle evaluation metric — choosing training losses that align with AUC, logloss, RMSE/RMSLE/MAE, F1/F-beta, QWK, MAP@K, IoU/Dice, mAP, pinball and custom metrics; threshold and rounding optimisation, calibration, metric-aware post-processing and exact metric re-implementation. Use whenever deciding a loss, post-processing, or when a metric seems hard to optimise.
---

# Metric Optimisation

Step 1 is always: implement the exact metric (read the host's code if published), test it,
and know whether it is global, per-group-averaged, or weighted. `python -m kgkit metrics`
lists the built-in implementations (`kgkit.metrics.get(name)`).

## Metric → training & post-processing

| Metric | Train with | Post-process |
|---|---|---|
| ROC AUC / Gini | logloss (BCE) or ranking losses; only ordering matters | rank-average blends; no calibration needed |
| Logloss / balanced logloss | BCE/CE; class weights for balanced variants | temperature scaling / clipping (e.g. [1e-4, 1-1e-4]); averaging probabilities, not ranks |
| Accuracy | CE | argmax; prior shift correction if test priors differ |
| F1 / F-beta (binary) | BCE (maybe pos_weight) | tune threshold on OOF (`best_threshold`); F-beta>1 → lower threshold |
| Macro F1 (multiclass) | CE | per-class probability scaling/thresholds tuned on OOF |
| Multilabel F1 | BCE | per-class thresholds (`per_class_thresholds`); top-k fallback (never predict empty if the metric punishes it) |
| QWK | MSE regression on ordinal target (or CE + expected value) | `OptimizedRounder` thresholds on OOF |
| MCC | BCE | threshold tune |
| RMSE | MSE | clip; bias correction |
| RMSLE / MSLE | MSE on log1p(y) | expm1 inverse, clip ≥ 0 |
| MAE | L1 / Huber / quantile 0.5 | median-style blending |
| MAPE / SMAPE | weighted L1 (1/|y|), or log-target | SMAPE: zero-prediction handling per item; shrink small predictions |
| Pinball / quantile | quantile loss per quantile | enforce monotone quantiles (sort) |
| Pearson / Spearman | MSE or correlation loss | per-group normalisation if metric is per-group |
| MAP@K / NDCG@K | lambdarank / classification on candidates | ensure K unique predictions; fill with popular items |
| IoU / Dice (segmentation) | BCE + Dice / Lovasz | threshold, min-area filtering, empty-mask classifier gate |
| mAP (detection) | detector losses | conf threshold low (mAP likes many boxes), WBF, NMS IoU tune |
| F2 on detections / tolerance-based AP (events) | per-step BCE | peak detection, tolerance-aware NMS |
| Custom / weighted | sample weights mirroring metric weights | simulate the metric on OOF |

## Principles

- **Fit post-processing on OOF**, and estimate its honest gain with nested CV (fit on k−1
  folds' OOF, evaluate on the remaining fold) — thresholds overfit small validation sets.
- When the metric is non-differentiable, train on a smooth surrogate then optimise decisions.
- If the metric averages per group (per image/user/series), compute CV exactly that way and
  consider per-group post-processing (e.g. per-group normalisation, per-image thresholds).
- Check metric edge cases: empty predictions, all-negative groups, ties, duplicates, NaN rules,
  maximum number of predictions, string formatting (RLE encoding orientation, space-separated
  lists).
- Expected-value decoding: for metrics like QWK or MAE on ordinal labels, predict
  E[y] = Σ p_k · k from a classifier, then round optimally.
- Logloss competitions: overconfident wrong predictions are fatal; blend/average and clip.
- For "balanced" or class-weighted metrics, rebalance probabilities with the prior ratio
  (`class_prior_shift`) instead of retraining.
