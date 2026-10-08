---
name: ensembling
description: How Kaggle Grandmasters build ensembles — OOF management, diversity, averaging/rank/geometric blending, Caruana hill climbing, constrained weight optimisation, stacking with nested CV, multi-seed and multi-fold averaging, and avoiding blend overfitting. Use when combining models or when asked to squeeze the final points out of a leaderboard.
---

# Ensembling

## Prerequisites (non-negotiable)

- Every candidate model has **OOF predictions on the same frozen folds** and **test
  predictions** averaged over its fold models — saved via `Ledger().log(oof=..., test_pred=...)`.
- OOFs and test preds are on the same scale (probabilities vs logits vs ranks) and same row order.
- The blend is scored with the exact competition metric.

## 1. Diversity sources (ranked by usual value)

1. Different model families (GBDT vs NN vs linear/kNN; CNN vs ViT; DeBERTa vs LLM).
2. Different inputs/views (feature sets, resolutions, crops, text fields, max_len, tokenizers).
3. Different targets/losses (regression vs classification of same target; auxiliary targets).
4. Different training data (with/without external or pseudo-labels, different time windows).
5. Different seeds / fold splits — cheap, always helps a little; average them *within* a model
   before blending across models.

Check `kgkit.ensemble.oof_correlation`: two models at 0.99 correlation add little; a weaker model
at 0.85 correlation can add a lot.

## 2. Blending methods (try in this order)

| Method | Use |
|---|---|
| Simple mean of top-k diverse models | robust baseline; hard to overfit |
| Rank average | AUC/ranking metrics; models with different calibration |
| **Hill climbing (Caruana)** with replacement | default; `python -m kgkit blend --method hill` |
| Constrained weights (SLSQP, simplex) | smooth metrics (logloss/RMSE), few models |
| Geometric mean / power average | probabilities for logloss; power>1 for AUC-ish sharpening |
| Stacking (ridge/logistic/LightGBM meta on OOFs) | many models; potentially non-linear interactions; validate nested |

`kgkit.ensemble.cv_blend_score` gives the **honest** blended score (weights fit on k−1 folds,
scored on the held-out fold). If the honest score is much worse than the in-sample blend
score, you're overfitting the blend — use fewer models, simpler weights, or plain averages.

## 3. Stacking

- Level-1: OOFs of many models (+ optionally a few strong raw features / meta-features).
- Level-2: Ridge / logistic regression (robust), or shallow LightGBM / small MLP; use the
  **same folds** to produce level-2 OOF; never fit level-2 on test-time predictions.
- Multi-level stacking rarely pays except in very large Playground-style ensembles.

## 4. Post-blend steps

- Re-tune thresholds/rounding on the *blended* OOF (`kgkit.thresholds`).
- Calibration for logloss (temperature scaling / isotonic fit on OOF).
- Clip to valid ranges.

## 5. Practical rules

- Blend weights must be learned on OOF, not on public LB. One or two LB probes to sanity-check
  a blend is fine; LB-tuned weights are a classic shake-up victim.
- Prefer fewer, stronger, diverse models over a huge pile; prune models with ~0 weight.
- Keep the blend reproducible: the ledger entry records weights and member experiment ids.
- For code competitions, account for inference time of every member; drop the members with the
  worst (Δscore / runtime) ratio first.
- Retrain-on-full-data models have no OOF: assign them the weights learned for their CV twins.
