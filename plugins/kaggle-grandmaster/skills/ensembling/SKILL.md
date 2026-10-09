---
name: ensembling
description: How Kaggle Grandmasters build ensembles — OOF management, diversity, averaging/rank/geometric blending, Caruana hill climbing, constrained weight optimisation, stacking with nested CV, multi-level and residual stacking over large model libraries, library pruning, multi-seed and multi-fold averaging, and avoiding blend overfitting. Use when combining models or when asked to squeeze the final points out of a leaderboard.
---

# Ensembling

## Prerequisites (non-negotiable)

- Every candidate model has **OOF predictions on the same frozen folds** and **test
  predictions** averaged over its fold models — saved via `Ledger().log(oof=..., test_pred=...)`.
  *Why it matters (be precise about the mechanism):* each OOF value is individually honest
  whatever the split, but when splits differ, the level-1 models that produced the OOFs you use
  to **fit** blend weights / a stacker on folds ≠ k were themselves trained on fold k's rows —
  so fold k's labels leak into the level-2 fit, and nested blend scores and stacking CV become
  optimistic. The leak is small for a plain average or a few hill-climbing weights, and grows with
  meta-model flexibility (stacking, many weights, post-processing fit on the blend).
  Remedy: regenerate OOFs on the shared folds (cheap models first); if impossible, keep
  mismatched models to simple/low-degree-of-freedom blends and treat their blended CV as
  optimistic. The ledger records a `folds_hash` per experiment and `kgkit blend` warns when
  members' folds differ.
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
| Stacking (ridge/logistic/shallow GBDT meta on OOFs) | many models; potentially non-linear interactions; `--method stack` |
| Residual stacking | one strong model with systematic errors that other models or raw features can explain; `--method residual --base <id>` |
| Multi-level stack + final hill climb | Playground-scale libraries (50–800 OOFs); `--method stack --levels 2` after `--prune` |

`kgkit.ensemble.cv_blend_score` gives the **honest** blended score (weights fit on k−1 folds,
scored on the held-out fold). If the honest score is much worse than the in-sample blend
score, you're overfitting the blend — use fewer models, simpler weights, or plain averages.

## 3. Stacking

- Level-1: OOFs of many models (+ optionally a few strong raw features / meta-features, via
  `kgkit.ensemble.stack(..., X_extra=...)`, so the meta-model can learn *where* each model is right).
- Level-2: Ridge / logistic regression (robust), or shallow GBDT / small MLP; use the
  **same folds** to produce level-2 OOF; never fit level-2 on test-time predictions.
- **Large libraries.** Playground winners now stack 100+ models in 3–4 levels (2026 churn: 150 of
  850 experiments, 4 levels). Recipe:
  1. `kgkit blend --exp <all> --prune 40` drops near-duplicates (OOF corr > 0.995 with a better
     model). Duplicates only add weight-fitting noise.
  2. `--method stack --levels 1`: a layer of [linear, shallow GBDT] meta-models on the same folds, then
     a hill-climbed blend of that layer. Then try `--levels 2`.
  3. Accept the deeper stack only if its *honest* final score (printed: blend weights fit on k−1 folds)
     beats the shallower one by more than paired noise (`kgkit ledger compare`). Each level reuses the
     same folds, so it adds a little optimism. A gain that appears only in the in-sample score is that
     optimism.
- **Residual stacking.** Instead of re-weighting everything, a stage-2 model learns the errors of one
  strong base model from the other OOFs (+ raw features): `--method residual --base <id>`
  (`kgkit.ensemble.residual_stack`). It is useful when error analysis shows a slice where another model is
  right. For GBDT bases the equivalent is boosting from the base margin (`init_score`).
- **Distillation of the stack**: one model trained on the stack's OOF/test predictions as soft targets
  can match much of the stack at a fraction of the inference cost (`leaderboard-boosters`), which matters
  in code competitions.

## 4. Post-blend steps

- Re-tune thresholds/rounding on the *blended* OOF (`kgkit.thresholds`).
- Calibration for logloss (temperature scaling / isotonic fit on OOF).
- Clip to valid ranges.

## 5. Practical rules

- Blend weights must be learned on OOF, not on public LB. One or two LB probes to sanity-check
  a blend is fine; LB-tuned weights are a classic shake-up victim.
- With a handful of models, prefer fewer, stronger, diverse models over a pile and prune those with ~0
  weight. With hundreds of logged experiments, keep the library: prune duplicates and let a stack +
  honest scoring choose. Either way, save OOF + test predictions for *every* experiment, including
  failed ones. A model that loses alone can still add to the blend.
- Teammates' and public notebooks' predictions join through `kgkit ledger import` (CV recomputed on
  the shared folds); see `competition-strategy` for the merge protocol.
- Keep the blend reproducible: the ledger entry records weights and member experiment ids.
- For code competitions, account for inference time of every member; drop the members with the
  worst (Δscore / runtime) ratio first.
- Retrain-on-full-data models have no OOF: assign them the weights learned for their CV twins.
