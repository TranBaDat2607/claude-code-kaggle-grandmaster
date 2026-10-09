---
name: tabular-mastery
description: Winning playbook for tabular Kaggle competitions (including Playground Series) — GBDT (LightGBM, XGBoost, CatBoost) recipes, feature engineering patterns, categorical handling, NN models for tabular (MLP, TabM, transformers), GPU acceleration, synthetic-data competitions, and tabular ensembling. Use for any rows-and-columns problem.
---

# Tabular Mastery

Gradient-boosted trees + great features + diverse ensemble is still the winning recipe.
Neural nets add diversity at the end; on some problems (many numeric features, synthetic
Playground data) a well-tuned MLP/TabM is competitive and strongly boosts the blend.

## 1. Baseline in the first hour

- LightGBM with sensible defaults on raw features (categoricals as `category` dtype),
  frozen folds, early stopping, logged to the ledger, one submission.
  Template: `templates/train_gbdt.py` (handles LGBM/XGB/CatBoost, OOF, test preds, ledger).
- Then CatBoost and XGBoost on the same features — they diverge in categorical handling and
  tree growth, giving free ensemble diversity.
- **Diverse baselines on day 1, not at the end**: the same template runs `--model linear`, `knn`,
  `svm` and `mlp` (scaled numerics, one-hot categoricals) on the same folds. This shows which families
  fit the data. A linear model close to the GBDT suggests a simple signal or a leak, and a large gap
  suggests interactions worth engineering. Their OOFs are also blend material: a GBDT + NN + SVR ensemble
  with no feature engineering has placed 2nd in a Playground. Re-run them after big data changes as a
  leak check.
- Strong default-settings references for the blend: TabPFN (v2.5+ handles up to ~50k rows × 2k
  features and reports beating tuned GBDTs at that scale; check its licence and the competition rules)
  and AutoGluon (its OOF predictions can be stacking inputs). Validate both on the frozen folds like
  everything else.
- Read `references/gbdt-params.md` for starting parameters and tuning order.

## 2. Feature engineering — where tabular competitions are won

Work through these families; validate each on CV; keep a feature-group switch per family so
ablations are one flag.

| Family | Ideas | Notes |
|---|---|---|
| Domain / ratio | ratios, differences, products of meaningful columns; per-unit rates; physical formulas | highest yield; read the data description deeply |
| Group aggregates | mean/std/min/max/count of numeric by categorical (and combos); value − group mean; rank within group | `kgkit.features.group_aggregates`; use train+test pooled |
| Frequency | count encoding of categoricals & of rounded numerics | `count_encode`; reveals duplicates/synthetic structure |
| Target encoding | OOF smoothed mean (and std/quantiles) per category & category pairs | `oof_target_encode` — always OOF with the frozen folds |
| Interactions | categorical crosses; numeric × categorical aggregates | GBDTs find many but not all |
| Binning / rounding | round numerics, digit extraction (e.g. decimals reveal generation process), quantile bins | Playground synthetic data often benefits |
| Missingness | is_null flags, null counts per row | missingness patterns are often predictive |
| Time | date parts, cyclical, elapsed, time since last event per entity, rolling stats | see `time-series` for leakage rules |
| Text in tables | TF-IDF + SVD, lengths, embeddings from a small sentence model | adds a lot when present |
| Row-level stats | sum/mean/std/skew across similar columns; number of zeros | especially for anonymised features |
| Original dataset | Playground data is synthesised from a real dataset — append the original (flag it) | check rules; usually allowed and helpful |

**Feature search at scale.** Hand-designed families find the obvious features; brute force finds
the rest. Recent Playground winners generated 10,000+ groupby features (COL1 × COL2 × STAT) and kept
the best few hundred; pairwise categorical combinations (8 columns → 28 crosses) are another reliable
source. `kgkit features search` does this on the frozen folds:

```bash
python -m kgkit features search data/train.csv --target <y> --test data/test.csv \
    --folds data/folds.csv:fold --kinds cnt,te,grp,num2 --aggs mean,std,nunique,diff_mean,rank \
    --batch 40 --max-candidates 5000 --screen-folds 0,1 --recheck-seed 7 --minutes 60
```

It generates candidates (count / OOF target encodings of columns and pairs, groupby aggregates of
numerics by categoricals and pairs, pairwise arithmetic), screens them in batches with a fast fixed
GBDT (LightGBM if installed, else sklearn HGB), keeps batches that improve the paired fold score,
prunes each kept batch to its most important members, and re-checks the final set on an
independently seeded split. Hundreds of accept/reject decisions overfit the screening folds, so trust
the recheck line, not the screen CV. Then load the kept specs in the training script
(`kgkit.featsearch.materialize(load_specs(...), train, test, y, folds)`) and decide with the real model
and `kgkit ledger compare`. On large data, run it on a GPU (RAPIDS cuDF / GPU GBDTs) or on a row
subsample, and widen `--screen-folds` for the final confirmation.

**Feature selection:** permutation importance on OOF or null importance (shuffle target,
compare importance distributions); drop features that hurt CV; adversarial-drifting features
that do not help CV. Forward selection by feature *groups* is cheaper than per feature.

## 3. Categorical handling

- LightGBM native categorical (`categorical_feature`) for low/medium cardinality; tune
  `cat_smooth`, `min_data_per_group`. High cardinality: target/count encoding instead.
- CatBoost's ordered target statistics are excellent out of the box — pass `cat_features`
  and also give it categorical *combinations*.
- XGBoost: `enable_categorical=True` with `tree_method="hist"`.
- NNs: embeddings (dim ≈ min(50, (card+1)//2)), or one-hot for tiny cardinality.

## 4. NN models for tabular (diversity for the blend)

- MLP with: quantile/Gaussian-rank transform of numerics, categorical embeddings, batchnorm
  or layernorm, SiLU, dropout 0.1–0.3, AdamW, OneCycle/cosine, 20–100 epochs, early stopping.
- Strong modern choices: TabM (parameter-efficient ensembling MLP, strongest when tuned), RealMLP
  (strong at library defaults), FT-Transformer, piecewise-linear / periodic numeric embeddings.
  Tabular foundation models: TabPFN v2.5+ covers up to ~50k rows × 2k features in one forward pass
  (earlier versions: ~10k rows), and recent TabArena snapshots rank it near the top. Subsample or
  use its large-data variants above that. All of these are diversity for the blend.
- Seed-average 3–5 runs; NN OOFs typically correlate ~0.9 with GBDT → valuable in blends.

## 5. Training discipline

- Same folds for everything; save OOF + test preds per model (`Ledger().log(oof=..., test_pred=...)`).
- Average test predictions across fold models; optionally retrain on full data with
  1.1–1.2× mean best iterations for the final.
- Seed averaging for final models: 3–5 seeds for NNs; GBDTs are cheap enough for many more (a
  100-seed XGBoost bag beat its average single seed by ~0.003 MAP@3 in a Playground). Bagging
  `subsample`/`colsample` helps stability. Average seeds *within* a model before blending.
- Full-data refit after the configuration is frozen (see `leaderboard-boosters`).
- GPU: XGBoost `device="cuda"`, CatBoost `task_type="GPU"`, LightGBM GPU builds; RAPIDS cuDF/cuML
  for fast feature engineering & kNN/SVR models on big data.
- Large data: `reduce_mem_usage`, parquet, polars for feature engineering, float32.

## 6. Targets & losses

- Skewed positive targets: train on `log1p(y)` (and invert), or Tweedie/Poisson/Gamma objectives.
- RMSLE metric → RMSE on log1p. MAE → L1/Huber objective or median-targeting. MAPE →
  weight by 1/|y|. Classification metrics → see `metric-optimization`.
- Multiclass with ordinal structure → regression + `OptimizedRounder` often beats softmax for QWK.
- Clip predictions to the observed target range.

## 7. Playground Series specifics

Synthetic data generated from an original dataset: large (100k–1M rows), small signal-to-noise
gains, LB shake-ups of ±0.001 are common. Winning pattern: many diverse models (GBDTs with
different FE, NNs, linear/kNN/SVR models on transformed features), *many* OOFs, hill climbing
or ridge/logistic stacking over 20–100 OOFs, appending the original dataset, and selecting finals
on CV. Keep everything on 5–10 identical folds.

## 8. Ensembling for tabular

LightGBM + CatBoost + XGBoost + NN + (optional) linear/kNN/SVR on different feature views →
`python -m kgkit blend --exp ... --method hill --folds data/folds.csv:fold`. Level-2 stacking
with ridge/logistic on OOFs (+ a few raw features) can add more; validate with nested CV.

Playground-scale ensembles are much bigger: the March 2026 churn winner was a 4-level stack of 150
models selected from 850 experiments; the April 2025 winner stacked Lasso, SVR, kNN, RF, MLP, TabPFN,
XGBoost and LightGBM under XGBoost + MLP meta-models and a final weighted average. That only pays when
every experiment saved OOF + test predictions on the same folds (the ledger does this). Tooling:
`kgkit blend --prune 40` (drop near-duplicate OOFs), `--method stack --levels 2` (layers of
linear + shallow-GBDT meta-models, then hill climbing; compare the honest score with `--levels 1`), and
`--method residual --base <id>` (stage 2 learns one strong model's errors). See `ensembling`.
Distillation also scales here: train a single model on the ensemble's OOF/test predictions as soft
targets (`leaderboard-boosters`).
