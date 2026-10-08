# GBDT starting parameters and tuning order

## LightGBM

```python
params = dict(
    objective="binary",          # regression | regression_l1 | huber | tweedie | multiclass | lambdarank
    metric="auc",
    learning_rate=0.03,          # 0.05–0.1 while exploring, 0.01–0.02 for final models
    n_estimators=20000,          # with early_stopping(200)
    num_leaves=63,               # main capacity knob (15–255)
    max_depth=-1,
    min_child_samples=40,        # 20–200; raise on noisy data
    subsample=0.8, subsample_freq=1,
    colsample_bytree=0.6,        # 0.3–0.9; low values often win with many features
    reg_alpha=0.1, reg_lambda=1.0,
    min_split_gain=0.0,
    max_bin=255,                 # 63 for speed, 511–1023 for fine numeric splits
    verbose=-1, n_jobs=-1, random_state=seed,
)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
```

Variants worth trying for diversity: `boosting_type="dart"` (no early stopping; fix rounds, slow
but often +), `extra_trees=True`, `linear_tree=True`, `path_smooth`, GOSS (`data_sample_strategy="goss"`).

## XGBoost

```python
params = dict(
    objective="binary:logistic", eval_metric="auc",
    learning_rate=0.03, n_estimators=20000, early_stopping_rounds=200,
    max_depth=6,                 # 4–10; or grow_policy="lossguide" + max_leaves
    min_child_weight=5,
    subsample=0.8, colsample_bytree=0.6, colsample_bylevel=1.0,
    reg_alpha=0.1, reg_lambda=1.0, gamma=0.0,
    tree_method="hist", device="cuda", enable_categorical=True,
    max_bin=256, random_state=seed,
)
```

## CatBoost

```python
params = dict(
    loss_function="Logloss", eval_metric="AUC",
    learning_rate=0.05, iterations=20000, od_type="Iter", od_wait=300,
    depth=6,                     # 4–10 (symmetric trees)
    l2_leaf_reg=3,               # 1–30
    random_strength=1, bagging_temperature=0.5,   # or bootstrap_type="Bernoulli", subsample=0.8
    border_count=254,
    one_hot_max_size=4,
    task_type="GPU",             # big speedup on large data
    random_seed=seed, verbose=0,
)
model.fit(Pool(X_tr, y_tr, cat_features=cats), eval_set=Pool(X_va, y_va, cat_features=cats), use_best_model=True)
```
Text columns: `text_features=[...]` gives CatBoost built-in text processing.

## Tuning order (do features first!)

1. Fix learning_rate high (0.05–0.1) for speed.
2. Capacity: num_leaves / max_depth / depth, min_child_samples / min_child_weight.
3. Sampling: colsample_bytree, subsample.
4. Regularisation: reg_lambda, reg_alpha, min_split_gain / gamma, l2_leaf_reg.
5. Lower learning_rate to 0.01–0.02 for the final and increase rounds.
Use Optuna (see `hyperparameter-tuning`) with the frozen folds — or 1–2 folds for speed then
confirm on all. Expect small gains (often < features). Diverse *settings* (deep vs shallow,
low vs high colsample) are useful ensemble members.

## Objectives cheat sheet

| Metric | Objective |
|---|---|
| AUC / logloss | binary logloss (`is_unbalance`/`scale_pos_weight` hurts calibration — prefer none for logloss) |
| RMSE | L2; RMSLE → L2 on log1p(y) |
| MAE | L1 / Huber / quantile 0.5 |
| Counts / sales | Poisson, Tweedie (variance_power 1.1–1.5) |
| Ranking (NDCG/MAP) | lambdarank / rank:ndcg / YetiRank with group ids |
| QWK | L2 regression + OptimizedRounder |
| Multiclass logloss | multiclass softmax; consider per-class calibration |
| Quantile/pinball | quantile objective per alpha |
