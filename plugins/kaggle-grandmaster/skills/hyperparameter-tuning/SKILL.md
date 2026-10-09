---
name: hyperparameter-tuning
description: Efficient hyperparameter optimisation for Kaggle — when tuning is worth it, Optuna recipes with pruning on frozen folds, search spaces for LightGBM/XGBoost/CatBoost/NNs/transformers, budget allocation, and avoiding CV overfitting from excessive search. Use when the user wants to tune models or asks for an Optuna setup.
---

# Hyperparameter Tuning

**Priority:** validation > data/features > model family > loss/augmentation > hyperparameters.
Tune *after* the feature set stabilises, and spend tuning budget where gains are measurable
(beyond paired noise: `kgkit ledger compare` the tuned config against the accepted baseline).

## Optuna recipe (GBDT)

```python
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from kgkit.cv import split_indices
from kgkit import metrics as M

metric = M.get("auc")
folds = pd.read_csv("data/folds.csv")["fold"].to_numpy()

def objective(trial):
    params = dict(
        objective="binary", learning_rate=0.05, n_estimators=10000, verbose=-1,
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 0, 1),
    )
    scores = []
    for k, (f, tr, va) in enumerate(split_indices(folds)):
        m = lgb.LGBMClassifier(**params).fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
                                              callbacks=[lgb.early_stopping(200, verbose=False)])
        scores.append(metric(y[va], m.predict_proba(X.iloc[va])[:, 1]))
        trial.report(np.mean(scores), k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return np.mean(scores)

study = optuna.create_study(direction="maximize" if metric.greater_is_better else "minimize",
                            sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
                            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
                            storage="sqlite:///artifacts/optuna.db", study_name="lgbm_v1", load_if_exists=True)
study.optimize(objective, n_trials=100, timeout=3 * 3600)
```

- Use a persistent `storage` so studies survive restarts and can run in parallel workers.
- Tune with a higher learning rate, then lower it for the final fit.
- Speed: tune on 1–2 folds or a subsample, confirm the top-5 configs on all folds.
- Log the best trial(s) as ledger experiments; keep 2–3 *different* good configs for ensemble
  diversity rather than only the argmax.

## Search spaces

- **XGBoost**: max_depth 3–12, min_child_weight 1–100 (log), subsample 0.5–1, colsample_bytree
  0.2–1, reg_lambda 1e-3–30 (log), reg_alpha 1e-3–10 (log), gamma 0–5.
- **CatBoost**: depth 4–10, l2_leaf_reg 1–30 (log), random_strength 0–10, bagging_temperature 0–1
  (or subsample 0.5–1 with Bernoulli), border_count {128, 254}.
- **MLP (tabular)**: layers 2–4, width 128–1024, dropout 0–0.4, lr 1e-4–3e-3 (log), wd 1e-6–1e-2,
  batch {256…4096}, embedding dims.
- **Transformers**: lr {1e-5, 2e-5, 3e-5}, epochs 2–5, warmup 0–0.1, LLRD 0.8–1.0, max_len,
  pooling — grid over 4–8 combos beats random search here.
- **CNNs**: lr, weight decay, image size, augmentation strength, epochs — coarse grids; image size
  and backbone matter far more than fine lr tuning.

## Overfitting the CV

Hundreds of trials on a small dataset will find configs that fit the fold noise. Mitigations:
repeated CV with a different fold seed for confirmation, keep a small untouched holdout for the
final decision, prefer regions of good configs (robust) over single peaks.
