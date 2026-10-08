---
name: kg-baseline
description: Build a fast, strong, correctly-validated baseline for the current competition from the plugin's templates, log it to the ledger and produce a validated submission file.
argument-hint: "[model: lgbm|xgb|catboost|image|transformer]"
allowed-tools: [Bash, PowerShell, Read, Write, Edit, Glob, Grep]
---

# /kg-baseline

Arguments: `$ARGUMENTS`

1. Read `.kaggle-gm/competition.json` and CLAUDE.md; require frozen folds (run `/kg-cv` first if
   `data/folds.csv` is missing).
2. Choose the template from `${CLAUDE_PLUGIN_ROOT}/templates/` by task: tabular → `train_gbdt.py`;
   images → `train_image.py`; text → `train_transformer.py`. Copy into `src/`, vendor kgkit with
   `python -m kgkit vendor src/` so `from kgkit import ...` works, and adapt the CONFIG block
   (paths, target, id, metric, features, model).
3. Smoke test (1 fold, tiny settings), then the full run. Keep the baseline simple and fast — its
   job is to be correct and to calibrate CV vs LB.
4. Ensure it logs to the ledger with OOF + test predictions and writes `subs/<exp_id>.csv`.
5. `python -m kgkit validate subs/<exp_id>.csv data/sample_submission.csv`.
6. Report CV ± std, per-fold scores, runtime, and recommend submitting it (via `/kg-submit`) to
   calibrate CV↔LB.
