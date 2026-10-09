---
name: kg-baseline
description: Build fast, strong, correctly-validated baselines for the current competition from the plugin's templates — the main model plus diverse families on the same folds — log them to the ledger, decide the accepted baseline and produce a validated submission file.
argument-hint: "[model: lgbm|xgb|cat|hgb|linear|knn|svm|mlp|image|transformer]"
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
4. **Tabular: diverse baselines on the same folds.** Also run `--model linear` and `--model mlp`
   (plus `knn` / `svm` on data below ~50k rows, or a row subsample), each a few minutes. Report
   which families fit (GBDT ≫ linear → interactions to engineer; linear ≈ GBDT → simple signal or a leak
   to check). Their OOFs are kept for the blend. Images/text: one fast backbone is enough on day 1;
   a second family comes in the exploration phase.
5. Ensure each run logs to the ledger with OOF + test predictions and writes `subs/<exp_id>.csv`.
6. Mark the main baseline as the reference: `python -m kgkit ledger decide <exp_id> baseline "<why>"`.
7. `python -m kgkit validate subs/<exp_id>.csv data/sample_submission.csv`.
8. Report CV ± std, per-fold scores, runtime per family, and recommend submitting the main baseline
   (via `/kg-submit`) to calibrate CV↔LB.
