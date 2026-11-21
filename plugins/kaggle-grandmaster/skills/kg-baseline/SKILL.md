---
name: kg-baseline
description: "Builds validated baselines from the plugin's templates and sets the accepted baseline. Use when the workspace has frozen folds but no accepted baseline."
argument-hint: "[model: lgbm|xgb|cat|hgb|linear|knn|svm|mlp|image|transformer]"
allowed-tools:
  - Read
  - Glob
  - Grep
  - 'Bash(kgkit ledger *)'
  - 'Bash(kgkit validate *)'
  - 'Bash(kgkit status *)'
---

# /kg-baseline

Run kgkit with the Bash tool as `kgkit <command>`: the plugin puts it on PATH, also on Windows (Git Bash),
and this command pre-approves the kgkit calls it makes. Older text that says `python -m kgkit` means the same
command. Submitting, pushing or publishing still asks.

Arguments: `$ARGUMENTS`

1. Read `.kaggle-gm/competition.json` and CLAUDE.md; require frozen folds (run `/kg-cv` first if
   `data/folds.csv` is missing).
2. Choose the template from `${CLAUDE_PLUGIN_ROOT}/templates/` by task: tabular → `train_gbdt.py`;
   images → `train_image.py`; text → `train_transformer.py`. Copy into `src/`, vendor kgkit with
   `kgkit vendor third_party/` so `from kgkit import ...` works (the templates
   look in `third_party/`), and adapt the CONFIG block
   (paths, target, id, metric, features, model).
3. Smoke test (1 fold, tiny settings), then the full run. Keep the baseline simple and fast — its
   job is to be correct and to calibrate CV vs LB. With a local NVIDIA GPU, train XGBoost / CatBoost
   there: `python src/train_gbdt.py --model xgb --device cuda` (or `KG_DEVICE=cuda`). Without `--name`
   the run is named `<model>_baseline`. Keep the templates' compute defaults (skill `compute-efficiency`).
   Images and text: after the smoke test run `--bench` once. An INPUT-BOUND verdict (images decoded from full size
   every epoch) slows every later experiment, so pre-resize or cache the inputs before the full run.
4. **Tabular: diverse baselines on the same folds.** Also run `--model linear` and `--model mlp`
   (plus `knn` / `svm` on data below ~50k rows, or a row subsample), each a few minutes. Report
   which families fit (GBDT ≫ linear → interactions to engineer; linear ≈ GBDT → simple signal or a leak
   to check). Their OOFs are kept for the blend. Images/text: one fast backbone is enough on day 1;
   a second family comes in the exploration phase.
5. Ensure each run logs to the ledger with OOF + test predictions and writes `subs/<exp_id>.csv`.
6. Mark the main baseline as the reference: `kgkit ledger decide <exp_id> baseline "<why>"`.
7. `kgkit validate subs/<exp_id>.csv data/sample_submission.csv`.
8. Report CV ± std, per-fold scores, runtime per family, and recommend submitting the main baseline
   (via `/kg-submit`) to calibrate CV↔LB.
