# Kaggle competition: {{SLUG}}

Workspace managed by the **kaggle-grandmaster** plugin. Read this before every session.

## Facts (keep updated — `.kaggle-gm/competition.json` is the source of truth)

- Metric: **{{METRIC}}** ({{DIRECTION}} is better). The exact implementation lives in `src/metric.py` and is unit-tested against a known value.
- Task: {{TASK}} — target: `{{TARGET}}`
- Code competition: {{CODE_COMP}}
- Final deadline: {{DEADLINE}} (always confirm on the competition page; deadlines are 23:59 UTC)

## Layout

```
data/        raw competition files (never edited, never committed)
src/         training / inference code (committed)
notebooks/   exploration only — anything that matters gets moved into src/
artifacts/   per-experiment OOF + test predictions + models (artifacts/<exp_id>/)
subs/        submission CSVs, named after the experiment id
reports/     EDA, recon, adversarial validation, post-mortems
kernels/     Kaggle notebooks / datasets for code-competition inference
.kaggle-gm/  competition.json (facts), ledger.jsonl (every experiment, committed)
```

## Non-negotiable rules

1. **The CV scheme is frozen** in `data/folds.csv` (or a documented time split). Every model uses those exact folds so OOFs are blendable. Change it only with a written reason in the ledger notes, and then re-run the models that matter.
2. **Every run is logged** with `kgkit.experiment.Ledger().log(...)` — CV, fold scores, params, features, OOF and test predictions. No unlogged results, no "I think it was 0.812".
3. **One change per experiment, compared and decided.** Compare against the *accepted baseline* (`python -m kgkit ledger baseline`), not the highest CV, with the same seed(s) and folds: `python -m kgkit ledger compare <new> --truth data/train.csv:<target> --folds data/folds.csv:fold`. Judge noise by the paired per-fold differences, not the fold std (fold difficulty cancels). Record the verdict with `python -m kgkit ledger decide <new> keep|discard|inconclusive "<why>" --parent <baseline>`. Ideas live in `python -m kgkit backlog`.
4. **Trust CV, verify with LB.** Record every public LB score with `python -m kgkit ledger lb <id> <score>` and watch the CV↔LB correlation. A rising LB with flat CV is overfitting the public split.
5. **No target leakage**: target-derived statistics are computed out-of-fold; scalers/encoders/feature selection are fit inside the fold; no future information in time-series features.
6. **Validate every submission** with `python -m kgkit validate subs/<file>.csv data/sample_submission.csv` before submitting.
7. **Commit code before long runs** so the ledger's git hash points at the exact code.

## Commands

- `/kaggle-grandmaster:kg-status` — where we stand
- `/kaggle-grandmaster:kg-experiment <idea>` — run one disciplined experiment
- `/kaggle-grandmaster:kg-grind` — autonomous improvement loop
- `/kaggle-grandmaster:kg-gpu` — train on Kaggle GPUs within the weekly quota (screen, promote, resume)
- `/kaggle-grandmaster:kg-ensemble` — blend the best diverse models
- `/kaggle-grandmaster:kg-submit <file>` — validate, submit, log
- `/kaggle-grandmaster:kg-final` — choose the final submissions

## Current plan / notes

<!-- Keep a short running log here: what worked, what did not, what to try next. -->
