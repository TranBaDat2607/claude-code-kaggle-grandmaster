# Kaggle competition: {{SLUG}}

Workspace managed by the **kaggle-grandmaster** plugin. Read this before every session.

## Facts (keep updated — `.kaggle-gm/competition.json` is the source of truth)

- Metric: **{{METRIC}}** ({{DIRECTION}} is better). The exact implementation lives in `src/metric.py` and is unit-tested against a known value.
- Task: {{TASK}} — target: `{{TARGET}}`
- Code competition: {{CODE_COMP}}
- Final deadline: {{DEADLINE}} (always confirm on the competition page; deadlines are 23:59 UTC)

## Layout

```
data/         competition files and anything made from them (never edited, never committed)
src/          the pipeline: training / inference scripts and the modules they share (committed)
studies/      one folder per analysis a decision rests on: its script + a README (question, result, ledger ids)
notebooks/    exploration only, numbered (01_eda.ipynb); anything that matters moves into src/ or studies/
kernels/      Kaggle notebooks / datasets you write (code-competition inference); `gpu build` writes to .kaggle-gm/build/
third_party/  code that is not yours: a vendored kgkit (`kgkit vendor third_party/`), cloned libraries
work/         scratch, ignored by git: probes, quick checks, logs, downloads
artifacts/    per-experiment OOF + test predictions + models (artifacts/<exp_id>/, ignored)
subs/         submission CSVs, named after the experiment id (ignored)
reports/      EDA, recon, adversarial validation, final selection, post-mortem (markdown; logs go in work/)
.kaggle-gm/   competition.json (facts), ledger.jsonl (every experiment), backlog (committed); remote/, build/ ignored
```

When the competition ends, `/kaggle-grandmaster:kg-showcase` adds `solution/` (the final pipeline, for readers who want
the result) and `journey/` (how the ideas developed, written from the ledger) and makes the repo presentable. It can
only tell the journey the ledger recorded, so rule 9 below matters from day one.

## Non-negotiable rules

1. **The CV scheme is frozen** in `data/folds.csv` (or a documented time split). Every model uses those exact folds so OOFs are blendable. Change it only with a written reason in the ledger notes, and then re-run the models that matter.
2. **Every run is logged** with `kgkit.experiment.Ledger().log(...)` — CV, fold scores, params, features, OOF and test predictions. No unlogged results, no "I think it was 0.812".
3. **One change per experiment, compared and decided.** Compare against the *accepted baseline* (`python -m kgkit ledger baseline`), not the highest CV, with the same seed(s) and folds: `python -m kgkit ledger compare <new> --truth data/train.csv:<target> --folds data/folds.csv:fold`. Judge noise by the paired per-fold differences, not the fold std (fold difficulty cancels). Record the verdict with `python -m kgkit ledger decide <new> keep|discard|inconclusive "<why>" --parent <baseline>`. Ideas live in `python -m kgkit backlog`.
4. **Trust CV, verify with LB.** Record every public LB score with `python -m kgkit ledger lb <id> <score>` and watch the CV↔LB correlation. A rising LB with flat CV is overfitting the public split.
5. **No target leakage**: target-derived statistics are computed out-of-fold; scalers/encoders/feature selection are fit inside the fold; no future information in time-series features.
6. **Validate every submission** with `python -m kgkit validate subs/<file>.csv data/sample_submission.csv` before submitting. A wrapper script may do this and submit, but let `kaggle competitions submit` print Kaggle's reply (no `-q`, no redirect): the hook reads the script and logs the submit from that reply; any submit it misses, `python -m kgkit status` adds from Kaggle's list.
7. **Commit code at natural points.** Each record also keeps the exact code it ran from: when the tree has uncommitted
   changes, a snapshot commit under `refs/kg/<id>` (`git_snapshot` in the record; the working tree and the staging
   area are not touched). Never delete `refs/kg/*`: the journey's code links point at them.
8. **Competition Data never goes into git.** Kaggle's rules forbid publishing it, and that covers copies and subsets
   too: rows of `train.csv`, its text, its labels keyed by ID, notebook outputs that print them. Keep such files in
   `data/` (ignored). Derived labels or predictions keyed by the competition's IDs are a judgment call: decide on
   purpose. A file deleted in a later commit is still in history. `python -m kgkit audit-repo` checks every
   branch, the history and what `git add -A` would commit, and whether the GitHub remote is public. Run it before
   making the repo public.
9. **Keep the repo readable: it goes public when the competition ends.**
   - A variant is a flag (or a config file) of the script it varies, never a copy (`train_v2.py`): each record keeps
     its command (`cmd`), so the flag says what changed. A new script only for a new model family or pipeline stage,
     named for what it does.
   - Probes, one-off checks and every log go in `work/` (`> work/logs/<name>.log`). When a probe's result decides
     something, move it to `studies/<name>/` with a short README: the question, the result, the ledger ids it decided.
   - Name the phase when the work changes direction (a new model family, new data, ensembling, the final week):
     `python -m kgkit ledger phase "<name>"`. Decide every record; a sweep at once:
     `python -m kgkit ledger decide 12..20 discard "<why>"`. The journey is told phase by phase from these.

## Commands

- `/kaggle-grandmaster:kg-status` — where we stand
- `/kaggle-grandmaster:kg-experiment <idea>` — run one disciplined experiment
- `/kaggle-grandmaster:kg-grind` — autonomous improvement loop
- `/kaggle-grandmaster:kg-gpu` — train on Kaggle GPUs within the weekly quota (screen, promote, resume)
- `/kaggle-grandmaster:kg-ensemble` — blend the best diverse models
- `/kaggle-grandmaster:kg-submit <file>` — validate, submit, log
- `/kaggle-grandmaster:kg-final` — choose the final submissions
- `/kaggle-grandmaster:kg-postmortem`, then `/kaggle-grandmaster:kg-showcase` — after the deadline: what worked, then
  the repo made public with `solution/` and `journey/`

## Current plan / notes

<!-- Keep a short running log here: what worked, what did not, what to try next. -->
