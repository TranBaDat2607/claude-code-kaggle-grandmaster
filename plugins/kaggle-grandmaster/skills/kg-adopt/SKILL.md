---
name: kg-adopt
description: Adopt an existing mid-competition Kaggle repo in place — write .kaggle-gm/ (competition facts), import the past Kaggle submissions and the experiments the repo already tracks, mark the accepted baseline — so the hooks, budget checks and kgkit work. Use instead of kg-start when the competition work already exists (kernel folders, an experiment log, a validation set).
argument-hint: [competition-slug]
allowed-tools: [Read, Glob, Grep]
---

# /kg-adopt — adopt an existing competition repo

Arguments: `$ARGUMENTS` (competition slug; optional when the repo's `kernel-metadata.json` or README names it).

kgkit: use the absolute path from the session context ("kgkit home"), else `$KGKIT_HOME`, else
`${CLAUDE_PLUGIN_ROOT}`. Invoke as `PYTHONPATH=<kgkit home> python -m kgkit <cmd>`
(PowerShell: `$env:PYTHONPATH="<kgkit home>"; python -m kgkit <cmd>`).

The repo already has its own layout, scripts, ledger and validation. **Adopt, do not rebuild**: no data
download, no new folds, no new baseline, no moving files. The point is to switch on what the plugin
does around the existing work: submission logging and the daily budget, the session status brief,
`kgkit status` / `ledger` / `kernels top` / `discussions`.

1. **Facts.** Read the README, kernel folders (`kernel-metadata.json`: `competition_sources`,
   `enable_internet`, `enable_gpu`) and any notes to find: slug, metric and direction, task, target,
   id column, code competition (runtime limit, internet), daily submissions, deadline. Fill gaps with
   `kaggle competitions list -s <slug>` (deadline) and the `competition-recon` skill; ask the user
   only for what you cannot determine.
2. **Workspace.** Run from the repo root:
   `python -m kgkit adopt <slug> --metric <m> --task <t> --target <col> --id-col <id>
   [--code-competition --runtime-hours H --no-internet] [--daily-subs N] --deadline <YYYY-MM-DD>
   [--validation holdout+lb --holdout-n <rows> --eval-truth <FILE>:<label cols>]`. Use `holdout+lb` when
   the repo decides on a fixed labelled holdout plus the public LB rather than K-fold OOF (no
   `data/folds.csv` will be asked for); `--eval-truth` names the holdout labels (one column per label of a
   multilabel target) so candidates logged with `ledger add <name> auto --pred <FILE>:<cols>` are compared
   paired on them. Add `--public-lb-n <rows>` when the public split's size is known.
   Fields can be changed later with `python -m kgkit config key=value`.
   It writes only `.kaggle-gm/competition.json` (no folders, `.gitignore` or `CLAUDE.md`) and imports
   the history from `kaggle competitions submissions <slug>` into `.kaggle-gm/submissions.jsonl`, so
   today's used submissions count against the budget. If the CLI is not authenticated, say so and
   continue; the import can be re-run later with `--submissions-file`. Re-running is safe: existing
   state is kept and already-logged submissions are skipped.
3. **Ledger.** Find the repo's own experiment record (a CSV/markdown/JSON log, a results folder,
   commit messages). For each experiment that matters (the current best, the ones behind
   submissions, the main alternatives, not every run):
   - with OOF predictions on file: `python -m kgkit ledger import <name> <oof.npy|FILE:COL>
     [--test <test preds>] --truth <train FILE:target> --folds <FILE:fold> --source notebook --notes "<origin>"`
     so the CV is recomputed on the repo's folds and the run is blendable;
   - with a score only: `python -m kgkit ledger add <name> <cv> <notes>`;
   - with no validation score at all (a fork of public notebooks, a blend with public files):
     `python -m kgkit ledger add <name> lb-only <notes>`, then attach its public score.
   Attach public scores you can match to a submission with `python -m kgkit ledger lb <id> <score>`.
   Mark the model the user currently builds on with `python -m kgkit ledger decide <id> baseline`.
   Do not invent numbers: skip an experiment whose CV you cannot find, and say so.
4. **Folds.** If the repo has a fixed validation split, point to it (or export it as `data/folds.csv`
   with `id,fold` columns when the training code can read it). Do not redraw folds mid-competition.
5. **Rules.** If the repo has a `CLAUDE.md`, leave it alone except, with the user's agreement, a short
   kaggle-grandmaster section (log every run, compare against the accepted baseline, validate before
   submitting, the `/kg-*` commands). If it has none, offer to create one from
   `${CLAUDE_PLUGIN_ROOT}/templates/CLAUDE.competition.md`, adapted to the repo's real paths.
6. **Commit** `.kaggle-gm/` (it is the experiment memory) once the user agrees.

Finish with `python -m kgkit status`: facts, submissions today vs the daily limit, the accepted
baseline and best public LB, what could not be imported, and the next step (usually `/kg-ideas` or
`/kg-experiment`). Do not submit or push kernels from this command.
