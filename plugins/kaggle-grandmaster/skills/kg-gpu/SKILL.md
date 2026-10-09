---
name: kg-gpu
description: Run training on Kaggle's GPUs within the weekly quota — plan the GPU budget, screen ideas on one fold (two per 2xT4 session, with pruning), promote winners to full-fold remote runs, resume timed-out runs, collect results into the ledger with GPU-hour accounting.
argument-hint: "[plan | screen <idea> [<idea2>] | run <experiment> | resume <kernel> | status | collect <kernel>]"
allowed-tools: [Bash, PowerShell, Read, Write, Edit, Glob, Grep, Agent]
disable-model-invocation: true
---

# /kg-gpu: quota-aware remote training

Request: `$ARGUMENTS` (default: `plan`).

Load the `kaggle-gpu` skill first and follow its protocol. Commands below are
`python -m kgkit gpu ...` (PYTHONPATH=$KGKIT_HOME).

## Always first
1. `gpu quota` and `gpu log`: hours plannable, reset time, runs in flight, measured cost per fold.
2. `python -m kgkit status`: current best experiment, its fold scores and std (the baseline).

## plan
`gpu plan`, then propose a concrete schedule until the next reset and until the deadline: which screens
(paired two per session), which promotions, and what is reserved for finals and inference. Flag
use-it-or-lose-it hours before the reset.

## screen <idea> [<idea2>]
1. Implement each idea behind a flag in the training script and run `--smoke` locally. Commit.
2. Build one session for both screens on fold 0, with reduced cost settings that are identical for the baseline comparison:
   `gpu build --name <a> --folds 0 --cmd "... --folds {fold}" --extra-job <b> "... --folds {fold}"
   --prune-against best --hours <~1.3x estimate>` (attach preprocessed data with `--dataset`).
3. Confirm with the user (quota is spent), then `gpu push`. Run `gpu wait <kernel> --collect` in the background.
4. Read the `collect` summary: Δ vs the baseline's same fold, compared with the fold std. Then decide
   PROMOTE / DISCARD / RE-TEST (second fold or seed), and note the result in CLAUDE.md and the backlog.

## run <experiment>
Full-fold run of a promoted idea: `gpu build` with all folds, `--hours` from the measured
cost per fold × folds ÷ 2 + overhead. Confirm, push, wait, collect. Then report the ledger id,
CV ± std, Δ vs the best, and the GPU-hours spent.

## resume <kernel>
Rebuild the same command with `--resume-from <kernel> --slug <name>-train-r<N>`. Finished folds are
skipped and timed-out folds continue from their checkpoint.

## status / collect <kernel>
`gpu status`, or `gpu collect <kernel>`. Act on every diagnostic line (idle GPU, input-bound, deadline,
pruned) as the skill's table says.

## Report
Hours used / remaining / until the reset, runs launched or collected with results (ledger ids, fold deltas),
diagnostics and fixes, and the next 1-3 GPU actions in priority order.
