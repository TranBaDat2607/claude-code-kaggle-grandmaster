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
2. `python -m kgkit status` and `python -m kgkit ledger baseline`: the accepted baseline, its fold
   scores and std.

## plan
`gpu plan`, then propose a concrete schedule until the next reset and until the deadline: which screens
(paired two per session), which promotions, and what is reserved for finals and inference. Flag
use-it-or-lose-it hours before the reset.

## screen <idea> [<idea2>]
1. Implement each idea behind a flag in the training script and run `--smoke` locally. Commit.
2. Build one session for both screens on fold 0, with reduced cost settings that are identical for the
   baseline comparison. If no control screen of the baseline exists at these settings yet, make it one of
   the two jobs:
   `gpu build --name <a> --folds 0 --cmd "... --folds {fold}" --extra-job <b> "... --folds {fold}"
   --prune-against baseline --hours <~1.3x estimate>` (attach preprocessed data with `--dataset`).
3. Confirm with the user (quota is spent), then `gpu push`. Run `gpu wait <kernel> --collect` in the background.
4. Decide with a paired test against the control on that fold:
   `python -m kgkit ledger compare artifacts/<a> artifacts/<control> --truth data/train.csv:<target>
   --folds data/folds.csv:fold --fold 0`. PROMOTE (z ≥ 2) / DISCARD / RE-TEST (positive but within noise:
   second fold or seed). Record it on the backlog item (`kgkit backlog set <n> --screen-gain <gain>`) and in
   CLAUDE.md.

## run <experiment>
Full-fold run of a promoted idea: `gpu build` with all folds, `--hours` from the measured
cost per fold × folds ÷ 2 + overhead. Confirm, push, wait, collect. Then `kgkit ledger compare <id>`
against the accepted baseline, `kgkit ledger decide`, record `--full-gain` on the backlog item, and report
the ledger id, CV ± std, the paired gain and verdict, and the GPU-hours spent.

## resume <kernel>
Rebuild the same command with `--resume-from <kernel> --slug <name>-train-r<N>`. Finished folds are
skipped and timed-out folds continue from their checkpoint.

## status / collect <kernel>
`gpu status`, or `gpu collect <kernel>`. Act on every diagnostic line (idle GPU, input-bound, deadline,
pruned) as the skill's table says.

## Report
Hours used / remaining / until the reset, runs launched or collected with results (ledger ids, fold deltas),
diagnostics and fixes, and the next 1-3 GPU actions in priority order.
