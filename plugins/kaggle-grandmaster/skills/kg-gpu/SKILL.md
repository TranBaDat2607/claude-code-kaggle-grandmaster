---
name: kg-gpu
description: "Runs training on Kaggle GPUs within the weekly quota: plan, one-fold screens, full runs, resume, collect. Use to spend GPU hours on planned experiments."
argument-hint: "[plan | screen <idea> [<idea2>] | run <experiment> | local <experiment> | resume <kernel> | status | collect <kernel>]"
allowed-tools:
  - Read
  - Glob
  - Grep
  - Agent
  - 'Bash(kgkit gpu quota *)'
  - 'Bash(kgkit gpu plan *)'
  - 'Bash(kgkit gpu build *)'
  - 'Bash(kgkit gpu status *)'
  - 'Bash(kgkit gpu wait *)'
  - 'Bash(kgkit gpu collect *)'
  - 'Bash(kgkit gpu release *)'
  - 'Bash(kgkit gpu log *)'
  - 'Bash(kgkit backlog *)'
  - 'Bash(kgkit ledger *)'
  - 'Bash(kgkit status *)'
disable-model-invocation: true
---

# /kg-gpu: quota-aware remote training

Run kgkit with the Bash tool as `kgkit <command>`: the plugin puts it on PATH, also on Windows (Git Bash),
and this command pre-approves the kgkit calls it makes. Older text that says `python -m kgkit` means the same
command. Submitting, pushing or publishing still asks.

Request: `$ARGUMENTS` (default: `plan`).

Load the `kaggle-gpu` skill first and follow its protocol. Commands below are `kgkit gpu ...`.

## Always first
1. `gpu quota` and `gpu log`: GPU and TPU hours plannable, reset time, runs in flight (and which pushes
   hold a reservation), measured cost per fold. With several accounts registered in the Kaggle pane,
   `gpu quota --all` (the hours on each) and `gpu plan` (quota-bound or time-bound, counting them all).
2. `kgkit status` and `kgkit ledger baseline`: the accepted baseline, its fold
   scores and std.

## plan
`gpu plan`, then propose a concrete schedule until the next reset and until the deadline: which screens
(paired two per session), which promotions, and what is reserved for finals and inference. Flag
use-it-or-lose-it hours before the reset.

## screen <idea> [<idea2>]
1. Implement each idea behind a flag in the training script and run `--smoke` locally. Commit. A new script, or a
   change to its input pipeline: run its `--bench` first (locally, or in a 0.3 h session) and fix an INPUT-BOUND or
   GPU IDLE verdict before spending hours of quota on it (skill `compute-efficiency`).
2. Build one session for both screens on fold 0, with reduced cost settings that are identical for the
   baseline comparison. If no control screen of the baseline exists at these settings yet, make it one of
   the two jobs:
   `gpu build --name <a> --folds 0 --cmd "... --folds {fold}" --extra-job <b> "... --folds {fold}"
   --prune-against baseline --hours <~1.3x estimate>` (attach preprocessed data with `--dataset`).
3. State the stop rule next to the promote rule (e.g. "stop: pruned against the control, margin = its fold std;
   promote: z >= 2"), then confirm with the user (quota is spent) and `gpu push`. Run `gpu wait <kernel> --collect`
   in the background.
4. Decide with a paired test against the control on that fold:
   `kgkit ledger compare artifacts/<a> artifacts/<control> --truth data/train.csv:<target>
   --folds data/folds.csv:fold --fold 0`. PROMOTE (z ≥ 2) / DISCARD / RE-TEST (positive but within noise:
   second fold or seed). Record it on the backlog item (`kgkit backlog set <n> --screen-gain <gain>`); CLAUDE.md
   `## Current state` changes only if the plan does.

## run <experiment>
Full-fold run of a promoted idea: `gpu build` with all folds, `--hours` from the measured
cost per fold × folds ÷ 2 + overhead. Confirm, push, wait, collect. Then `kgkit ledger compare <id>`
against the accepted baseline, `kgkit ledger decide`, record `--full-gain` on the backlog item, and report
the ledger id, CV ± std, the paired gain and verdict, and the GPU-hours spent.

## resume <kernel>
Rebuild the same command with `--resume-from <kernel> --slug <name>-train-r<N>`. Finished folds are
skipped and timed-out folds continue from their checkpoint.

## status / collect <kernel>
`gpu status` (re-checks every run still marked in flight and frees the quota of finished ones), or
`gpu collect <kernel>`. Act on every diagnostic line (idle GPU, input-bound, deadline, pruned) as the
skill's table says. A run that was deleted or never started: `gpu release <kernel>`.

## local <experiment>
On a machine with its own GPU, the same `gpu build` folder runs there: `gpu push <dir> --backend local [--hours H]`,
then `gpu wait local/<slug> --collect` in the background. No quota is spent, so it needs no quota check; it still
fills the user's GPU, so say what will run. Use it for screens and runs that fit the local card's memory, and Kaggle
for the rest (the skill's section 2).

## Other kernels (notebooks, inference, holdout diagnostics)
`gpu push <kernel-dir> --hours <cap>` works on any folder with `kernel-metadata.json`: same quota,
input and session checks, `-t` cap and run log. `gpu collect <kernel>` brings its files back to
`data/ext/kout-<slug>/` (the folder `--input-layout nested` mounts a kernel output at), takes the outcome from
Kaggle's status (complete / error / cancelled) and accounts the session's hours; nothing goes into the ledger (no
`kg-train.json` maps files to experiments), so log OOFs with `ledger import` or a score with `ledger add`.
The cap is a hard kill (no runner stops the script first): set it to ~1.3-1.5x the measured runtime, and make
sure the kernel saves its model before a long inference phase. If a cap stops it, `gpu wait` says so and lists the
files it saved; a re-push with the same cap warns.

## TPU
When the GPU quota is gone (or a run is longer than a GPU session), follow the skill's TPU reference (`references/tpu-xla.md`):
validate the XLA port off-quota, then `gpu build --accelerator TpuV5E8 --hours <= 9`, and queue it about
half a day before it is needed. A queued TPU job can only be removed by deleting the kernel: ask first.

## Report
Hours used / remaining / until the reset, runs launched or collected with results (ledger ids, fold deltas),
diagnostics and fixes, and the next 1-3 GPU actions in priority order.
