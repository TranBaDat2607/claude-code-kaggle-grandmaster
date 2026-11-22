---
name: kg-experiment
description: "Runs one disciplined experiment against the accepted baseline and records keep or discard. Use when testing an idea, or when deciding and recording a finished experiment."
argument-hint: <hypothesis or change to test>
allowed-tools:
  - Read
  - Glob
  - Grep
  - Agent
  - 'Bash(kgkit backlog *)'
  - 'Bash(kgkit exp *)'
  - 'Bash(kgkit ledger *)'
  - 'Bash(kgkit status *)'
---

# /kg-experiment

Run kgkit with the Bash tool as `kgkit <command>`: the plugin puts it on PATH, also on Windows (Git Bash),
and this command pre-approves the kgkit calls it makes. Older text that says `python -m kgkit` means the same
command. Submitting, pushing or publishing still asks.

Hypothesis: `$ARGUMENTS`

Workspace state when this command loaded (`kgkit status --offline`):

!`kgkit status --offline -n 10 || true`

Follow the experiment protocol (`grandmaster-playbook` §2). Besides the training run, which prints its own paired
verdict, the bookkeeping is two kgkit calls: mark the idea running, then `kgkit exp finish`.

1. The state above names the **accepted baseline** (last KEEP, not the highest CV) and its CV ± std; run
   `kgkit status` only if it is missing or out of date. If nothing is accepted yet, make one first
   (`kgkit exp finish <id> baseline "<why>"`). If the idea is in the backlog, mark it
   `kgkit backlog set <n> --status running --first-experiment "<command you will run>"`; otherwise add it (same
   options). Keep its number `<n>` for step 6.
2. Restate the hypothesis, the *single* change, the baseline id, and the success criterion:
   paired gain ≥ ~2 standard errors of the per-fold differences (or of the OOF bootstrap) with most
   folds improving. Not "Δ > fold std": the fold std measures fold difficulty, which cancels in a
   paired comparison. For a run of hours, also state its stop rule up front: pruned against the baseline's curve
   (`gpu build --prune-against`, or `kgkit.budget` in a loop of your own), so cutting it short is decided in advance
   like the verdict.
3. Implement behind a flag/config in the existing script, never in a copy of it (`train_v2.py`): the record keeps
   the command it ran, so the flag says what changed. A quick probe that is not the experiment goes in `work/`. Smoke test, then full run on frozen folds
   with the baseline's seed(s). When the change touches the input pipeline, the resolution / `max_len` or the
   model size, run the script's `--bench` (or `kgkit bench --run "<cmd>"`) after the smoke test, so the
   cost in GPU time is known before the run. A pure speed change (AMP, compile, caching) is an experiment too:
   keep it on its speed-up with the score unchanged (skill `compute-efficiency`). For long runs (> ~10 min) run in the background and keep working on
   analysis meanwhile; or delegate the whole run to the `experiment-runner` agent.
4. Log to the ledger with OOF/test preds and `parent=<baseline id>` (notes: hypothesis); the templates take
   `--parent <baseline id>`. When they log, they print the paired comparison with it (per-fold deltas, folds
   improved, the paired SE, a row-level test on the target) and end with the `kgkit exp finish` call that records
   it: no separate comparison call. A script of your own does the same with
   `kgkit.experiment.report_comparison(rec, truth="data/train.csv:<target>")` after `Ledger().log`, and follows the
   [training contract](training-contract.md): frozen folds, `--folds`/`--assemble`, a unit table for a metric that
   is not one array of row predictions (segmentation, detection, 3D, problem sets, matches), `KG_SCREEN`. Many values
   of one lever are one sweep (`/kg-grind --sweep`), not one experiment each. Nothing about
   the model (stopping round, epoch, checkpoint) may be chosen on the fold it predicts: the templates early-stop on
   an inner slice / keep the last epoch and log `oof_selection`. When the comparison says the baseline was selected
   otherwise, re-run the baseline before deciding a small gain. In a `holdout+lb` workspace log the run with its
   holdout predictions, `kgkit ledger add <new> auto --pred <FILE>:<cols>`: the score is recomputed and compared,
   paired, with the accepted baseline right away (no folds). A run of another model line (a CNN beside the GBDT, a
   new backbone) takes `--track <name>`: the line's first run becomes its baseline (`kgkit exp finish <id>
   baseline --track cnn`), and its later runs are compared with that, not with the main line's. A run is compared
   with its nearest kept ancestor (`--parent`), else its line's baseline.
5. Decide with numbers, from the verdict the run printed: KEEP / DISCARD / INCONCLUSIVE. INCONCLUSIVE → second
   seed, logged with the same `--variant` so its spread counts as seed noise (or keep only if it adds no
   complexity/runtime); a flagged huge jump → `validation-auditor` before accepting. A `best of N` line means
   the run was one of N tries against that baseline (a sweep, or ideas tried in turn): KEEP then needs more than
   2 SE, because the luckiest of N noise runs gets that far; `exp finish` counts the tries as they are when it
   records. Never lower `--tries` to pass a run. The comparison needs other
   options (`--groups data/train.csv:<group>` when rows are clustered and `eval_groups` is unset, `--rope <gain>`
   for changes too small to matter)? Give them to `exp finish`: it then compares afresh. With no verdict printed
   (an older script), `kgkit exp finish <new>` alone compares and records nothing.
6. Record it, in one call: `kgkit exp finish <new> keep|discard|inconclusive "<gain, folds k/K>" --backlog <n>`.
   It records the decision with the baseline it was compared against (KEEP makes it the new baseline: the accepted
   decision made last is the baseline; `exp finish <id> baseline` restores any record) and closes the backlog idea
   with the result (`inconclusive` keeps it running for the second seed). Deciding against the verdict is allowed
   with the reason in the note. A model wanted for its diversity is not judged on its own CV: blend it
   (`kgkit blend`), which compares the new blend, paired, with the best accepted blend. The learning goes in the decision's "why" and the idea's result, not in CLAUDE.md, which
   loads into every session. Rewrite `## Current state` in CLAUDE.md only when this decision changes it (the plan,
   a direction now closed, a new standing gotcha); a longer narrative goes in `reports/journal.md`. Commit code. A
   sweep of runs is decided at once: `kgkit ledger decide 12..20 discard "<why>"`.
7. Suggest the next experiment based on what was learned. When it changes direction (a new model family, new data,
   ensembling, the final week), name the new phase first: `kgkit ledger phase "<name>"`.
