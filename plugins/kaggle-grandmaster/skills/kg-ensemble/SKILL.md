---
name: kg-ensemble
description: "Builds the best honest ensemble from ledger experiments and writes a blended submission. Use when several validated models exist."
argument-hint: "[experiment ids...] [--method hill|weights|rank|stack|residual] [--prune N]"
allowed-tools:
  - Read
  - Glob
  - Grep
  - Agent
  - 'Bash(kgkit blend *)'
  - 'Bash(kgkit ledger *)'
---

# /kg-ensemble

Run kgkit with the Bash tool as `kgkit <command>`: the plugin puts it on PATH, also on Windows (Git Bash),
and this command pre-approves the kgkit calls it makes. Older text that says `python -m kgkit` means the same
command. Submitting, pushing or publishing still asks.

Arguments: `$ARGUMENTS` (optional explicit experiment ids; otherwise choose from the ledger).

Delegate to the `ensemble-architect` agent, or follow these steps directly for a small ledger:

1. **Audit the candidates**: same folds, same row order, same prediction scale. Drop any that differ.
2. **Check diversity**: OOF correlations; a near-duplicate adds nothing.
3. **Blend**: `kgkit blend --exp ... --truth data/train.csv:<target> --folds data/folds.csv:fold --method hill
   --sample data/sample_submission.csv`, then the other methods. For a large library, `--prune 40` first, then
   `--method stack --levels 1|2` or `--method residual --base <id>`.
4. **Judge on honest scores**: compare in-sample with honest (nested) scores. Accept the winner only if
   `kgkit ledger compare` says it beats the simpler blend; otherwise keep the simpler one.
5. **Re-tune post-processing** (thresholds, rounding) on the blended OOF.
6. **Validate the submission file**; if validation fails, fix it and validate again before reporting.

Blends that include public submission files (no OOF) cannot be scored this way: log them LB-only
(`kgkit ledger add <name> lb-only --sub <file>`, then `kgkit ledger lb`) and follow `ensembling` §6.

Report members, weights, single vs blended (honest) CV, and whether it deserves a submission slot. A new single
model kept for its diversity is recorded `kgkit ledger decide <id> keep --member`: a plain keep would make it the
accepted baseline, and the next `ledger compare` of an ensemble would run against it.
