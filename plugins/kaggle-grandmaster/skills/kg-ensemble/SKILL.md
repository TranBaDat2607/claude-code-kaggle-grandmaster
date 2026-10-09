---
name: kg-ensemble
description: Build the best honest ensemble from ledger experiments (diversity analysis, hill climbing / weights / rank / stacking with nested CV, post-processing re-tune) and write a validated blended submission.
argument-hint: "[experiment ids...] [--method hill|weights|rank|stack|residual] [--prune N]"
allowed-tools: [Bash, PowerShell, Read, Write, Edit, Glob, Grep, Agent]
---

# /kg-ensemble

Arguments: `$ARGUMENTS` (optional explicit experiment ids; otherwise choose from the ledger).

Delegate to the `ensemble-architect` agent (or follow its procedure directly for small ledgers):
candidate audit (same folds/row order/scale) → correlation & diversity → `python -m kgkit blend
--exp ... --truth data/train.csv:<target> --folds data/folds.csv:fold --method hill --sample
data/sample_submission.csv` (+ alternative methods) → compare in-sample vs honest scores → for large
libraries `--prune 40` then `--method stack --levels 1|2`, or `--method residual --base <id>` → accept the
winner only if `kgkit ledger compare` says it beats the simpler blend → re-tune thresholds/rounding on the
blended OOF → validate the submission.

Report members, weights, single vs blended (honest) CV, and whether it deserves a submission slot.
