---
name: kg-cv
description: "Designs, verifies and freezes the cross-validation folds (data/folds.csv). Use when the workspace has no frozen folds or the split must change."
argument-hint: "[strategy] [group-column]"
allowed-tools:
  - Read
  - Glob
  - Grep
  - 'Bash(kgkit config *)'
  - 'Bash(kgkit folds *)'
  - 'Bash(kgkit ledger *)'
---

# /kg-cv

Run kgkit with the Bash tool as `kgkit <command>`: the plugin puts it on PATH, also on Windows (Git Bash),
and this command pre-approves the kgkit calls it makes. Older text that says `python -m kgkit` means the same
command. Submitting, pushing or publishing still asks.

Arguments: `$ARGUMENTS` (optional strategy override and group column).

Follow the `validation-strategy` skill.

If decisions in this competition are made on a fixed labelled holdout plus the public LB (no K-fold OOF:
e.g. a small expert-labelled set, or an accepted baseline forked from public inference notebooks), do not
invent folds. Record the mode and the holdout labels instead:
`kgkit config validation=holdout+lb holdout_n=<rows> eval_truth=<FILE>:<label cols, comma-separated>`
(plus `eval_groups=<FILE>:<col>` when several rows share a patient/study), so the session brief stops
asking for `data/folds.csv`. Log each candidate with its holdout predictions
(`kgkit ledger add <name> auto --pred <FILE>:<cols>`): the score is recomputed from them and compared,
paired, against the accepted baseline. Do not judge differences by the holdout size alone: two models
scored on the same rows differ far less by chance than one score varies.

1. Determine how train/test were split (recon report, data page, adversarial validation, time
   columns, entity ids). State the evidence.
2. Pick the scheme from the decision table; if groups are suspected but not explicit, look for them
   (duplicate feature vectors, entity-like columns, image hashes, text near-duplicates).
3. Build: `kgkit folds data/train.csv --strategy <s> --target <y> [--group <g>] --id-col
   <id> --n-splits <k> --out data/folds.csv` (time series: write a split script using
   `kgkit.cv.time_series_splits` with gap ≥ horizon, saved as `src/splits.py`).
4. Verify with the fold report (balance, group confinement, temporal order).
5. Freeze: commit the script/seed (folds.csv itself may be in data/ which is git-ignored — record
   the exact command in CLAUDE.md so it is reproducible). Regenerated on another machine or OS, it
   keeps its `folds_hash` when every row gets the same fold (the ledger fingerprints the fold column,
   not the file's bytes); if `ledger compare` then warns of different fold splits, the split changed.
6. If a previous fold scheme existed, warn that old OOFs are no longer blendable with new ones.
