---
name: kg-cv
description: Design, build, verify and freeze the cross-validation scheme (data/folds.csv) so it mirrors how the test set was split.
argument-hint: "[strategy] [group-column]"
allowed-tools: [Bash, PowerShell, Read, Write, Edit, Glob, Grep]
---

# /kg-cv

Arguments: `$ARGUMENTS` (optional strategy override and group column).

Follow the `validation-strategy` skill.

1. Determine how train/test were split (recon report, data page, adversarial validation, time
   columns, entity ids). State the evidence.
2. Pick the scheme from the decision table; if groups are suspected but not explicit, look for them
   (duplicate feature vectors, entity-like columns, image hashes, text near-duplicates).
3. Build: `python -m kgkit folds data/train.csv --strategy <s> --target <y> [--group <g>] --id-col
   <id> --n-splits <k> --out data/folds.csv` (time series: write a split script using
   `kgkit.cv.time_series_splits` with gap ≥ horizon, saved as `src/splits.py`).
4. Verify with the fold report (balance, group confinement, temporal order).
5. Freeze: commit the script/seed (folds.csv itself may be in data/ which is git-ignored — record
   the exact command in CLAUDE.md so it is reproducible).
6. If a previous fold scheme existed, warn that old OOFs are no longer blendable with new ones.
