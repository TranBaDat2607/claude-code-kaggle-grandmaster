---
name: kg-eda
description: "Competition EDA into reports/eda.md: leaks, drift, hidden groups, adversarial validation. Use after the data is downloaded, before designing folds."
argument-hint: "[train-file] [test-file]"
allowed-tools:
  - Read
  - Glob
  - Grep
  - Agent
  - 'Bash(kgkit eda *)'
  - 'Bash(kgkit adv *)'
---

# /kg-eda

Run kgkit with the Bash tool as `kgkit <command>`: the plugin puts it on PATH, also on Windows (Git Bash),
and this command pre-approves the kgkit calls it makes. Older text that says `python -m kgkit` means the same
command. Submitting, pushing or publishing still asks.

Arguments: `$ARGUMENTS` (defaults: `data/train.*` and `data/test.*`; target/id from
`.kaggle-gm/competition.json`).

1. Tabular: `kgkit eda <train> --test <test> --target <y> --id-col <id> --out reports/eda.md`
   then `kgkit adv <train> <test> --drop <id>,<y> --save-p artifacts/adv_p_test.csv`.
2. Non-tabular (images/audio/text) or anything needing deeper investigation: delegate to the
   `data-detective` agent with the dataset description.
3. Interpret every red flag for the user: what it means for validation, features, and leaks; the
   concrete action for each.
4. The findings stay in `reports/eda.md`. In the workspace CLAUDE.md, add only what every session
   must keep in mind (a leak, a shift, a trap in the data) to `## Current state`, one line each with a pointer
   to the report: CLAUDE.md loads into every session, so it holds the conclusions, not the analysis.
