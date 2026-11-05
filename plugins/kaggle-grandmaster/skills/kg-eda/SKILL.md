---
name: kg-eda
description: Competition-focused EDA — red flags, leaks, drift, hidden groups, adversarial validation — written to reports/eda.md.
argument-hint: "[train-file] [test-file]"
allowed-tools:
  - Read
  - Glob
  - Grep
  - Agent
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit eda *)'
  - 'Bash(python -m kgkit eda *)'
  - 'PowerShell(python -m kgkit eda *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit adv *)'
  - 'Bash(python -m kgkit adv *)'
  - 'PowerShell(python -m kgkit adv *)'
---

# /kg-eda

Run kgkit with the Bash tool, as `PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit <command>`: this command
pre-approves that form for the kgkit calls it makes. Submitting, pushing or publishing still asks (so does a
PowerShell `$env:` assignment).

Arguments: `$ARGUMENTS` (defaults: `data/train.*` and `data/test.*`; target/id from
`.kaggle-gm/competition.json`).

Run kgkit with the absolute "kgkit home" path from the session context (or `$KGKIT_HOME`):

1. Tabular: `python -m kgkit eda <train> --test <test> --target <y> --id-col <id> --out reports/eda.md`
   then `python -m kgkit adv <train> <test> --drop <id>,<y> --save-p artifacts/adv_p_test.csv`.
2. Non-tabular (images/audio/text) or anything needing deeper investigation: delegate to the
   `data-detective` agent with the dataset description.
3. Interpret every red flag for the user: what it means for validation, features, and leaks; the
   concrete action for each.
4. Append the key findings to the notes section of the workspace CLAUDE.md.
