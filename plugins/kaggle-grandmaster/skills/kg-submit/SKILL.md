---
name: kg-submit
description: Validate a submission file, check the daily budget, submit it to Kaggle with a ledger-linked message, poll for the public score and record it in the ledger.
argument-hint: <submission-file> [experiment-id]
allowed-tools:
  - Read
  - Glob
  - Grep
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit validate *)'
  - 'Bash(python -m kgkit validate *)'
  - 'PowerShell(python -m kgkit validate *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit status *)'
  - 'Bash(python -m kgkit status *)'
  - 'PowerShell(python -m kgkit status *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit ledger *)'
  - 'Bash(python -m kgkit ledger *)'
  - 'PowerShell(python -m kgkit ledger *)'
  - 'Bash(kaggle competitions submission-limits *)'
  - 'PowerShell(kaggle competitions submission-limits *)'
  - 'Bash(kaggle competitions submissions *)'
  - 'PowerShell(kaggle competitions submissions *)'
disable-model-invocation: true
---

# /kg-submit

Run kgkit with the Bash tool, as `PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit <command>`: this command
pre-approves that form for the kgkit calls it makes. Submitting, pushing or publishing still asks (so does a
PowerShell `$env:` assignment).

Arguments: `$ARGUMENTS`

1. Resolve the file and the experiment id (from the argument, the ledger `submission` field, or the
   file name `subs/<exp_id>.csv`). Show the experiment's CV.
2. Validate: `python -m kgkit validate <file> data/sample_submission.csv` — stop on errors.
3. Budget: `kaggle competitions submission-limits <slug> --json` → `numAllowedNow` is what the team has left
   today, teammates' and website submissions included (CLI 2.2.4+; with an older CLI, count today's UTC
   entries in `kaggle competitions submissions <slug> --format json` against `daily_submissions` in
   `.kaggle-gm/competition.json`). Warn if this uses the last slot.
4. Sanity: compare this file's prediction distribution with the previous best submission (correlation,
   mean); if it is nearly identical (corr > 0.999) say the slot is probably wasted. The hook asks before
   it sends a byte-identical file again.
5. Submit: `kaggle competitions submit <slug> -f <file> -m "<exp_id> | CV <cv> | <short note>"`.
   (Code competitions: submit the notebook version — see `code-competitions`.)
6. Poll `kaggle competitions submissions <slug> --format json` every ~60 s (max ~15 min) until the
   score appears (code competitions: Kaggle reruns the notebook first, which takes hours, so poll every
   ~30 min in the background instead); then `python -m kgkit ledger lb <exp_id> <public_score>` and show
   the updated CV-LB correlation from `python -m kgkit status`.
7. Interpret: does LB move with CV as expected? If not, suggest `/kg-debug`.
