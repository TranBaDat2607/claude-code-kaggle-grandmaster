---
name: kg-submit
description: "Validates a submission file, checks the daily budget, submits it to Kaggle and records the score. Use to submit a file."
argument-hint: <submission-file> [experiment-id]
allowed-tools:
  - Read
  - Glob
  - Grep
  - 'Bash(kgkit validate *)'
  - 'Bash(kgkit status *)'
  - 'Bash(kgkit ledger *)'
  - 'Bash(kaggle competitions submission-limits *)'
  - 'PowerShell(kaggle competitions submission-limits *)'
  - 'Bash(kaggle competitions submissions *)'
  - 'PowerShell(kaggle competitions submissions *)'
disable-model-invocation: true
---

# /kg-submit

Run kgkit with the Bash tool as `kgkit <command>`: the plugin puts it on PATH, also on Windows (Git Bash),
and this command pre-approves the kgkit calls it makes. Older text that says `python -m kgkit` means the same
command. Submitting, pushing or publishing still asks.

Arguments: `$ARGUMENTS`

1. Resolve the file and the experiment id (from the argument, the ledger `submission` field, or the
   file name `subs/<exp_id>.csv`). Show the experiment's CV.
2. Validate: `kgkit validate <file> data/sample_submission.csv` — stop on errors.
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
   ~30 min in the background instead); then `kgkit ledger lb <exp_id> <public_score>` and show
   the updated CV-LB correlation from `kgkit status`.
7. Interpret: does LB move with CV as expected? If not, suggest `/kg-debug`.
