---
name: kg-submit
description: Validate a submission file, check the daily budget, submit it to Kaggle with a ledger-linked message, poll for the public score and record it in the ledger.
argument-hint: <submission-file> [experiment-id]
allowed-tools: [Bash, PowerShell, Read, Glob, Grep]
disable-model-invocation: true
---

# /kg-submit

Arguments: `$ARGUMENTS`

1. Resolve the file and the experiment id (from the argument, the ledger `submission` field, or the
   file name `subs/<exp_id>.csv`). Show the experiment's CV.
2. Validate: `python -m kgkit validate <file> data/sample_submission.csv` — stop on errors.
3. Budget: `kaggle competitions submissions <slug> --format json` → count today's (UTC) submissions vs
   `daily_submissions` in `.kaggle-gm/competition.json`. Warn if this uses the last slot.
4. Sanity: compare this file's prediction distribution with the previous best submission (correlation,
   mean); if it is nearly identical (corr > 0.999) say the slot is probably wasted.
5. Submit: `kaggle competitions submit <slug> -f <file> -m "<exp_id> | CV <cv> | <short note>"`.
   (Code competitions: submit the notebook version — see `code-competitions`.)
6. Poll `kaggle competitions submissions <slug> --format json` every ~60 s (max ~15 min) until the
   score appears; then `python -m kgkit ledger lb <exp_id> <public_score>` and show the updated CV-LB
   correlation from `python -m kgkit status`.
7. Interpret: does LB move with CV as expected? If not, suggest `/kg-debug`.
