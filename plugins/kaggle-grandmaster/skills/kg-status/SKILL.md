---
name: kg-status
description: Show where the competition stands — metric, deadline, accepted baseline, best CV, best LB, CV-LB correlation, recent experiments and decisions, top of the backlog, submissions left today — and recommend the next actions.
allowed-tools: [Read, Glob, Grep]
---

# /kg-status

1. `python -m kgkit status -n 15` (kgkit home from the session context or `$KGKIT_HOME`). It prints
   the deadline with days left and the endgame phase (see `final-submission-selection`), submissions
   logged today (UTC) against the daily limit (plus Kaggle's own count when the last submit printed
   one) and the cached GPU quota.
2. If the Kaggle CLI is authenticated: `kaggle competitions submissions <slug> --format json` →
   submissions teammates or the website used today, which the local log cannot see, and the latest
   scores not yet attached to the ledger (offer to attach them with `python -m kgkit ledger lb`).
3. If the GPU quota is stale or missing and remote runs are planned: `python -m kgkit gpu quota`.
4. `python -m kgkit backlog list` and the CLAUDE.md notes. Flag experiments that were run but never
   decided (no `decision` in the ledger table) and offer to compare/decide them.
5. If `status` warned about many decisions on the same folds, recommend the CV re-check
   (`grandmaster-playbook`, "Guarding against overfitting the CV itself").

Present a compact dashboard and the top 3 recommended next actions with reasons.
