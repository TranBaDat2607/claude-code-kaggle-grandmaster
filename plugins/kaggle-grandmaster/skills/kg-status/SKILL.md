---
name: kg-status
description: Show where the competition stands — metric, deadline, best CV, best LB, CV-LB correlation, recent experiments, submissions left today — and recommend the next actions.
allowed-tools: [Bash, PowerShell, Read, Glob, Grep]
---

# /kg-status

1. `python -m kgkit status -n 15` (kgkit home from the session context or `$KGKIT_HOME`).
2. If the Kaggle CLI is authenticated: `kaggle competitions submissions <slug> --format json` →
   submissions used today (UTC) and the latest scores not yet attached to the ledger (offer to
   attach them with `python -m kgkit ledger lb`).
3. Days left until the deadline; which endgame phase applies (see `final-submission-selection`).
4. Read `reports/backlog.md` and the CLAUDE.md notes.

Present a compact dashboard and the top 3 recommended next actions with reasons.
