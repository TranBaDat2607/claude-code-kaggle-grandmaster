---
name: kg-status
description: Show where the competition stands — metric, deadline, accepted baseline, best CV, best LB, CV-LB correlation, recent experiments and decisions, top of the backlog, submissions left today — and recommend the next actions.
allowed-tools:
  - Read
  - Glob
  - Grep
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit status *)'
  - 'Bash(python -m kgkit status *)'
  - 'PowerShell(python -m kgkit status *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit backlog *)'
  - 'Bash(python -m kgkit backlog *)'
  - 'PowerShell(python -m kgkit backlog *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit gpu quota *)'
  - 'Bash(python -m kgkit gpu quota *)'
  - 'PowerShell(python -m kgkit gpu quota *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit ledger *)'
  - 'Bash(python -m kgkit ledger *)'
  - 'PowerShell(python -m kgkit ledger *)'
  - 'Bash(kaggle competitions submissions *)'
  - 'PowerShell(kaggle competitions submissions *)'
---

# /kg-status

Run kgkit with the Bash tool, as `PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit <command>`: this command
pre-approves that form for the kgkit calls it makes. Submitting, pushing or publishing still asks (so does a
PowerShell `$env:` assignment).

1. `python -m kgkit status -n 15` (kgkit home from the session context or `$KGKIT_HOME`). It prints
   the deadline with days left and the endgame phase (see `final-submission-selection`), submissions
   logged today (UTC) against the daily limit (plus Kaggle's own count when the last submit printed
   one) and the cached GPU quota. It first adds to the log the submissions Kaggle lists that the hooks
   never saw (a script, the website, a teammate) and says whether the count was "checked against
   Kaggle's list" or is "local log only" (no CLI or login, or `--offline`): trust a local-only count
   no further than that.
2. If the Kaggle CLI is authenticated: `kaggle competitions submissions <slug> --format json` →
   the latest scores not yet attached to the ledger. Offer to attach them: `python -m kgkit ledger sync-lb` does
   it for every submission that names its record; `python -m kgkit ledger lb` does one by hand.
3. If the GPU quota is stale or missing and remote runs are planned: `python -m kgkit gpu quota`.
4. `python -m kgkit backlog list` and the CLAUDE.md notes. Flag experiments that were run but never
   decided (no `decision` in the ledger table) and offer to compare/decide them.
5. If `status` warned about many decisions on the same folds, recommend the CV re-check
   (`grandmaster-playbook`, "Guarding against overfitting the CV itself").

Present a compact dashboard and the top 3 recommended next actions with reasons.
