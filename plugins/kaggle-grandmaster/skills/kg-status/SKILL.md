---
name: kg-status
description: Show where the competition stands — metric, deadline, accepted baseline, best CV, best LB, CV-LB correlation, recent experiments and decisions, top of the backlog, submissions left today — and recommend the next actions.
allowed-tools:
  - Read
  - Glob
  - Grep
  - 'Bash(kgkit status *)'
  - 'Bash(kgkit backlog *)'
  - 'Bash(kgkit gpu quota *)'
  - 'Bash(kgkit ledger *)'
  - 'Bash(kgkit standing *)'
  - 'Bash(kaggle competitions submissions *)'
  - 'PowerShell(kaggle competitions submissions *)'
---

# /kg-status

Run kgkit with the Bash tool as `kgkit <command>`: the plugin puts it on PATH, also on Windows (Git Bash),
and this command pre-approves the kgkit calls it makes. Older text that says `python -m kgkit` means the same
command. Submitting, pushing or publishing still asks.

Workspace state when this command loaded (`kgkit status -n 15`):

!`kgkit status -n 15 || true`

1. Read the status above (run `kgkit status -n 15` again only if it is missing). It prints
   the deadline with days left and the endgame phase (see `final-submission-selection`), submissions
   logged today (UTC) against the daily limit (plus Kaggle's own count when the last submit printed
   one) and the cached GPU quota. It first adds to the log the submissions Kaggle lists that the hooks
   never saw (a script, the website, a teammate) and says whether the count was "checked against
   Kaggle's list" or is "local log only" (no CLI or login, or `--offline`): trust a local-only count
   no further than that.
2. If the Kaggle CLI is authenticated: `kaggle competitions submissions <slug> --format json` →
   the latest scores not yet attached to the ledger. Offer to attach them: `kgkit ledger sync-lb` does
   it for every submission that names its record; `kgkit ledger lb` does one by hand.
3. If the GPU quota is stale or missing and remote runs are planned: `kgkit gpu quota`.
4. If `status` prints a **Permissions** line, offer `kgkit permissions --write` once (it lists what it adds).
5. `kgkit backlog list` and CLAUDE.md `## Current state` (`reports/journal.md` for the history). Flag experiments that were run but never
   decided (no `decision` in the ledger table) and offer to compare/decide them.
6. If `status` warned about many decisions on the same folds, recommend the CV re-check
   (`grandmaster-playbook`, "Guarding against overfitting the CV itself").
7. The public-LB line leads with the best score that is not public-LB fitted. Fitted files follow it,
   "display only": never present one as the team's standing. If `status` prints **LB-FITTING** (most recent
   submissions fit or probe the public LB while the honest track is flat), say so first: those files cannot
   raise the private score.
8. If `status` prints **PLATEAU** or **CROWD**, make `/kg-ideas` the first recommended action, aimed at what
   the stack lacks (a new model family, data view or signal), with the GPU hours it names. Do not recommend
   more members of an existing family, or public-LB fitting. If PLATEAU says no public board is cached, offer
   `kgkit standing`: it reads the whole board, which can take a few minutes, and
   caches it for the crowd check.

Present a compact dashboard and the top 3 recommended next actions with reasons.
