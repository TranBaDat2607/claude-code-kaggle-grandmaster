---
name: kg-recon
description: "Researches a Kaggle competition and writes reports/recon.md. Use before modelling a new competition."
argument-hint: <competition-slug>
allowed-tools:
  - Read
  - Glob
  - Grep
  - Agent
  - 'Bash(kgkit pages *)'
  - 'Bash(kgkit discussions *)'
  - 'Bash(kgkit kernels top *)'
  - 'Bash(kgkit kernels lineage *)'
  - 'Bash(kgkit config *)'
  - 'Bash(kaggle competitions list *)'
  - 'PowerShell(kaggle competitions list *)'
  - 'Bash(kaggle competitions files *)'
  - 'PowerShell(kaggle competitions files *)'
---

# /kg-recon

Run kgkit with the Bash tool as `kgkit <command>`: the plugin puts it on PATH, also on Windows (Git Bash),
and this command pre-approves the kgkit calls it makes. Older text that says `python -m kgkit` means the same
command. Submitting, pushing or publishing still asks.

Competition: `$ARGUMENTS`

Launch the `competition-analyst` agent and the `solution-researcher` agent **in parallel** for this
competition. When both return, merge their findings into `reports/recon.md` (template in the
`competition-recon` skill) and `reports/prior-art.md`, update `.kaggle-gm/competition.json` if the
workspace exists, and present to the user:

1. TL;DR — what wins here and the biggest risks.
2. Metric + the post-processing it implies (see `metric-optimization`).
3. Recommended validation scheme and why it mirrors the test split.
4. Shake-up risk (low/med/high) with reasoning.
5. Ranked initial backlog (top 8), each with expected gain and cost.
6. Anything the user must confirm or do (accept rules, external data declarations, deadlines).
