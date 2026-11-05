---
name: kg-recon
description: Research a Kaggle competition (metric, data, rules, timeline, top notebooks, discussions, similar past competitions, leak and shake-up risk) and write reports/recon.md.
argument-hint: <competition-slug>
allowed-tools:
  - Read
  - Glob
  - Grep
  - Agent
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit pages *)'
  - 'Bash(python -m kgkit pages *)'
  - 'PowerShell(python -m kgkit pages *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit discussions *)'
  - 'Bash(python -m kgkit discussions *)'
  - 'PowerShell(python -m kgkit discussions *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit kernels top *)'
  - 'Bash(python -m kgkit kernels top *)'
  - 'PowerShell(python -m kgkit kernels top *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit config *)'
  - 'Bash(python -m kgkit config *)'
  - 'PowerShell(python -m kgkit config *)'
  - 'Bash(kaggle competitions list *)'
  - 'PowerShell(kaggle competitions list *)'
  - 'Bash(kaggle competitions files *)'
  - 'PowerShell(kaggle competitions files *)'
---

# /kg-recon

Run kgkit with the Bash tool, as `PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit <command>`: this command
pre-approves that form for the kgkit calls it makes. Submitting, pushing or publishing still asks (so does a
PowerShell `$env:` assignment).

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
