---
name: kg-kernel
description: Package the current best solution for a Kaggle code competition — offline inference script, weight/code/wheel datasets, kernel metadata with internet off, runtime estimate on hidden-test size, push and poll.
argument-hint: "[experiment-id or blend id]"
allowed-tools:
  - Read
  - Glob
  - Grep
  - Agent
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit validate *)'
  - 'Bash(python -m kgkit validate *)'
  - 'PowerShell(python -m kgkit validate *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit ledger *)'
  - 'Bash(python -m kgkit ledger *)'
  - 'PowerShell(python -m kgkit ledger *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit status *)'
  - 'Bash(python -m kgkit status *)'
  - 'PowerShell(python -m kgkit status *)'
  - 'Bash(kaggle kernels status *)'
  - 'PowerShell(kaggle kernels status *)'
disable-model-invocation: true
---

# /kg-kernel

Run kgkit with the Bash tool, as `PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit <command>`: this command
pre-approves that form for the kgkit calls it makes. Submitting, pushing or publishing still asks (so does a
PowerShell `$env:` assignment).

Target: `$ARGUMENTS` (default: the best blend/experiment in the ledger).

Delegate to the `kernel-packager` agent with the target experiment(s), the competition constraints
from `.kaggle-gm/competition.json`, and the inference template
`${CLAUDE_PLUGIN_ROOT}/templates/inference_kernel.py`. Uploading datasets and pushing kernels are
outward-facing: confirm with the user before doing them unless already authorised.

Report artefacts, kernel version, measured runtime vs limit (with ≥20% margin), OOF-parity check,
and the submit command.
