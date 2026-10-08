---
name: kg-kernel
description: Package the current best solution for a Kaggle code competition — offline inference script, weight/code/wheel datasets, kernel metadata with internet off, runtime estimate on hidden-test size, push and poll.
argument-hint: "[experiment-id or blend id]"
allowed-tools: [Bash, PowerShell, Read, Write, Edit, Glob, Grep, Agent]
disable-model-invocation: true
---

# /kg-kernel

Target: `$ARGUMENTS` (default: the best blend/experiment in the ledger).

Delegate to the `kernel-packager` agent with the target experiment(s), the competition constraints
from `.kaggle-gm/competition.json`, and the inference template
`${CLAUDE_PLUGIN_ROOT}/templates/inference_kernel.py`. Uploading datasets and pushing kernels are
outward-facing: confirm with the user before doing them unless already authorised.

Report artefacts, kernel version, measured runtime vs limit (with ≥20% margin), OOF-parity check,
and the submit command.
