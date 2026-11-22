---
name: kg-kernel
description: "Packages the best solution as an offline Kaggle code-competition kernel, then pushes and polls it. Use when a code competition needs its submission notebook."
argument-hint: "[experiment-id or blend id]"
allowed-tools:
  - Read
  - Glob
  - Grep
  - Agent
  - 'Bash(kgkit validate *)'
  - 'Bash(kgkit ledger *)'
  - 'Bash(kgkit status *)'
  - 'Bash(kaggle kernels status *)'
  - 'PowerShell(kaggle kernels status *)'
disable-model-invocation: true
---

# /kg-kernel

Run kgkit with the Bash tool as `kgkit <command>`: the plugin puts it on PATH, also on Windows (Git Bash),
and this command pre-approves the kgkit calls it makes. Older text that says `python -m kgkit` means the same
command. Submitting, pushing or publishing still asks.

Target: `$ARGUMENTS` (default: the best blend/experiment in the ledger).

1. **Delegate** to the `kernel-packager` agent with the target experiment(s), the competition constraints
   from `.kaggle-gm/competition.json`, the inference template
   `${CLAUDE_PLUGIN_ROOT}/templates/inference_kernel.py`, and the inference speed checklist
   `${CLAUDE_PLUGIN_ROOT}/skills/compute-efficiency/references/inference.md` (fp16, both T4s, shared
   preprocessing, exports with OOF parity, LLM engines and quantisation).
2. **Confirm before anything outward-facing**: uploading datasets and pushing kernels need the user's yes unless
   already authorised.
3. **Check** the measured runtime against the limit with ≥20% margin and the OOF-parity check; if either fails,
   fix and re-run before reporting.
4. **Report** artefacts, kernel version, runtime vs limit, OOF parity, and the submit command.
