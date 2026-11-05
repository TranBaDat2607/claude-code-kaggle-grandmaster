---
name: kg-debug
description: Diagnose a CV-LB disagreement, a suspicious score jump, or a failed/odd submission using the systematic CV-LB debugging procedure and the validation auditor.
argument-hint: "[symptom description or experiment id]"
allowed-tools:
  - Read
  - Glob
  - Grep
  - Agent
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit adv *)'
  - 'Bash(python -m kgkit adv *)'
  - 'PowerShell(python -m kgkit adv *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit validate *)'
  - 'Bash(python -m kgkit validate *)'
  - 'PowerShell(python -m kgkit validate *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit score *)'
  - 'Bash(python -m kgkit score *)'
  - 'PowerShell(python -m kgkit score *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit ledger *)'
  - 'Bash(python -m kgkit ledger *)'
  - 'PowerShell(python -m kgkit ledger *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit status *)'
  - 'Bash(python -m kgkit status *)'
  - 'PowerShell(python -m kgkit status *)'
---

# /kg-debug

Run kgkit with the Bash tool, as `PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit <command>`: this command
pre-approves that form for the kgkit calls it makes. Submitting, pushing or publishing still asks (so does a
PowerShell `$env:` assignment).

Symptom: `$ARGUMENTS`

Follow the `cv-lb-debugging` skill:
1. Classify the symptom with its table and list the likely causes in order.
2. Run the cheap checks yourself: submission validation and distribution comparison, re-scoring the
   inference path on a validation fold (parity with OOF), CV-LB correlation from the ledger, public
   LB noise estimate.
3. In parallel, launch the `validation-auditor` agent on the relevant code and folds.
4. If shift is suspected: `python -m kgkit adv` and drop/transform tests.
5. Conclude with the root cause (or ranked hypotheses with the experiment that would distinguish
   them), the fix, and how it changes trust in CV vs LB going forward. Record it in CLAUDE.md notes.
