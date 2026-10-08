---
name: kg-debug
description: Diagnose a CV-LB disagreement, a suspicious score jump, or a failed/odd submission using the systematic CV-LB debugging procedure and the validation auditor.
argument-hint: "[symptom description or experiment id]"
allowed-tools: [Bash, PowerShell, Read, Write, Edit, Glob, Grep, Agent]
---

# /kg-debug

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
