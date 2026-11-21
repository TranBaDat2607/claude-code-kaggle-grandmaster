---
name: kg-debug
description: "Diagnoses a CV-LB gap, a suspicious score jump or a failed submission in the workspace. Use when a result looks wrong."
argument-hint: "[symptom description or experiment id]"
allowed-tools:
  - Read
  - Glob
  - Grep
  - Agent
  - 'Bash(kgkit adv *)'
  - 'Bash(kgkit validate *)'
  - 'Bash(kgkit score *)'
  - 'Bash(kgkit ledger *)'
  - 'Bash(kgkit status *)'
---

# /kg-debug

Run kgkit with the Bash tool as `kgkit <command>`: the plugin puts it on PATH, also on Windows (Git Bash),
and this command pre-approves the kgkit calls it makes. Older text that says `python -m kgkit` means the same
command. Submitting, pushing or publishing still asks.

Symptom: `$ARGUMENTS`

Follow the `cv-lb-debugging` skill:
1. Classify the symptom with its table and list the likely causes in order.
2. Run the cheap checks yourself: submission validation and distribution comparison, re-scoring the
   inference path on a validation fold (parity with OOF), CV-LB correlation from the ledger, public
   LB noise estimate.
3. In parallel, launch the `validation-auditor` agent on the relevant code and folds.
4. If shift is suspected: `kgkit adv` and drop/transform tests.
5. Conclude with the root cause (or ranked hypotheses with the experiment that would distinguish
   them), the fix, and how it changes trust in CV vs LB going forward. Record it in `reports/journal.md`
   (or a `studies/<name>/README.md` when a script decided it), and one line in CLAUDE.md `## Current state`
   when it changes how CV or the LB is trusted from now on.
