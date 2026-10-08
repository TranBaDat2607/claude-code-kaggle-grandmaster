---
name: kg-experiment
description: Run one disciplined experiment for a hypothesis — single change, frozen folds, same seeds, ledger logging, delta vs current best, keep/discard decision.
argument-hint: <hypothesis or change to test>
allowed-tools: [Bash, PowerShell, Read, Write, Edit, Glob, Grep, Agent]
---

# /kg-experiment

Hypothesis: `$ARGUMENTS`

Follow the experiment protocol (`grandmaster-playbook` §2):

1. `python -m kgkit status` → current best experiment id, its CV ± std, and code path.
2. Restate the hypothesis, the *single* change, the baseline id, and the success criterion
   (Δ > fold std, or majority of folds improved; ≥ 2 seeds if marginal).
3. Implement behind a flag/config in the existing script. Smoke test, then full run on frozen folds.
   For long runs (> ~10 min) run in the background and keep working on analysis meanwhile; or
   delegate the whole run to the `experiment-runner` agent.
4. Log to the ledger (notes: hypothesis + Δ vs baseline) with OOF/test preds.
5. Decide KEEP / DISCARD / INCONCLUSIVE; if KEEP, it becomes the new baseline. Add one line of
   learning to the CLAUDE.md notes. Commit code.
6. Suggest the next experiment based on what was learned.
