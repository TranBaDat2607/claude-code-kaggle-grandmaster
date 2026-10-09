---
name: kg-experiment
description: Run one disciplined experiment for a hypothesis — single change, frozen folds, same seeds, ledger logging with its parent baseline, paired comparison (per-fold deltas + OOF bootstrap) against the accepted baseline, recorded keep/discard decision.
argument-hint: <hypothesis or change to test>
allowed-tools: [Bash, PowerShell, Read, Write, Edit, Glob, Grep, Agent]
---

# /kg-experiment

Hypothesis: `$ARGUMENTS`

Follow the experiment protocol (`grandmaster-playbook` §2):

1. `python -m kgkit status` and `python -m kgkit ledger baseline` → the **accepted baseline** id
   (last KEEP, not the highest CV), its CV ± std and code path. If nothing is accepted yet, decide one
   first (`python -m kgkit ledger decide <id> baseline`). If the idea is in the backlog, mark it
   `python -m kgkit backlog set <n> --status running`; otherwise add it.
2. Restate the hypothesis, the *single* change, the baseline id, and the success criterion:
   paired gain ≥ ~2 standard errors of the per-fold differences (or of the OOF bootstrap) with most
   folds improving. Not "Δ > fold std": the fold std measures fold difficulty, which cancels in a
   paired comparison.
3. Implement behind a flag/config in the existing script. Smoke test, then full run on frozen folds
   with the baseline's seed(s). For long runs (> ~10 min) run in the background and keep working on
   analysis meanwhile; or delegate the whole run to the `experiment-runner` agent.
4. Log to the ledger with OOF/test preds and `parent=<baseline id>` (notes: hypothesis).
5. Decide with numbers:
   `python -m kgkit ledger compare <new> --truth data/train.csv:<target> --folds data/folds.csv:fold`
   (add `--groups data/train.csv:<group>` when rows are clustered). KEEP / DISCARD / INCONCLUSIVE:
   INCONCLUSIVE → second seed (or keep only if it adds no complexity/runtime); a flagged huge jump →
   `validation-auditor` before accepting.
6. Record it: `python -m kgkit ledger decide <new> keep|discard|inconclusive "<gain, folds k/K>"
   --parent <baseline>`; KEEP makes it the new baseline. `python -m kgkit backlog set <n> --status done
   --exp <new> --result "<decision + gain>"`. Add one line of learning to the CLAUDE.md notes. Commit code.
7. Suggest the next experiment based on what was learned.
