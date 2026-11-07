---
name: kg-grind
description: Autonomous improvement loop — repeatedly pick the highest-value idea from the backlog, run it as a disciplined experiment against the accepted baseline, decide with a paired comparison, update the backlog, and re-check the CV for overfitting, within a time or experiment budget.
argument-hint: "[budget, e.g. 4h or 10 experiments] [focus area]"
allowed-tools:
  - Read
  - Glob
  - Grep
  - Agent
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit backlog *)'
  - 'Bash(python -m kgkit backlog *)'
  - 'PowerShell(python -m kgkit backlog *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit kernels pull *)'
  - 'Bash(python -m kgkit kernels pull *)'
  - 'PowerShell(python -m kgkit kernels pull *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit ledger *)'
  - 'Bash(python -m kgkit ledger *)'
  - 'PowerShell(python -m kgkit ledger *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit status *)'
  - 'Bash(python -m kgkit status *)'
  - 'PowerShell(python -m kgkit status *)'
disable-model-invocation: true
---

# /kg-grind — autonomous grandmaster loop

Run kgkit with the Bash tool, as `PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit <command>`: this command
pre-approves that form for the kgkit calls it makes. Submitting, pushing or publishing still asks (so does a
PowerShell `$env:` assignment).

Budget / focus: `$ARGUMENTS` (default: 6 experiments, no specific focus).

If the target is a public rank or score ("public top 10", "LB 0.965"), say once that a gain fitted to the public LB
has zero expected private value. Grind toward the honest equivalent instead: a validation gain over the accepted
baseline beyond the paired noise, which `kgkit status` reports. The loop never fits or probes the public LB.

## Guardrails
- **Never submit to Kaggle** unless the user explicitly authorised autonomous submissions in this
  request; prepare and validate submission files instead.
- Never modify `data/` or the frozen folds; never delete ledger entries, backlog items or artefacts.
- Commit code before each long run; one hypothesis per experiment; everything logged and decided.
- Stop early and report if: a comparison is flagged as a suspicious jump (run `validation-auditor`
  first), repeated crashes, disk/GPU memory issues, or the budget is spent.

## Loop
1. **Orient**: `python -m kgkit status` (accepted baseline, decisions on these folds, backlog top),
   CLAUDE.md notes, `reports/` (recon, eda, error analyses).
2. **Backlog**: `python -m kgkit backlog list`. Refill it when fewer than ~5 open ideas by
   (a) dispatching `error-analyst` on the accepted baseline, (b) the domain skill's untried levers,
   (c) `solution-researcher` / `kgkit kernels pull` for prior art, (d) when runs are long or `gpu collect` reports
   an idle or input-bound GPU, a speed fix from `compute-efficiency`: a faster loop buys more experiments. Add each with
   `kgkit backlog add "<idea>" --gain --prob --cost --evidence --source`.
3. **Pick** the top idea by (gain × probability) / cost, favouring cheap high-information tests early
   and scaling runs late. Batch independent cheap experiments in parallel when hardware allows
   (e.g. two GPUs → two `experiment-runner` agents). Without a local GPU, run GPU experiments
   through `/kg-gpu` (1-fold screens, two per Kaggle session, with pruning). Pushing spends the
   user's weekly quota, so it needs their authorisation for this loop.
4. **Run** it via the `/kg-experiment` protocol (or delegate to `experiment-runner`): parent = accepted
   baseline, `kgkit ledger compare`, `kgkit ledger decide`. For a promoted screen, record both gains on
   the backlog item (`--screen-gain`, `--full-gain`).
5. **Update**: `kgkit backlog set <n> --status done --exp <id> --result ...`, CLAUDE.md notes (one line per
   experiment), commit. `kgkit backlog render` at the end of the loop.
6. Every ~3 kept improvements: refresh the ensemble (`ensemble-architect`) and prepare a candidate
   submission file with its honest CV.
7. When `status` warns about many decisions on the same folds, or every ~20 decisions: run
   `python -m kgkit recheck --seed <new> --run "<old pipeline>" --run "<current pipeline>" --exp <root_id>
   <baseline_id> --truth data/train.csv:<target>` (root from `kgkit ledger lineage`). If the gain
   shrank or did not survive, reverse-ablate the marginal KEEPs before adding new ideas
   (`grandmaster-playbook`, "Guarding against overfitting the CV itself"). If `kgkit backlog fidelity` shows
   screens disagree with full runs, raise screen fidelity before promoting more ideas.
8. Repeat until the budget is exhausted.

## Final report
Table of experiments (id, hypothesis, paired Δ, folds improved, decision), the new accepted baseline and
best blend (honest CV), any CV re-check results, submission files ready for the user's review, and the
refreshed top-5 backlog.
