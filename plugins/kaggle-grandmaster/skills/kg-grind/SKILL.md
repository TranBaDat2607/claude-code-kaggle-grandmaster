---
name: kg-grind
description: Autonomous improvement loop — repeatedly pick the highest-value idea from the backlog, run it as a disciplined experiment against the accepted baseline, decide with a paired comparison, update the backlog, and re-check the CV for overfitting, within a time or experiment budget.
argument-hint: "[resume | budget, e.g. 4h or 10 experiments] [focus area]"
allowed-tools:
  - Read
  - Glob
  - Grep
  - Agent
  - 'Bash(kgkit backlog *)'
  - 'Bash(kgkit exp *)'
  - 'Bash(kgkit grind *)'
  - 'Bash(kgkit kernels pull *)'
  - 'Bash(kgkit ledger *)'
  - 'Bash(kgkit status *)'
disable-model-invocation: true
---

# /kg-grind — autonomous grandmaster loop

Run kgkit with the Bash tool as `kgkit <command>`: the plugin puts it on PATH, also on Windows (Git Bash),
and this command pre-approves the kgkit calls it makes. Older text that says `python -m kgkit` means the same
command. Submitting, pushing or publishing still asks.

Budget / focus: `$ARGUMENTS` (default: 6 experiments, no specific focus; `resume` continues the grind in progress).

If the target is a public rank or score ("public top 10", "LB 0.965"), say once that a gain fitted to the public LB
has zero expected private value. Grind toward the honest equivalent instead: a validation gain over the accepted
baseline beyond the paired noise, which `kgkit status` reports. The loop never fits or probes the public LB.

## Before the loop
The grind's budget, progress and the runs it started live in `.kaggle-gm/grind.json` (`kgkit grind`), not in this
conversation: a compaction or a new session continues it from there.
- **`resume`**: `kgkit grind start --resume --session ${CLAUDE_SESSION_ID}`, then `kgkit grind status --session
  ${CLAUDE_SESSION_ID} --kaggle`. Act on every run it lists before picking new ideas: wait for a running one, re-run
  a stopped one with the folds it names (a fold with a checkpoint resumes mid-schedule), collect a finished Kaggle
  run. It refuses while another session drove the grind in the last 30 min (`--takeover` only if the user says that
  session is gone).
- **A new grind**: `kgkit grind start --session ${CLAUDE_SESSION_ID} --budget <4h | 10 | 4h,10> [--focus "<area>"]`.
  It refuses while a grind is in progress: offer to resume it, or to end it with `kgkit grind stop`.

Ask everything the budget needs in one message, before the first experiment: whether remote GPU runs are authorised and for how many hours, and anything the focus leaves open. Then the loop runs without waiting on the user. If `kgkit status` prints a **Permissions** line, offer `kgkit permissions --write` first: otherwise every kgkit call after the user's reply, and in each `experiment-runner`, asks.

## Guardrails
- **Never submit to Kaggle** unless the user explicitly authorised autonomous submissions in this
  request; prepare and validate submission files instead.
- Never modify `data/` or the frozen folds; never delete ledger entries, backlog items or artefacts.
- Commit code at natural points (each record keeps the exact code it ran from either way); one hypothesis per
  experiment, behind a flag of the existing script, never a copy of it; everything logged and decided (a sweep at
  once: `kgkit ledger decide 12..20 discard "<why>"`); probes and logs in `work/`.
- Stop early and report if: a comparison is flagged as a suspicious jump (run `validation-auditor`
  first), repeated crashes, disk/GPU memory issues, or the budget is spent.

## Loop
1. **Orient**: `kgkit grind status --session ${CLAUDE_SESSION_ID}` (budget left, runs in flight; on BUDGET SPENT go
   to the final report), `kgkit status` (accepted baseline, decisions on these folds, backlog top),
   CLAUDE.md `## Current state`, `reports/` (recon, eda, error analyses, `journal.md`).
2. **Backlog**: `kgkit backlog list`. Refill it when fewer than ~5 open ideas by
   (a) dispatching `error-analyst` on the accepted baseline, (b) the domain skill's untried levers,
   (c) `solution-researcher` / `kgkit kernels pull` for prior art, (d) when runs are long or `gpu collect` reports
   an idle or input-bound GPU, a speed fix from `compute-efficiency`: a faster loop buys more experiments. Add each with
   `kgkit backlog add "<idea>" --gain --prob --cost --evidence --source`.
3. **Pick** the top idea by (gain × probability) / cost, favouring cheap high-information tests early
   and scaling runs late. When it starts a new direction (a model family, new data, ensembling), name the phase
   first: `kgkit ledger phase "<name>"`. Batch independent cheap experiments in parallel when hardware allows
   (e.g. two GPUs → two `experiment-runner` agents). Without a local GPU, run GPU experiments
   through `/kg-gpu` (1-fold screens, two per Kaggle session, with pruning). Pushing spends the
   user's weekly quota, so it needs their authorisation for this loop.
4. **Run** it via the `/kg-experiment` protocol (or delegate to `experiment-runner`). Before starting each training
   run: `kgkit grind note --session ${CLAUDE_SESSION_ID} --started <run name> --cmd "<command>"`, so a later session
   finds it. A local fold of more than ~30 min: add the template's `--checkpoint-every 1` (or every few epochs), so
   a killed run resumes mid-fold. Parent = accepted
   baseline (`--parent`): the template prints the paired verdict against it when it logs, so there is no
   separate comparison call. For a promoted screen, record both gains on the backlog item (`--screen-gain`,
   `--full-gain`).
5. **Update**, in one call: `kgkit exp finish <id> keep|discard|inconclusive "<gain, folds k/K>" --backlog <n>`
   records the decision and closes the idea with its result (the learning goes there and in the decision's
   "why"; inconclusive keeps the idea running for a second seed), then commit. Rewrite CLAUDE.md `## Current state` only when it changed; never one line
   per experiment (it loads into every session). At the end of the loop: `kgkit backlog render`, and a dated
   entry in `reports/journal.md` for what the loop tried and learned.
6. Every ~3 kept improvements: refresh the ensemble (`ensemble-architect`) and prepare a candidate
   submission file with its honest CV.
7. When `status` warns about many decisions on the same folds, or every ~20 decisions: run
   `kgkit recheck --seed <new> --run "<old pipeline>" --run "<current pipeline>" --exp <root_id>
   <baseline_id> --truth data/train.csv:<target>` (root from `kgkit ledger lineage`). If the gain
   shrank or did not survive, reverse-ablate the marginal KEEPs before adding new ideas
   (`grandmaster-playbook`, "Guarding against overfitting the CV itself"). If `kgkit backlog fidelity` shows
   screens disagree with full runs, raise screen fidelity before promoting more ideas.
8. Repeat until the budget is exhausted, then `kgkit grind stop --session ${CLAUDE_SESSION_ID} --reason "<budget spent | why it stopped>"`.

## Final report
Table of experiments (id, hypothesis, paired Δ, folds improved, decision), the new accepted baseline and
best blend (honest CV), any CV re-check results, submission files ready for the user's review, and the
refreshed top-5 backlog.
