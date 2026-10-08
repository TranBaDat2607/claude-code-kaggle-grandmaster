---
name: kg-grind
description: Autonomous improvement loop — repeatedly pick the highest-value idea from the backlog, run it as a disciplined experiment, log, keep or discard, and refresh the backlog, within a time or experiment budget.
argument-hint: "[budget, e.g. 4h or 10 experiments] [focus area]"
allowed-tools: [Bash, PowerShell, Read, Write, Edit, Glob, Grep, Agent, WebSearch, WebFetch]
disable-model-invocation: true
---

# /kg-grind — autonomous grandmaster loop

Budget / focus: `$ARGUMENTS` (default: 6 experiments, no specific focus).

## Guardrails
- **Never submit to Kaggle** unless the user explicitly authorised autonomous submissions in this
  request; prepare and validate submission files instead.
- Never modify `data/` or the frozen folds; never delete ledger entries or artefacts.
- Commit code before each long run; one hypothesis per experiment; everything logged.
- Stop early and report if: CV jumps suspiciously (run `validation-auditor` first), repeated crashes,
  disk/GPU memory issues, or the budget is spent.

## Loop
1. **Orient**: `python -m kgkit status`, read CLAUDE.md notes, `reports/` (recon, eda, error analyses).
2. **Backlog**: maintain `reports/backlog.md` — ideas with (expected gain, probability, cost,
   status). Refill it when thin by (a) dispatching `error-analyst` on the best model, (b) consulting
   the domain skill for untried levers, (c) `solution-researcher` for prior art.
3. **Pick** the top idea by (gain × probability) / cost, favouring cheap high-information tests early
   and scaling runs late. Batch independent cheap experiments in parallel when hardware allows
   (e.g. two GPUs → two `experiment-runner` agents).
4. **Run** it via the `/kg-experiment` protocol (or delegate to `experiment-runner`).
5. **Update**: backlog status, CLAUDE.md notes (one line per experiment), commit.
6. Every ~3 kept improvements: refresh the ensemble (`ensemble-architect`) and prepare a candidate
   submission file with its honest CV.
7. Repeat until the budget is exhausted.

## Final report
Table of experiments (id, hypothesis, Δ CV, decision), new best single model and best blend (honest
CV), submission files ready for the user's review, and the refreshed top-5 backlog.
