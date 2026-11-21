---
name: kg-ideas
description: Generate a ranked, evidence-backed backlog of next experiments from error analysis, the domain playbook's untried levers, public notebooks, discussion/prior-art research and the ledger history, stored with kgkit backlog.
argument-hint: "[focus area]"
allowed-tools:
  - Read
  - Glob
  - Grep
  - Agent
  - 'Bash(kgkit backlog *)'
  - 'Bash(kgkit discussions *)'
  - 'Bash(kgkit kernels top *)'
  - 'Bash(kgkit kernels pull *)'
  - 'Bash(kgkit ledger *)'
  - 'Bash(kgkit status *)'
---

# /kg-ideas

Run kgkit with the Bash tool as `kgkit <command>`: the plugin puts it on PATH, also on Windows (Git Bash),
and this command pre-approves the kgkit calls it makes. Older text that says `python -m kgkit` means the same
command. Submitting, pushing or publishing still asks.

Focus: `$ARGUMENTS`

1. Read what was already tried: `kgkit ledger list -n 50` and `kgkit backlog list --all`
   (done/dropped ideas with their results; don't re-propose a failed idea unless the evidence changed).
   Check `kgkit status` for a **PLATEAU** or **CROWD** line. Either one means the stack is saturated:
   the last experiments added ~0, or many teams sit within noise of your score. Then rank first an idea that
   brings what the stack, the public notebooks and the public datasets lack: a new model family, a new data view
   (a representation, another granularity, the data-generation process), or a new signal. Give more members of a
   family already in the stack gain 1, whatever their CV in isolation. Spend the GPU quota left before the
   deadline on that kind of idea first.
2. In parallel: `error-analyst` agent on the accepted baseline (`kgkit ledger baseline`). It needs
   per-row predictions with labels: OOF in a K-fold workspace; in a `holdout+lb` workspace (`kgkit status`
   shows the validation mode) give it the holdout predictions of each leg instead, and say how many rows
   there are. If the baseline has neither (an LB-only fork of public notebooks), skip it and note why;
   do not spawn it to fail;
   `solution-researcher` agent for techniques from similar past competitions (skip if
   `reports/prior-art.md` is recent); `kgkit kernels top --sort-by hotness` for newly popular
   notebooks, and `kgkit kernels pull` on any that look new (read their `REVIEW.md`); refresh the
   discussion index (`kgkit discussions sync`, then `discussions top --sort recent` and
   `discussions search "leak|trick|magic|cv.*lb|shake|external"`) and read the threads that matter
   (`discussions read <id>` or WebFetch the printed URL). Tag ideas from there `--source forum`.
3. Walk the relevant domain skill's lever list and mark which levers are untried. Then brainstorm
   beyond the lists (data-generation process, metric quirks, what the error slices suggest) and label
   those ideas `--source brainstorm`.
4. Add each idea: `kgkit backlog add "<idea>" --gain <1-5> --prob <0-1> --cost <hands-on hours>
   [--gpu-hours <unattended Kaggle GPU hours>] --evidence "<slice/metric numbers, thread, notebook>"
   --source <error-analysis|forum|notebook|prior-art|domain|brainstorm> --first-experiment "<smallest test>"`.
   Gain scale: 1 = within noise, 3 = solid step, 5 = leak or new data source. Mark cheap high-information
   experiments by giving them a low cost. Put background GPU time in `--gpu-hours`, not `--cost`: hours that
   fit in the quota left before the deadline rank as nearly free (unused quota expires), so a long unattended
   retrain is not buried behind small manual tweaks. Ideas already ruled out go in with
   `--status dropped --result "<why>"` so they are never re-proposed.
5. `kgkit backlog render` (writes `reports/backlog.md`) and present the top 10.
