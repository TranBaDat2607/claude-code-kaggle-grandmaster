---
name: kg-ideas
description: Generate a ranked, evidence-backed backlog of next experiments from error analysis, the domain playbook's untried levers, public notebooks, discussion/prior-art research and the ledger history, stored with kgkit backlog.
argument-hint: "[focus area]"
allowed-tools: [Bash, PowerShell, Read, Write, Edit, Glob, Grep, Agent, WebSearch, WebFetch]
---

# /kg-ideas

Focus: `$ARGUMENTS`

1. Read what was already tried: `python -m kgkit ledger list -n 50` and `python -m kgkit backlog list --all`
   (done/dropped ideas with their results; don't re-propose a failed idea unless the evidence changed).
2. In parallel: `error-analyst` agent on the accepted baseline (`kgkit ledger baseline`);
   `solution-researcher` agent for techniques from similar past competitions (skip if
   `reports/prior-art.md` is recent); `python -m kgkit kernels top --sort-by hotness` for newly popular
   notebooks, and `kgkit kernels pull` on any that look new (read their `REVIEW.md`); refresh the
   discussion index (`python -m kgkit discussions sync`, then `discussions top --sort recent` and
   `discussions search "leak|trick|magic|cv.*lb|shake|external"`) and read the threads that matter
   (`discussions read <id>` or WebFetch the printed URL). Tag ideas from there `--source forum`.
3. Walk the relevant domain skill's lever list and mark which levers are untried. Then brainstorm
   beyond the lists (data-generation process, metric quirks, what the error slices suggest) and label
   those ideas `--source brainstorm`.
4. Add each idea: `python -m kgkit backlog add "<idea>" --gain <1-5> --prob <0-1> --cost <hours>
   --evidence "<slice/metric numbers, thread, notebook>" --source <error-analysis|forum|notebook|prior-art|domain|brainstorm>
   --first-experiment "<smallest test>"`. Gain scale: 1 = within noise, 3 = solid step, 5 = leak or new data
   source. Mark cheap high-information experiments by giving them a low cost.
5. `python -m kgkit backlog render` (writes `reports/backlog.md`) and present the top 10.
