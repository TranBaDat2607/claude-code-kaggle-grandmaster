---
name: kg-ideas
description: Generate a ranked, evidence-backed backlog of next experiments from error analysis, the domain playbook's untried levers, discussion/prior-art research and the ledger history.
argument-hint: "[focus area]"
allowed-tools: [Bash, PowerShell, Read, Write, Edit, Glob, Grep, Agent, WebSearch, WebFetch]
---

# /kg-ideas

Focus: `$ARGUMENTS`

1. Read the ledger (`python -m kgkit ledger list -n 50`) to know what was already tried (and failed).
2. In parallel: `error-analyst` agent on the best experiment; `solution-researcher` agent for
   techniques from similar past competitions (skip if `reports/prior-art.md` is recent); check the
   competition discussion for new findings (WebFetch, sort by recent/votes).
3. Walk the relevant domain skill's lever list and mark which levers are untried.
4. Merge into `reports/backlog.md`: idea, evidence, expected gain, probability, cost, first
   experiment; ranked by (gain × probability) / cost. Mark cheap high-information experiments.
5. Present the top 10.
