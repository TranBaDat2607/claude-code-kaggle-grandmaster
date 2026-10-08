---
name: kg-recon
description: Research a Kaggle competition (metric, data, rules, timeline, top notebooks, discussions, similar past competitions, leak and shake-up risk) and write reports/recon.md.
argument-hint: <competition-slug>
allowed-tools: [Bash, Read, Write, Edit, Glob, Grep, WebFetch, WebSearch, Agent]
---

# /kg-recon

Competition: `$ARGUMENTS`

Launch the `competition-analyst` agent and the `solution-researcher` agent **in parallel** for this
competition. When both return, merge their findings into `reports/recon.md` (template in the
`competition-recon` skill) and `reports/prior-art.md`, update `.kaggle-gm/competition.json` if the
workspace exists, and present to the user:

1. TL;DR — what wins here and the biggest risks.
2. Metric + the post-processing it implies (see `metric-optimization`).
3. Recommended validation scheme and why it mirrors the test split.
4. Shake-up risk (low/med/high) with reasoning.
5. Ranked initial backlog (top 8), each with expected gain and cost.
6. Anything the user must confirm or do (accept rules, external data declarations, deadlines).
