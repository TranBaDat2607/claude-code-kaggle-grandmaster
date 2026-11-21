---
description: "Finishing a logged experiment takes one bookkeeping call: the verdict was printed at log time, `kgkit exp finish` records the decision and closes the backlog idea (#127)"
tags: [shell, capability, workflow, experiments]
max_turns: 8
allowed_tools: [Bash, Read, Glob, Grep, Skill]
---

Experiment 0003 (adds feature f3) has finished and is in the ledger; it tests backlog idea #4. Wrap it up: decide it against the accepted baseline, record the decision and close the backlog idea.
