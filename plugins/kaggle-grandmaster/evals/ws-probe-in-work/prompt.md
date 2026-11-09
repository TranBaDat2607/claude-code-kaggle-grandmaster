---
description: "A quick data check in a workspace: the throwaway script goes in work/ (ignored), not next to the pipeline in src/ or at the top"
tags: [shell, capability, workflow, layout]
max_turns: 12
allowed_tools: [Bash, Read, Write, Glob, Grep, Skill]
---

Quick check before the next experiment: what fraction of rows in data/train.csv have f1 > 1, split by target? Put
the code in a small script so I can rerun it later.
