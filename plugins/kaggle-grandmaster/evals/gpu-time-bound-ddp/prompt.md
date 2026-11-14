---
description: "Quota to spare across six accounts, 11 h runs, deadline in 4 days: weigh time to result, offer DDP for screens"
tags: [portable, capability, gpu]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

I'm LoRA fine-tuning a 4B LLM on Kaggle's 2x T4. One training run takes about 11 hours on one T4 (peak 13 of
15 GB at micro-batch 1, gradient accumulation 4). So far every 12-hour session trains two copies of the same
recipe, one per GPU, and I average them. I have six Kaggle accounts now, about 180 GPU hours a week in total,
and most of this week's hours are still unused; the competition ends in 4 days. I still have five ideas to test
(a different LoRA rank, a longer max length, two data mixes, a loss change). How should I schedule the next runs?
