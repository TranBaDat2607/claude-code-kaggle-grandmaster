---
description: "bf16 autocast and torch.compile on a P100, NaN loss, 2xT4 plan: fp16 + GradScaler, one fold per GPU"
tags: [portable, regression, deep-learning]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

I'm fine-tuning a ConvNeXt-Small image classifier in a Kaggle notebook on a P100, using
torch.autocast(dtype=torch.bfloat16) and torch.compile. Training is much slower than I expected, and the loss
becomes NaN in epoch 2. I'm also thinking of switching to the 2x T4 machine with DDP to go faster. What should
I change?
