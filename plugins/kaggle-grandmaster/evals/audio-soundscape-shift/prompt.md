---
description: "Bird-call model validated on focal clips but tested on soundscapes, plus a CPU-only inference limit"
tags: [portable, regression, audio]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

Kaggle bird-sound competition (BirdCLEF-style). Training data are short focal recordings, each labelled with one
primary species (some list secondary species). The test set is 4-minute soundscapes scored per 5-second window
with macro ROC AUC, and inference must run CPU-only within 90 minutes for about 700 soundscapes. My
EfficientNet-B0 on mel spectrograms scores 0.95 AUC on a random 20% split of the focal files, but the public LB
is 0.71, and my notebook times out after about 80% of the files. What's wrong, and how do I fix both problems?
