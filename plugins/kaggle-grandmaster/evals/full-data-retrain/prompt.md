---
description: "Full-data refit for the final: fixed ~1.1-1.2x iterations, unverifiable on CV, keep the fold version"
tags: [portable, regression, boosters]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

My best model is a 5-fold LightGBM whose test predictions I average over the folds (CV 0.8432; early stopping
picked about 1,400 trees on average). For my final submission, should I retrain on 100% of the training data?
How exactly, and what are the risks?
