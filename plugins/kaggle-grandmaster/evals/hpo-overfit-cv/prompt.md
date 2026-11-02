---
description: "2,000 Optuna trials on 3,000 rows overfit the folds: confirm on re-drawn folds, don't search more"
tags: [portable, regression, tuning]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

Tabular Kaggle competition, 3,000 training rows, metric AUC. I ran 2,000 Optuna (TPE) trials on my 5-fold CV
for LightGBM. The best trial scores CV 0.8912 against 0.8871 for my hand-picked defaults, but on the public LB
the tuned model scores lower than the defaults. Should I run 5,000 more trials with a wider search space?
