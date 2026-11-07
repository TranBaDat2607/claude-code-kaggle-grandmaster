---
description: "50M-row tabular competition: slow pandas groupbys and LightGBM 2h per fold on CPU: cuDF on a T4 (not P100), GPU XGBoost/CatBoost, both GPUs, measured"
tags: [portable, capability, compute]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

In a Kaggle competition I have 50 million rows and 60 columns. My pandas feature code (lots of groupby aggregations
and merges) takes 40 minutes, and LightGBM takes about 2 hours per fold in the Kaggle CPU notebook. I still have my
weekly GPU quota (T4 x2 or P100). What should I change to iterate faster?
