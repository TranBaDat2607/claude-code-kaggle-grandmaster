---
description: "Overlapping 20-day return targets with shuffled KFold: purged/embargoed time CV, per-date normalisation"
tags: [portable, regression, time-series]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

Kaggle market-prediction competition: each row is (date, stock), the target is the stock's return over the next
20 trading days, and the metric is the mean over dates of the Spearman correlation between my predictions and
the targets across stocks. I use KFold(n_splits=5, shuffle=True) over rows and get CV 0.09, but the public LB
is 0.01. My features are raw price ratios and volumes. What is going on, and how should I validate and build
features?
