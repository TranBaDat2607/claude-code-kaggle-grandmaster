---
description: "Two-stage recsys: popularity-only candidates cap recall@20, so ranker tuning cannot help"
tags: [portable, regression, recsys]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

Session-based recommendation competition on Kaggle (OTTO-style): for each test session predict 20 items per
event type (clicks, carts, orders); the metric is recall@20. My pipeline takes the 50 most popular items of the
last week as candidates for every session, then a LightGBM ranker re-ranks them with session-item features.
Validation recall@20 is stuck at 0.31, and three rounds of ranker tuning changed nothing. What should I do?
