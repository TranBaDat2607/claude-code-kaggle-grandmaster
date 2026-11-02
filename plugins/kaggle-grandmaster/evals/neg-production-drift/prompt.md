---
description: "Should not trigger: monitoring a production model at work"
tags: [portable, regression, negative]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

At work we run an XGBoost fraud-detection model in production. How should we monitor data drift (PSI or
similar), set alert thresholds, and decide when to retrain?
