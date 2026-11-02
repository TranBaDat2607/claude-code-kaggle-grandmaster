---
description: "28-day forecast with lag_1/rolling features computed from actual sales: horizon leakage and strategy"
tags: [portable, regression, time-series]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

Kaggle forecasting competition: predict daily unit sales for the next 28 days for about 30,000 store-item
series (many days have zero sales). My LightGBM uses lag_1, lag_2, lag_7 and a 7-day rolling mean of sales.
I validate on the last 28 days of train, and I computed all those features on the full table before splitting,
from the actual sales. Validation RMSE looks great. How do I produce the 28-day forecast, and is anything wrong
with my setup?
