---
max_turns: 6
allowed_tools: [Read, Glob, Grep, Skill]
---

My Kaggle model has 5-fold CV RMSE 0.21 but public LB RMSE 0.34. The data are daily store sales;
train covers 2021-01 to 2023-06 and the test period is 2023-07 to 2023-09. I used KFold(shuffle=True) and
features like 7-day and 28-day rolling mean of sales, and a target-mean encoding of store_id computed on
the full training set. What is going wrong and how do I fix it? Give a prioritised plan.
