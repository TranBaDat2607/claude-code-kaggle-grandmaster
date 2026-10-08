---
max_turns: 6
allowed_tools: [Read, Glob, Grep, Skill]
---

I have OOF predictions from 9 Kaggle models (LightGBM x3, CatBoost, XGBoost, two MLPs, a
TabM, and a kNN) for a binary AUC competition. Some were trained with different fold splits because I changed
the seed mid-competition. How should I build the final ensemble?
