---
max_turns: 6
allowed_tools: [Read, Glob, Grep, Skill]
---

Kaggle competition ends tomorrow, 2 final submissions allowed. The public LB uses ~1,500 test rows
(binary target, metric AUC); private uses ~8,500 rows. My candidates:
A) blend of 6 diverse models: CV 0.8912 (honest nested), public 0.8870
B) same blend with weights hand-tuned on the public LB: CV 0.8890, public 0.8935
C) single LightGBM: CV 0.8841, public 0.8901
D) blend A plus pseudo-labelled retrain: CV 0.8925, public 0.8862
Which two should I select and why?
