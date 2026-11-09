---
description: "Mid-competition, the user wants to copy train.py to train_focal.py for a variant; a flag of the same script keeps the record and the repo readable"
tags: [portable, capability, experiments]
max_turns: 8
allowed_tools: [Read, Glob, Grep, Skill]
---

Next Kaggle experiment: focal loss instead of BCE. My plan is to copy src/train.py to src/train_focal.py and
change the loss in the copy, so the baseline script stays untouched and I can always rerun it exactly. That's how I
did the last few (train_v2.py, train_v3.py, ...). Good plan?
