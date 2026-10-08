---
max_turns: 6
allowed_tools: [Read, Glob, Grep, Skill]
---

I'm in a Kaggle competition predicting a disease label from chest X-rays. train.csv has columns
image_id, patient_id, view, label (binary, 8% positive). Most patients have 3-10 images, and the test
set contains different patients than train. Metric is ROC AUC. How exactly should I set up
cross-validation, and what mistakes should I avoid? Be concrete.
