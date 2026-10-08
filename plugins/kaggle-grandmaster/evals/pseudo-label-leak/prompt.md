---
max_turns: 6
allowed_tools: [Read, Glob, Grep, Skill]
---

Kaggle image classification, metric accuracy, 5-fold CV. I trained a 5-fold ensemble (CV 0.912), predicted the
test set with the full ensemble (average of all 5 fold models), kept test predictions with confidence > 0.9 as
pseudo-labels, and added the same pseudo-labelled set to the training data of every fold. Retrained: CV jumped
to 0.931, but the public LB only improved from 0.909 to 0.911. What's happening, and how should I do
pseudo-labelling properly?
