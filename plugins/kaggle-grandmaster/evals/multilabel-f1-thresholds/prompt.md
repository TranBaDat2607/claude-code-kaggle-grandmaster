---
description: "Micro-F1 multilabel: OOF-tuned thresholds, never an empty row, honest nested check"
tags: [portable, capability, metric]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

Multilabel text classification on Kaggle: 20 labels, and the metric is micro-F1 over all (row, label) pairs. I
use 0.5 as the threshold for every label, and about 8% of rows end up with no predicted label, although every
row has at least one true label. How should I turn the probabilities into labels to maximise the score without
fooling myself?
