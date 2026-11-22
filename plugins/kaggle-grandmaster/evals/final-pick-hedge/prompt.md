---
description: "Two finals: a near-twin of the best CV adds no hedge; measure the pair by its expected best score"
tags: [portable, capability, final-selection]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

Tabular binary classification, AUC, 2 finals, deadline tomorrow. Every candidate below passed the CV gate and
`kgkit vault check`; none is public-tuned. Private LB is about 6,000 rows.
- 0042_stack_v7: CV 0.91240, the accepted baseline of the ensemble line
- 0044_stack_v7b: CV 0.91236, the same stack with one more GBDT member; OOF correlation with 0042 is 0.998
- 0038_nn_mlp: CV 0.91090, the accepted baseline of the NN line; OOF correlation with 0042 is 0.91
- 0035_lgbm: CV 0.90980, the main line's baseline, already inside both stacks
I plan to select the two best by CV, 0042 and 0044. Is that the best pair?
