---
description: "A sealed vault: final picks need a vault check first, spent only on the real candidates"
tags: [portable, capability, final-selection]
max_turns: 10
allowed_tools: [Read, Glob, Grep, Skill]
---

Tabular competition, 2 days left, 2 finals. At the start I sealed a vault with `kgkit folds ... --vault 0.1`;
`kgkit status` says: "vault: sealed 10% (612 groups), 20 of 20 checks left; nothing passed yet". Since then I made
41 keep/discard decisions on the same 5 folds. Candidates:
- 0031_stack12: CV 0.89210, nested-validated stack, the accepted baseline of the ensemble line
- 0027_lgbm_te: CV 0.89150, the accepted baseline of the main line
- 0029_cat_v4: CV 0.89120, kept as an ensemble member
- 0034_stack12_s2: CV 0.89225, the stack again with another seed for the meta-model, not decided yet
I'd like to pick 0034 and 0031 and be done. Anything I should do before selecting them on the website?
