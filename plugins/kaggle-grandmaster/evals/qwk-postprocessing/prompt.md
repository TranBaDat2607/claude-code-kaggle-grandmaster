---
max_turns: 6
allowed_tools: [Read, Glob, Grep, Skill]
---

Kaggle competition: predict an essay score that is an integer from 1 to 6; the metric is quadratic
weighted kappa (QWK). I'm fine-tuning DeBERTa-v3. Should I treat it as 6-class classification? How do I
turn model outputs into the final integer predictions to maximise QWK, and how do I avoid fooling myself?
