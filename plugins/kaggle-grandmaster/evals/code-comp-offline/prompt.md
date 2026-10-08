---
max_turns: 6
allowed_tools: [Read, Glob, Grep, Skill]
---

I have 5 fold checkpoints of a ConvNeXt model and a custom preprocessing package. The Kaggle
competition is a code competition: notebook submission, internet disabled, 9-hour GPU limit, and the hidden
test set is about 20x larger than the visible test sample. Walk me through packaging and making the inference
notebook robust.
