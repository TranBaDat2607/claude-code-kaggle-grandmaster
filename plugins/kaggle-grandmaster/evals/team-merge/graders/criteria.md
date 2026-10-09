---
type: llm
weight: 1
---

The response warns that exchanging OOF/test prediction files (or code/data) with someone who is not yet a
teammate is private sharing outside a team, which Kaggle rules prohibit and which can lead to
disqualification, so they should judge the merge from public signals (LB, model type, likely low
correlation of transformers vs GBDTs) and share predictions only after merging. It also flags at least one
of: the merged team's combined submission count must not exceed the allowance at merge time (they have used
most of their submissions), or the merger deadline. For the blend after merging, it explains that OOFs made on
different fold splits make blend weights / stacking optimistic, and recommends agreeing on one fold split
(re-training cheaper models on it) or keeping mismatched models to simple averages. Fail if it tells them to
swap OOF files before merging without mentioning the private-sharing rule.
