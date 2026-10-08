---
type: llm
weight: 1
---

The response recommends (or strongly considers) training a regression head on the ordinal score (or using
expected value of class probabilities) and then optimising the rounding thresholds/cut-points on
out-of-fold predictions to maximise QWK (e.g. an OptimizedRounder / Nelder-Mead / coordinate search),
rather than plain argmax or naive rounding. It warns that thresholds fit on the same OOF overstate the score
and suggests an honest check (nested / per-fold fitting, or fitting on OOF and applying to test only).
Fail if it only suggests softmax argmax with no threshold optimisation.
