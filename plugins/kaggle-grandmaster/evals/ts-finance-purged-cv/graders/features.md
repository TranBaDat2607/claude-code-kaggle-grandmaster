---
type: llm
---

PASS when the response adapts the modelling to the per-date correlation metric: normalise features (and
possibly the target) cross-sectionally per date (rank or z-score within each date), and evaluate the metric per
date with its stability over time rather than one pooled number. Noting the very low signal-to-noise and that
large private-LB shake-ups are normal in market prediction is a plus.

FAIL if it suggests nothing about per-date (cross-sectional) normalisation or per-date evaluation.
