---
type: llm
weight: 1
---

The response identifies that the CV gain is inflated by leakage: the test pseudo-labels came from an ensemble
that includes models trained on each fold's validation rows, so when those pseudo-labels are added to fold k's
training data they carry information learned from fold k's validation labels (and near-duplicate / correlated
test images can amplify this). It prescribes a leak-safe protocol, e.g. for fold k use pseudo-labels produced
only by models that never trained on fold k (such as the fold-k model's own test predictions), keep validation
folds free of pseudo-labels, and judge the gain by LB or a clean holdout. Mentioning soft labels, lower weight
for pseudo-labelled samples, or multiple rounds is a plus. Fail if it attributes the gap only to public-LB noise
or generic overfitting without identifying the pseudo-label leakage mechanism.
