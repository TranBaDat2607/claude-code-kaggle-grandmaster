---
type: llm
---

PASS when the response covers:

1. Retraining on all the data usually helps a little, but it cannot be validated on CV (no held-out rows).
2. With no validation set there is no early stopping, so train a fixed number of trees, a bit more than the fold
   average because there is more data (about 1.1-1.2x, so roughly 1,500-1,700), with the same parameters.
3. Keep the fold-average version as one of the final picks (or blend the full-data model with the fold models)
   as a hedge.

Seed-averaging the full-data model, or checking that its test predictions correlate highly with the fold
version's, is a plus. FAIL if it claims the full-data model can be validated on CV, or leaves the number of
trees to early stopping on the training data.
