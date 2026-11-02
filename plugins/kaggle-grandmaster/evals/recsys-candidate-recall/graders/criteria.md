---
type: llm
---

PASS when the response sees that the candidate set caps the score:

1. It recommends measuring candidate recall (recall@50 of the candidate set, per event type) as the upper bound
   the ranker can reach, and explains that a popularity-only list is the same for every session and misses most
   relevant items.
2. It recommends better candidate generation from several sources: the session's own history (recently clicked
   or carted items, repeats), co-visitation matrices (items seen together, time-weighted, separate for clicks
   and for carts/orders), and optionally embeddings (item2vec/ALS), tracking each source's recall contribution.

Ranker features built from those sources (co-visitation scores, recency, rank within each source) and a
validation that reproduces the test construction (later time window, truncated sessions) are a plus. FAIL if it
focuses on ranker hyperparameters or a different ranker without measuring or improving candidate recall.
