---
type: llm
---

PASS when the response gives a correct Redis design with commands: ZADD/ZINCRBY to record scores, ZREVRANGE (or
ZRANGE ... REV) WITH SCORES for the top 100, ZREVRANK (plus 1) for a player's own rank, a tie-break that favours
the earlier time (for example a composite score combining the points with an inverted timestamp, or a secondary
structure), and a weekly reset (a per-week key such as leaderboard:2026-W45 with EXPIRE, or renaming/deleting
the key on Monday).

FAIL if it brings in Kaggle or machine-learning competition concepts, or the Redis commands are wrong.
