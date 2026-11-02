---
type: llm
---

PASS when the response explains why this is unreliable and gives a better process:

1. A rating after a few hours rests on few games: new agents start from a default rating and settle only over
   many episodes, so a drop from 1100 to 950 in 3 hours says little about strength.
2. Decide between versions with a local arena: play many games (hundreds) between versions and against public
   or earlier bots, and compare win rates with enough games for statistical significance.

Also good: submit improved versions early so ratings converge; keep strong (and diverse) agents active; check
the rules on how many agents stay active, since a new upload can retire an older active one; make the bot
crash-free and within the per-turn time limit. FAIL if it endorses judging versions by the few-hour rating change.
