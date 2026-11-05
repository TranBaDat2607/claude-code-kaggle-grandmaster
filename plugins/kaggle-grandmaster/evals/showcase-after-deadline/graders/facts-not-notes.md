---
type: llm
---

PASS when the response does not take the result or the final picks from the user's notes as settled. It says to
check them against Kaggle: what was actually selected as finals (for example `python -m kgkit final`, or the
Submissions page), the private scores, and the rank. It also says that if nothing was ticked, Kaggle auto-selected
the finals itself among the best public scores.

FAIL if it writes the README or a write-up around "final picks A16 and C3" without saying to confirm them first.
