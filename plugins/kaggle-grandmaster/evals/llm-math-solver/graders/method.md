---
type: llm
---

PASS when the response improves the method in a way that can be trusted:

1. It warns against tuning on the 10 public examples alone and recommends a larger local evaluation set that
   mirrors the hidden problems' difficulty.
2. It recommends sampling several solutions per problem and majority voting over normalised integer answers
   (self-consistency), and/or program-aided solving (generate Python, execute it, use the result).

FAIL if the plan is only more prompt tweaking judged on the 10 public examples.
