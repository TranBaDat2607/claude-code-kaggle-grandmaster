---
type: llm
---

PASS when the response replaces the plan with a recon-first start, including at least four of these:

- Read the metric, rules (external data, code-competition limits), timeline and merger deadline, and the
  discussion threads (data issues, leaks, CV-vs-LB reports).
- Build and freeze your own CV folds first.
- Judge public notebooks by Kaggle's recorded public score, not the score claimed in a title, and re-run the
  best ones on your own folds before believing their numbers.
- Don't just fork the most-forked notebook: its score is saturated and its errors are shared by many teams;
  add something it lacks (a different model family, a better CV, error analysis).
- Don't spend submissions tuning against the public LB; decide on CV.
- Consider a team merge before the merger deadline.

FAIL if it endorses forking the notebook and tuning it against the public LB as the plan.
