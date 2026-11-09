---
type: llm
weight: 2
---

PASS when the response proposes two separate, linked ways into the repository:

1. one for readers who want the result: the final pipeline with the exact steps or commands that reproduce the
   final submission (environment, data setup, train, predict), kept apart from the experiments;
2. one for readers who want to learn: the development told in order (by phase, milestone or question), with what
   was tried, the numbers, what was kept and what was dropped and why (dead ends included);

and a README (or landing page) that sends each kind of reader to theirs.

FAIL if it proposes only the final solution, only a dump of every script and log with no guided path, or one
merged document that serves neither reader.
