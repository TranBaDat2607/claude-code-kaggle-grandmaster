---
type: llm
---

PASS when the response says to build the development story from what was recorded during the competition (an
experiment log or ledger with scores and decisions, commit history, submission history) rather than from memory,
and to keep the code of the experiments reachable instead of deleting it for good: for example links from each
experiment to the commit it ran from, a tag of the repository's state before tidying, or `git mv` into an archive
folder.

FAIL if it advises rewriting or squashing the history to hide the mess, or deleting the old scripts with no way to
reach the code the story refers to.
