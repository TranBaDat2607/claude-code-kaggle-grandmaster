---
type: llm
weight: 2
---

The workspace's CLAUDE.md says one-off probes go in `work/` (ignored by git), and an analysis that decides
something goes in `studies/<name>/`.

PASS when the reply shows the script was saved under `work/` or `studies/` (for example `work/f1_check.py`).

FAIL if it was saved in `src/`, at the top of the workspace, or anywhere else; or if no script was written.
