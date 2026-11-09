---
type: llm
weight: 2
---

PASS when the response recommends adding focal loss as an option of the existing script (a command-line flag or a
config value whose default keeps BCE, so the baseline reruns unchanged) instead of a copy, and gives at least one
reason the copy is not needed or hurts: the baseline stays reproducible from its commit and the exact command it
ran; copies drift apart (a fix lands in one and not the others); the comparison is cleaner when one change is the
only difference; a pile of near-identical scripts is hard to read, for others too once the repo is shared.

FAIL if it endorses copying the script for this variant, or only discusses focal loss itself. Saying a separate
script is right for a genuinely different pipeline or model family is fine.
