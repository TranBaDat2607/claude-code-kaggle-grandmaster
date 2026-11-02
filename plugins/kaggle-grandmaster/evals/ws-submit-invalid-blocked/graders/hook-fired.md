---
type: regex
pattern: 'kaggle-grandmaster blocked an invalid submission'
target: trace
arm: with-only
---

Plugin-fired indicator: the PreToolUse hook denied the submit. Not scored: catching the problem before ever calling the submit is just as good.
