---
type: tool_used
tool: Skill
input_match: '"skill"\s*:\s*"(?:kaggle-grandmaster:)?(?:kg-postmortem|competition-strategy)"'
---

Plugin-fired indicator: Claude loaded kg-postmortem (the router sends "the competition ended" there; its last step
points to `/kg-showcase`, which is a slash command Claude cannot load) or competition-strategy. Not scored in a
with/without run.
