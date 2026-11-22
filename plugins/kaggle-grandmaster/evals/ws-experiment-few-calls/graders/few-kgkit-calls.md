---
type: tool_used
tool: Bash
input_match: '\bkgkit\b'
min: 1
max: 3
arm: with-only
---

Turns per experiment (#127): the state arrived with the command and the comparison was made when 0003 was logged,
so finishing it takes at most three kgkit calls (typically one `kgkit exp finish`), not the old eight to ten.
