---
type: tool_used
tool: Bash
input_match: 'kgkit exp finish 0*3\b'
min: 1
arm: with-only
---

The decision and the backlog result were recorded with `kgkit exp finish 0003 ...`, the one call that replaces
`ledger compare` + `ledger decide` + `backlog set` (#127).
