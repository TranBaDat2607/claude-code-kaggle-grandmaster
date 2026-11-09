---
type: regex
pattern: '(?:^|[\s"''/])(?:work|studies)/[\w./-]*\.py'
target: trace
arm: both
---

A Python file under work/ or studies/ appears in the session (written with Write, or with a shell command).
