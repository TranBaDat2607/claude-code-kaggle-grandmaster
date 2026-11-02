---
type: llm
---

The file subs/sub_0003.csv is invalid: its header is `id,prediction` while data/sample_submission.csv has
`id,target`, and it has 499 rows where 500 are expected (id 2499 is missing).

PASS when the reply names at least one of these concrete problems (the wrong column name, or 499 vs 500 rows /
a missing id).

FAIL if it names no problem with the file, for example because it stopped before looking at it, or if it says
the file looks valid.
