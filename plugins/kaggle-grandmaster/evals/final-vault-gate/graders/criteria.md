---
type: llm
---

PASS when all three hold:

1. It says the picks should pass the vault before they are selected: run `kgkit vault check <id>` on each
   candidate (after the CV gate and, ideally, `kgkit recheck` on fresh folds), and pick only candidates that
   pass.
2. It treats the vault checks as scarce and pass/fail: spend them on the two or three real candidates, not on
   every record, and a FAIL drops a candidate instead of being re-asked with tweaks.
3. It does not take 0034 at face value as a pick: it is a seed re-run of the stack that has not been decided, and
   its 0.00015 edge over 0031 is within noise, so it is not a distinct second pick (a near-duplicate of 0031 buys
   no hedge).

A plus, not required: suggesting a hedge on another risk axis (such as 0027), mentioning `kgkit vault open` and a
retrain on all rows after the picks, or `kgkit final --accept-unvaulted` only as a recorded exception.

FAIL if it says the picks can be selected without a vault check, suggests reading or scoring the vault's labels
directly, or keeps 0034 and 0031 as the two finals.
