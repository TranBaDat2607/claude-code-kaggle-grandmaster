---
type: llm
---

PASS when the response diagnoses the validation mismatch and fixes it:

- A random split of focal recordings does not resemble the test: the test is long soundscapes (background noise,
  overlapping or absent calls, different recording conditions), and files from the same recording, recordist
  or site may sit on both sides of a random split.
- Fixes: group the split by recording/site, and validate on test-like data (labelled soundscapes, if any exist)
  rather than on focal clips; track per-class scores because the macro metric is driven by rare classes.

Training changes for the domain gap (random 5-second crops of weakly labelled clips, secondary labels as soft
targets, mixup with background/no-call noise) are a plus. FAIL if it treats the gap as generic overfitting
without naming the focal-vs-soundscape mismatch.
