---
type: llm
---

PASS when the response gets the quota timing right:

- The ~11 hours left this week expire at the weekly reset in ~30 hours, so they should be planned to be used
  before then (not left mostly unspent).
- More weekly quotas arrive before the deadline in 16 days (resets at roughly days 1.25, 8.25 and 15.25, so about
  three more; "two or three", or asking the user to confirm the weekly size, is fine).
- Some GPU time is reserved near the end for final models and inference.

FAIL if it ignores the reset, treats the 11 hours as all the GPU time left before the deadline, or plans to
save the 11 hours for later.
