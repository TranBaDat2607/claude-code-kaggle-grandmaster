---
type: llm
---

PASS when the response does both:

1. Points out that the validation metric must be computed exactly like the host's: Dice per image, then
   averaged, with the empty-mask rule. A global pixel-level Dice does not measure the competition score and
   hides how the empty images are handled.
2. Handles the empty images, which dominate the score: an empty-mask gate (an image-level classifier, or a rule
   on the maximum probability or predicted area) and/or removing small predicted components, with the
   probability threshold and the minimum area tuned on out-of-fold predictions against the per-image metric.

Loss changes (BCE + Dice or Lovasz), TTA or larger resolution are a plus. FAIL if it does not address the
per-image metric with empty masks.
