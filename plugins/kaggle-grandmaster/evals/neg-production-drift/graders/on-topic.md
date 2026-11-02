---
type: llm
---

PASS when the response gives practical production monitoring advice: drift metrics per feature and on the score
distribution (PSI, KS, or similar) with common thresholds (for example PSI below 0.1 stable, 0.1-0.25 watch, above
0.25 act), monitoring actual performance once (delayed) fraud labels arrive, alerting that avoids noise, and
retraining triggers (scheduled cadence and/or drift or performance thresholds) with validation before
redeploying.

FAIL if it frames the answer around Kaggle competitions (leaderboards, submissions, public/private LB), or does
not answer the question.
