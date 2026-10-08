---
type: llm
weight: 1
---

The response identifies leakage as the main cause: (1) shuffled KFold on temporal data lets the model train on
the future of each validation period; (2) rolling features computed without shifting by the forecast horizon
leak validation/test-period information, and (3) the target encoding fit on the full training set leaks the
target into validation (should be out-of-fold / time-respecting). It prescribes a forward-chaining time-based
validation whose validation window mirrors the 3-month test period, with a gap at least the forecast horizon,
and lag/rolling features shifted by the horizon. Mentioning adversarial validation or checking CV-LB agreement
afterwards is a plus. Fail if it attributes the gap mainly to overfitting/hyperparameters without naming the
time leakage.
