---
type: llm
---

PASS when the response does both:

1. Explains that lag_1, lag_2 and the 7-day rolling mean use actual sales that are not known when forecasting
   days 2-28 (at the forecast origin only data up to that day exists), so the validation, which reads the
   actual values inside the 28-day window, is leaky and over-optimistic.
2. Gives a sound multi-horizon strategy and a matching validation: either direct forecasting with lag and
   rolling features shifted by at least the horizon (per-horizon models, or horizon as a feature with lags >= h)
   or recursive forecasting that feeds predictions back (noting error accumulation/drift), and a validation that
   simulates the forecast origin (features built only from data up to the origin, a 28-day window, a gap of at
   least the horizon or several forward-chaining windows).

A Tweedie or Poisson objective for the zero-heavy counts is a plus. FAIL if it accepts lag_1 for all 28 days as
is, or never says the validation must reproduce what is known at the forecast origin.
