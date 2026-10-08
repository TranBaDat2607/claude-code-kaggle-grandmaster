---
name: time-series
description: Winning playbook for Kaggle time-series and forecasting competitions — leak-free time validation, lag/rolling/calendar features, global GBDT models, recursive vs direct multi-horizon forecasting, hierarchical reconciliation, deep forecasting models, financial/market prediction with purged CV, and sensor/event sequences. Use whenever the target depends on time or the test period is in the future.
---

# Time-Series Playbook

## 1. Validation first

- Forward-chaining splits (`kgkit.cv.time_series_splits`) with `gap ≥ forecast horizon`
  and validation windows the same length as the test period. The **last** fold best mirrors
  the LB; weight it, but use several folds to measure stability.
- Never shuffle. Never compute features with data after the forecast origin.
- For targets defined over overlapping windows (finance), use purged K-fold with embargo.
- Simulate the exact inference situation: at forecast origin T you only know data ≤ T (and
  sometimes with a reporting delay). Reproduce that delay in features.

## 2. The global GBDT recipe (wins most forecasting comps)

One LightGBM over all series (store×item etc.), rows = (series, date), features:
- **Lags** of the target at horizons ≥ forecast horizon (direct) or rolling recursion;
  lags at seasonal periods (7, 14, 28, 364 for daily).
- **Rolling stats** (mean/std/min/max/median/EWM) over windows, shifted by the horizon
  (`kgkit.features.lag_features(shift=h)`).
- **Calendar**: dow, dom, week, month, holidays/events (country-specific), paydays, school terms,
  days to/from events; cyclical encodings.
- **Series-level static**: category, store, item attributes; target encodings by group (OOF in time).
- **Price/promotion/exogenous**: relative price vs series mean, promo flags, weather.
- **Trend proxies**: ratio of recent window mean to long window mean.
Objectives: Tweedie/Poisson for sparse counts (M5), L2 on log1p for skewed sales, L1 for MAE.

## 3. Multi-horizon strategy

| Strategy | When |
|---|---|
| Direct (one model per horizon, or horizon as a feature with lags ≥ h) | stable, no error accumulation; default |
| Recursive (predict t+1, feed back) | short horizons, strong autocorrelation; risk of drift |
| Hybrid / ensemble of both | often best on LB |

## 4. Other model families (for diversity)

- Statistical baselines per series: seasonal naive, ETS, Theta, ARIMA (statsforecast) — a
  strong sanity check and blend member for few long series.
- Deep: N-BEATS / N-HiTS, TFT, PatchTST, TimesNet, LSTM/GRU/1D-CNN seq2seq (neuralforecast /
  pytorch-forecasting); pretrained foundation models (Chronos, TimesFM, Moirai) zero-shot or
  fine-tuned as features/blend members.
- Hierarchical data: forecast at several levels and reconcile (bottom-up, MinT); multiply
  level-forecasts by historical proportions.

## 5. Post-processing

- Multiply-by-factor calibration ("magic multipliers") tuned on the last validation window is a
  known M5-era trick — validate on several windows to avoid overfitting.
- Clip negatives; zero-out series that are dead (no sales in last N days); round counts only if
  the metric rewards it.

## 6. Financial / market prediction specifics

- Extremely low signal-to-noise; metric often correlation-based (Pearson/Spearman per time-step,
  Sharpe-like). Normalise features cross-sectionally per time step (rank/z-score).
- Purged CV with embargo; evaluate per-period metrics and their stability.
- Online/streaming API competitions: the evaluation iterates over time; your inference must
  update state incrementally and fit time budgets per step. Online learning (periodic re-fit
  with new data) often helps.
- Prefer robust, simple ensembles; private LB is future data — heavy shake-ups are normal.

## 7. Sensor / event sequences

Event detection (e.g. sleep onset, seizures): sequence models (1D U-Net/GRU/transformer) over
windows with multi-scale features; predict per-step probabilities, then peak-detection
post-processing tuned to the tolerance-based metric (AP over tolerances). Validate grouped by
subject.
