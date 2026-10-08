# Changelog

## 1.0.0 — 2026-10-08

First complete release of the `kaggle-grandmaster` plugin.

- **kgkit** toolkit: leak-free CV folds (stratified, group, stratified-group, multilabel,
  time-series with gap, purged), 35+ metrics with direction and input kind, threshold and
  QWK rounding optimisers, hill climbing / weight optimisation / honest nested blend scoring /
  stacking, adversarial validation, leak-safe feature primitives, EDA red flags, experiment
  ledger with git and fold fingerprints, metric-aware submission validation, CLI.
- **19 knowledge skills** covering the competition lifecycle and every major domain.
- **15 slash commands** from `kg-start` to `kg-postmortem`, including the autonomous `kg-grind` loop.
- **9 specialist subagents.**
- **Hooks**: session brief, Kaggle-prompt skill router, submission guard (validation, daily
  budget, credential protection), submission logger.
- **Templates**: GBDT, timm image, HF transformer, offline inference kernel.
- **Evals**: 7-case `claude plugin eval` suite with a no-plugin baseline arm.
- Verified with 63 tests, live Claude Code sessions, and a real-data dry run on
  playground-series-s6e2.
