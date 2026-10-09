# Changelog

## 1.1.0 — 2026-10-09

Quota-aware training on Kaggle's GPUs.

- **`kgkit gpu`**: `quota` (live `kaggle quota`, reset countdown, use-it-or-lose-it warning),
  `plan` (GPU hours until the deadline, final-week reserve, measured cost per fold), `build`
  (turns a local training command into a private kernel with your code, kgkit, competition
  state and frozen folds embedded), `push` (quota check, `-t` cap, accelerator), `wait`,
  `collect` (ledger import with fresh ids, GPU-hour accounting, diagnostics), `log`.
- **Remote runner** (`templates/kaggle_gpu_runner.py`): links attached inputs into `./data`, runs
  one process per (job, fold) per GPU so both T4s work, packs several jobs into one session
  (`--extra-job`, e.g. two 1-fold screens), stops gracefully before the session limit, resumes
  from a previous kernel's output, assembles each finished job once, samples GPU utilisation.
- **`kgkit.budget.TrainBudget`**: session-deadline guard with resume checkpoints and
  learning-curve pruning against a baseline (`--prune-against best`). Wired into the image and
  transformer templates, which also gain `--assemble`.
- **Skill `kaggle-gpu`** (+ throughput reference) and command **`/kg-gpu`**: screen → promote →
  full-fold protocol, accelerator choice, preprocessing off-quota, budgeting.
- **Hooks**: ask before a GPU `kaggle kernels push` that the cached quota cannot cover; log raw
  GPU pushes; quota line in the session brief; prompt route for GPU/quota questions.
- **Eval** `gpu-quota-budget`; tests for the budget guard, quota accounting, kernel build,
  the runner executed in a simulated `/kaggle` layout (two fake GPUs, resume, pruning, packed
  screens), collect, the new hooks and template deadline/resume/assemble.

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
