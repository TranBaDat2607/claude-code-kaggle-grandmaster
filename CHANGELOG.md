# Changelog

## 1.2.0 — 2026-10-09

Closing the gaps between the plugin and how Grandmasters actually run competitions: honest
experiment decisions, feature search and ensembling at Playground-winning scale, diverse
baselines, public-notebook tooling, and the career layer (competition choice, solo gold, teaming).

- **Paired experiment comparison** (`kgkit.compare`, `kgkit ledger compare`): per-fold paired test
  plus a paired (optionally group-wise) OOF bootstrap → KEEP / DISCARD / INCONCLUSIVE, with a leak flag
  for implausible jumps. Replaces the "gain > one fold std" rule everywhere. That rule threw away
  consistent small gains because fold difficulty cancels in a paired comparison. Works on unassembled
  1-fold screens (`artifacts/<name>/fold<k>_*.npy`, `--fold k`).
- **Accepted baseline and lineage**: ledger records gain `parent` and `decision`;
  `ledger decide / baseline / lineage`; the accepted baseline (last KEEP) replaces "highest CV" as the
  reference in the skills, `status`, the session brief and `gpu --prune-against baseline`.
  `status` warns after 20 decisions on one split (CV overfitting) and the playbook adds the
  fresh-seed recheck and reverse-ablation procedure.
- **Idea backlog** (`kgkit.backlog`, `kgkit backlog add|list|set|render|fidelity`): ranked by
  (gain × prob) / cost with evidence, source and outcome; `fidelity` measures whether cheap screens
  rank ideas like full CV. Wired into `/kg-ideas`, `/kg-grind`, `/kg-status`, the error analyst and the
  session brief.
- **Feature search** (`kgkit.featsearch`, `kgkit features search`): generates count / OOF target
  encodings of columns and pairs, groupby aggregates and pairwise arithmetic, screens them in batches
  with a fixed fast GBDT, prunes kept batches by importance and rechecks the result on an
  independently seeded split.
- **Ensembling at scale**: `prune_library` (drop near-duplicate OOFs), `multi_level_stack`
  (layers of linear + shallow-GBDT meta-models with a hill-climbed final blend and an honest score),
  `residual_stack`, raw features in `stack(X_extra=...)`; `kgkit blend --method stack|residual
  --levels --base --prune`.
- **Diverse baselines**: `train_gbdt.py` gains `--model linear|knn|svm|mlp` (sklearn pipelines);
  `/kg-baseline` and `/kg-start` run them on day 1.
- **Public notebooks** (`kgkit.kernels`, `kgkit kernels top|pull|review`): pulls a notebook with a
  `REVIEW.md` (CV scheme, model families, printed scores, external inputs, optional access check) and
  a local script with Kaggle paths remapped; `kgkit ledger import` brings its (or a teammate's) OOF
  in with CV recomputed on your folds.
- **Discussions** (`kgkit.discussions`, `kgkit discussions sync|top|search|solutions|read`): the competition forum from Meta Kaggle, Kaggle's official daily export (the CLI has no discussion commands). Topic index, title search, "Nth place solution" write-ups of any past competition, full threads from the cached message export, and rule facts (merger deadline, team size, daily submissions, metric).
- **Fresh-split recheck** (`kgkit.recheck`, `kgkit recheck`): re-draws the folds with the frozen recipe (now recorded by `kgkit folds` in `.kaggle-gm/folds_spec.json`) and a new seed, re-runs the old and current pipelines (templates honour `KG_FOLDS_FILE`), and reports how much of the gain survived. Recheck records are tagged and never become a baseline or best CV.
- **Skill `competition-strategy`**: Grandmaster requirements, which competitions award medals,
  choosing competitions, the solo gold, joining late, teaming rules (no private sharing before a
  merge, submission counts) and the merge protocol.
- Updated facts: TabPFN v2.5+ scale, TabM / RealMLP, AutoGluon, seed bagging at scale,
  Playground-scale stacks; screens are compared against a control screen at the same reduced settings.
- **Hooks**: router routes for `competition-strategy`, `competition-recon` and experiment-noise
  questions; session brief shows the accepted baseline and the backlog top.
- **Evals** `experiment-noise`, `team-merge`; 28 new tests (112 total).

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
