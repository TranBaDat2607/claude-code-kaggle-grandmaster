---
name: grandmaster-playbook
description: The master operating procedure for competing in a Kaggle (or Kaggle-style) machine learning competition at Grandmaster level. Use whenever the user is working on a Kaggle competition, wants to improve a leaderboard score, asks how to approach a data science competition, or is in a workspace containing .kaggle-gm/ — it routes to the specialised skills (recon, validation, tabular, vision, NLP/LLM, time series, ensembling, code competitions, final selection).
---

# The Grandmaster Playbook

Winning Kaggle is not about one clever model. It is a *process*: a trustworthy validation
loop, a high rate of small well-logged experiments, relentless study of the data and of
what others found, and disciplined ensembling and submission selection at the end.
Follow this procedure; deviate only with a reason.

## 0. Principles (internalise these)

1. **Validation is everything.** If your CV does not track the private LB, nothing else
   matters. Spend real time building it; then trust it over the public LB.
2. **Iteration speed wins.** A pipeline that gives a reliable CV number in 5 minutes beats
   a "better" one that takes 5 hours. Subsample, cache features, use smaller backbones /
   fewer folds for exploration; scale up only for confirmed ideas.
3. **One change at a time, logged, decided.** Every experiment goes in the ledger with CV, fold
   scores, its OOF/test predictions, the baseline it was compared against and the decision.
   Your future ensemble is built from this ledger.
4. **Read before you code.** The overview, evaluation page, data page, rules, and the top
   public notebooks + discussion threads routinely contain the single most valuable
   insight of the competition (a leak, a data quirk, a metric trick).
5. **Look at the data.** Raw rows, images, texts, audio, errors of your model. Most
   winning ideas come from error analysis, not hyperparameter search.
6. **Diversity beats a single best model.** Different algorithms, inputs, targets, seeds →
   ensemble. But only blend models validated on the same folds.
7. **Robustness over public LB rank.** Choose final submissions for expected *private*
   score; shake-ups punish public-LB chasers.

## 1. Phase map

| Phase | Time share | Goal | Skills / commands |
|---|---|---|---|
| Recon | day 1 | understand metric, data, rules, timeline, prior art | `competition-recon`, `/kg-recon` |
| Foundation | days 1–3 | metric implementation, frozen folds, EDA, adversarial validation, **diverse** baselines (GBDT + linear + NN/kNN on the same folds) + first submission | `validation-strategy`, `/kg-eda`, `/kg-cv`, `/kg-baseline` |
| Exploration | ~50% | many fast experiments: features, architectures, losses, augmentations, data cleaning | domain skills, `/kg-experiment`, `/kg-grind` |
| Scaling | ~25% | bigger models/backbones, more epochs, full data, more folds/seeds; pseudo-labels, distillation | `deep-learning-training`, `leaderboard-boosters` |
| Ensembling | last ~15% | diverse strong models, hill climbing, stacking, post-processing | `ensembling`, `/kg-ensemble` |
| Endgame | last days | robustness checks, inference within limits, final 2 picks | `code-competitions`, `final-submission-selection`, `/kg-final` |

Make a valid submission on day 1 — it de-risks the pipeline (format, kernel runtime, offline
packages) and calibrates CV vs LB. Run several model *families* early, not only the strongest one:
which families fit the data is information, a linear model that matches the GBDT hints at a leak
or a simple signal, and the blend needs them later anyway.

## 2. The experiment loop

```
backlog top idea → hypothesis → smallest change that tests it → run on frozen folds
→ log to ledger (parent = accepted baseline) → kgkit ledger compare <new>
→ KEEP / DISCARD / INCONCLUSIVE → kgkit ledger decide → backlog set → one line in CLAUDE.md notes
```

- **Compare against the accepted baseline, not the best CV.** The baseline is the last experiment
  you *decided* to keep (`kgkit ledger baseline`); the highest CV may be a lucky seed or a leak.
- **Paired noise, not fold std.** Fold std mostly measures how hard each fold is, and both models
  share that, so it cancels. Judge a gain by the per-fold *differences* (all folds up by a consistent
  amount = real, even if far below the fold std) and/or a paired bootstrap on the OOFs:
  `kgkit ledger compare <new> --truth data/train.csv:<target> --folds data/folds.csv:fold`
  (add `--groups` when rows are clustered, e.g. images per patient). Positive but within ~2 paired
  standard errors = INCONCLUSIVE: run a second seed, or keep it only if it costs nothing.
  A gain several times the fold std is a leak suspect: run `validation-auditor` first.
- **The backlog is data.** `kgkit backlog add/list/set` keeps ideas ranked by
  (gain × probability) / cost with their evidence and outcome, so failed ideas are not retried.
  Sources: EDA red flags, error analysis, discussion forum, public notebooks
  (`kgkit kernels pull`), past similar competitions' write-ups, domain knowledge, brainstorming.
  Tag each idea's source; after a few weeks, the sources whose ideas keep getting KEEPs deserve
  more of your time.
- Re-run the strongest pipeline end-to-end periodically from a clean checkout — catches
  hidden state and makes the final submission reproducible.

### Guarding against overfitting the CV itself

Every keep/discard decision on the same folds spends a little of their power to give an unbiased
estimate; after dozens of decisions the accepted baseline's CV is optimistic, and some KEEPs were
noise that happened to land positive. `kgkit status` warns after 20 decisions on one split.
- **Re-check on a fresh split** every ~20 decisions:
  `python -m kgkit recheck --seed 7 --run "<train the old pipeline>" --run "<train the current one>"
  --exp <old_id> <current_id> --truth data/train.csv:<target>` re-draws the folds with the frozen
  split's recipe (recorded by `kgkit folds`) and a new seed, runs both commands on it
  (`KG_FOLDS_FILE` is set and the templates honour it), and reports how much of the original gain
  survived: holds / shrank by more than half / did not survive. Recheck records are tagged and never
  become a baseline. "Old pipeline" is usually the root of `kgkit ledger lineage` (check out its commit
  in a git worktree, or toggle its flags off). `kgkit features search --recheck-seed` does the same
  for feature search.
- **Reverse ablation**: `kgkit ledger lineage` lists the chain of KEEPs behind the baseline; remove
  the marginal ones one at a time from the current pipeline. Changes that no longer help get dropped,
  which also simplifies the pipeline.
- A small untouched **holdout** (or the public LB, sparingly) for the final yes/no on big changes.

### Screens must predict full runs

Cheap screens (1 fold, lower resolution, fewer epochs, `kaggle-gpu`) are only useful if they rank
ideas like the full run does. Record both gains on the backlog item (`--screen-gain`, `--full-gain`)
and check `kgkit backlog fidelity` after a few promotions; low agreement means screening at higher
fidelity (2 folds, closer-to-final resolution) before promoting.

## 3. Choosing the domain playbook

| Data | Skill |
|---|---|
| Rows × columns, GBDT territory | `tabular-mastery` |
| Images, video, medical scans, detection, segmentation | `computer-vision` |
| Text classification/regression, NER, QA, LLM fine-tuning, LLM-as-solver, retrieval | `nlp-and-llm` |
| Forecasting, sequences over time, sensors, finance | `time-series` |
| Audio, EEG/ECG, spectrograms, bioacoustics | `audio-and-signal` |
| Agents in game environments (simulation comps), combinatorial optimisation (Santa) | `simulation-and-optimization` |
| Recommendation, ranking, retrieval, candidate generation | `recsys-and-ranking` |

Cross-cutting: `validation-strategy`, `metric-optimization`, `ensembling`,
`hyperparameter-tuning`, `deep-learning-training`, `leaderboard-boosters`,
`code-competitions`, `kaggle-cli`, `cv-lb-debugging`, `final-submission-selection`.
Across competitions (which ones to enter, solo gold, teaming and merging): `competition-strategy`.

## 4. The toolkit

The plugin ships `kgkit` (Python, numpy/pandas/sklearn/scipy). Locate it via the
`KGKIT_HOME` environment variable (set by the plugin's SessionStart hook) — otherwise it is
the `kgkit/` folder inside the plugin root. Run it as
`PYTHONPATH="$KGKIT_HOME" python -m kgkit <cmd>`, or vendor it into the project with
`python -m kgkit vendor src/` so training scripts can `from kgkit import ...`.

Key commands: `init`, `status`, `folds`, `eda`, `adv`, `ledger` (`compare`, `decide`, `baseline`,
`lineage`, `import`), `backlog`, `features search`, `kernels top/pull`, `blend` (hill / weights /
rank / multi-level `stack` / `residual`, `--prune`), `validate`, `metrics`, `gpu`.
Training scripts should use `kgkit.cv.split_indices`, `kgkit.metrics`, and
`kgkit.experiment.Ledger().log(..., parent=<baseline id>)`. Templates for tabular (GBDTs plus
linear / kNN / SVM / MLP), image and transformer training and offline inference kernels are in the
plugin's `templates/` folder.

## 5. Daily rhythm

1. `python -m kgkit status` — re-orient (accepted baseline, best CV, CV-LB correlation, last
   experiments, top of the backlog).
2. Check the competition discussion and newly popular notebooks (`kgkit kernels top --sort-by
   hotness`) for new findings (leaks, data updates, rule clarifications, techniques).
3. Run the highest-value experiments from the backlog; keep GPUs busy overnight with the
   long runs, use the day for analysis and fast experiments.
4. Use the daily submissions deliberately: to verify CV-LB agreement on *different kinds*
   of changes, not to hill-climb the public LB.
5. End the day by updating the notes in CLAUDE.md: learnings, next experiments.

## 6. Anti-patterns that lose medals

- Tuning on public LB; picking finals by public rank in a small/noisy public split.
- Random KFold on grouped / temporal data (leaks → CV way above LB).
- Fitting scalers / encoders / feature selection / target encoding on all data before CV.
- Comparing experiments run on different folds or seeds and calling it a gain.
- Discarding consistent small gains because they are below the fold std (paired noise is far smaller),
  and accepting single-seed gains that are within paired noise.
- Letting the highest CV, rather than a decided baseline, become the reference for new ideas.
- Huge hyperparameter searches before the features and validation are right.
- Ensembling OOFs from different fold splits (blend weights become optimistic garbage).
- Ignoring runtime limits until the last day of a code competition.
- Not reading the rules on external data / pretrained models / team size.
