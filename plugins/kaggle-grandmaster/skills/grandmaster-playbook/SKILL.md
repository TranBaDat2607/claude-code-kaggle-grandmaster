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
3. **One change at a time, logged.** Every experiment goes in the ledger with CV, fold
   scores, and its OOF/test predictions. Your future ensemble is built from this ledger.
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
| Foundation | days 1–3 | metric implementation, frozen folds, EDA, adversarial validation, first baseline + first submission | `validation-strategy`, `/kg-eda`, `/kg-cv`, `/kg-baseline` |
| Exploration | ~50% | many fast experiments: features, architectures, losses, augmentations, data cleaning | domain skills, `/kg-experiment`, `/kg-grind` |
| Scaling | ~25% | bigger models/backbones, more epochs, full data, more folds/seeds; pseudo-labels, distillation | `deep-learning-training`, `leaderboard-boosters` |
| Ensembling | last ~15% | diverse strong models, hill climbing, stacking, post-processing | `ensembling`, `/kg-ensemble` |
| Endgame | last days | robustness checks, inference within limits, final 2 picks | `code-competitions`, `final-submission-selection`, `/kg-final` |

Make a valid submission on day 1 — it de-risks the pipeline (format, kernel runtime, offline
packages) and calibrates CV vs LB.

## 2. The experiment loop

```
hypothesis → smallest change that tests it → run on frozen folds → log to ledger
→ compare with current best (Δ vs fold std, ≥2 seeds if marginal) → keep / discard
→ write one line of learning in CLAUDE.md "notes"
```

- Keep a running **idea backlog** ranked by (expected gain × probability) / cost.
  Sources: EDA red flags, error analysis, discussion forum, past similar competitions'
  winning write-ups, domain knowledge.
- A gain smaller than ~1 fold-std is noise until confirmed with another seed or more folds.
- Re-run the strongest pipeline end-to-end periodically from a clean checkout — catches
  hidden state and makes the final submission reproducible.

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

## 4. The toolkit

The plugin ships `kgkit` (Python, numpy/pandas/sklearn/scipy). Locate it via the
`KGKIT_HOME` environment variable (set by the plugin's SessionStart hook) — otherwise it is
the `kgkit/` folder inside the plugin root. Run it as
`PYTHONPATH="$KGKIT_HOME" python -m kgkit <cmd>`, or vendor it into the project with
`python -m kgkit vendor src/` so training scripts can `from kgkit import ...`.

Key commands: `init`, `status`, `folds`, `eda`, `adv`, `ledger`, `blend`, `validate`, `metrics`.
Training scripts should use `kgkit.cv.split_indices`, `kgkit.metrics`, and
`kgkit.experiment.Ledger().log(...)`. Templates for GBDT / image / transformer training and
offline inference kernels are in the plugin's `templates/` folder.

## 5. Daily rhythm

1. `python -m kgkit status` — re-orient (best CV, CV-LB correlation, last experiments).
2. Check the competition discussion for new findings (leaks, data updates, rule clarifications).
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
- Huge hyperparameter searches before the features and validation are right.
- Ensembling OOFs from different fold splits (blend weights become optimistic garbage).
- Ignoring runtime limits until the last day of a code competition.
- Not reading the rules on external data / pretrained models / team size.
