# Playground Series: synthetic data made from an original dataset

Read this when the competition is a Playground Series (or any competition whose rows a model generated from an
original dataset). The rest of `tabular-mastery` applies as for any tabular competition.

Synthetic data generated from an original dataset: large (100k–1M rows), small signal-to-noise
gains, LB shake-ups of ±0.001 are common. Winning pattern: many diverse models (GBDTs with
different FE, NNs, linear/kNN/SVR models on transformed features), *many* OOFs, hill climbing
or ridge/logistic stacking over 20–100 OOFs, appending the original dataset, and selecting finals
on CV. Keep everything on 5–10 identical folds.

**Large stacks.** With hundreds of experiments, the strongest Playground ensembles stack 100+ models in 2–4
levels: diverse level-1 members (linear, SVR, kNN, random forest, MLP, TabPFN, GBDTs) under linear, shallow-GBDT
or MLP meta-models, then a weighted average. It only pays when every experiment saved OOF and test predictions on
the same folds (the ledger does this). Tooling: `kgkit blend --prune 40` (drop near-duplicate OOFs),
`--method stack --levels 2` (layers of linear + shallow-GBDT meta-models, then hill climbing; keep the deeper
stack only if its honest score beats `--levels 1` by more than paired noise), and `--method residual --base <id>`
(stage 2 learns one strong model's errors). The steps are in `ensembling` (Stacking).

**The data-generation process is a lever of its own.** Late in a Playground most teams stack the same
public OOF libraries, and rank inside that crowd is noise (`kgkit status`: CROWD). What moves a team out
of it is a member the libraries lack, and the generator is the usual place to find one:
- Fingerprint the generator. Compare train/test rows with the original dataset: value grids, rounding
  and digit patterns, values the original never has, duplicated or near-copied rows, categories
  that collapse. Each artefact is a feature, or a hint at the model that made the data.
- Try a **class-conditional generative member**: model the feature distribution of each class and
  score a row with Bayes' rule, `log P(x | y=1) - log P(x | y=0) + log prior odds`. It errs differently
  from discriminative GBDTs and NNs, so it adds to a saturated stack. Fit it only on the training folds'
  labelled rows, exactly like any OOF member. A label-free density model of the rows usually adds ~0.
- It is expensive and not yet proven across competitions: screen it on one fold first, and give it
  `--gpu-hours` in the backlog. [synthetic-generators.md](synthetic-generators.md) has a worked example.
