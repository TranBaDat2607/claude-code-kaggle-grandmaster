"""kgkit — the Kaggle Grandmaster toolkit shipped with the kaggle-grandmaster plugin.

Small, dependency-light (numpy, pandas, scikit-learn, scipy) building blocks for the
parts of a competition that are easy to get subtly wrong:

- ``kgkit.cv``          leak-free fold assignment (stratified, group, stratified-group,
                        multilabel, time-series, purged) and fold sanity checks
- ``kgkit.metrics``     a registry of Kaggle metrics with direction + input type
- ``kgkit.thresholds``  threshold / rounding optimisation (F1, QWK, multilabel)
- ``kgkit.ensemble``    hill climbing, weight optimisation, rank blending, stacking (multi-level,
                        residual), library pruning
- ``kgkit.adversarial`` adversarial validation (train/test shift detection)
- ``kgkit.features``    OOF target encoding, count encoding, group aggregates, lags, dates
- ``kgkit.featsearch``  brute-force feature generation + batch screening on the frozen folds
- ``kgkit.eda``         a fast markdown EDA report focused on competition pitfalls
- ``kgkit.experiment``  the experiment ledger (CV, LB, artefacts, git commit, decisions, lineage) + seeding
- ``kgkit.compare``     paired fold test + paired bootstrap: is the new experiment really better?
- ``kgkit.backlog``     the ranked idea backlog and screen-vs-full fidelity
- ``kgkit.kernels``     study and reproduce public notebooks
- ``kgkit.discussions`` competition forums (topics, search, write-ups, threads) from Meta Kaggle
- ``kgkit.recheck``     re-draw the folds with a new seed and re-run old vs current: does the gain survive?
- ``kgkit.submission``  submission validation against sample_submission
- ``kgkit.state``       competition state (``.kaggle-gm/competition.json``)
- ``kgkit.gpu``         training on Kaggle GPUs: quota, remote kernels, GPU-hour accounting
- ``kgkit.budget``      training-loop guard: session deadline, resume checkpoints, learning-curve pruning

Run ``python -m kgkit --help`` for the command line interface.
"""

__version__ = "1.2.0"
