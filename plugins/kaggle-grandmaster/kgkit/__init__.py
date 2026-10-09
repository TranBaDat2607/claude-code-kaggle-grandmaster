"""kgkit — the Kaggle Grandmaster toolkit shipped with the kaggle-grandmaster plugin.

Small, dependency-light (numpy, pandas, scikit-learn, scipy) building blocks for the
parts of a competition that are easy to get subtly wrong:

- ``kgkit.cv``          leak-free fold assignment (stratified, group, stratified-group,
                        multilabel, time-series, purged) and fold sanity checks
- ``kgkit.metrics``     a registry of Kaggle metrics with direction + input type
- ``kgkit.thresholds``  threshold / rounding optimisation (F1, QWK, multilabel)
- ``kgkit.ensemble``    hill climbing, constrained weight optimisation, rank blending, stacking
- ``kgkit.adversarial`` adversarial validation (train/test shift detection)
- ``kgkit.features``    OOF target encoding, count encoding, group aggregates, lags, dates
- ``kgkit.eda``         a fast markdown EDA report focused on competition pitfalls
- ``kgkit.experiment``  the experiment ledger (CV, LB, artefacts, git commit) + seeding
- ``kgkit.submission``  submission validation against sample_submission
- ``kgkit.state``       competition state (``.kaggle-gm/competition.json``)
- ``kgkit.gpu``         training on Kaggle GPUs: quota, remote kernels, GPU-hour accounting
- ``kgkit.budget``      training-loop guard: session deadline, resume checkpoints, learning-curve pruning

Run ``python -m kgkit --help`` for the command line interface.
"""

__version__ = "1.1.0"
