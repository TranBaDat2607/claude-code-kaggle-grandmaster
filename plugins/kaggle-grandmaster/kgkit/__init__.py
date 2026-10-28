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
- ``kgkit.compare``     is the new experiment really better? Paired fold test, and the noise of a score or
                        a paired difference on any metric (bootstrap, Bayesian bootstrap, influence CLT,
                        projected to the public LB's size), seed noise from re-runs, ROPE
- ``kgkit.evalset``     the evaluation set's fingerprint and the paired noise measured on it
- ``kgkit.backlog``     the ranked idea backlog and screen-vs-full fidelity
- ``kgkit.kernels``     study and reproduce public notebooks; their public score on record and title claims
- ``kgkit.finals``      which submissions Kaggle would auto-select as finals and whether one is public-LB fitted
- ``kgkit.kaggle_submissions`` the team's scored and selected submissions from Kaggle's API (run in a child process)
- ``kgkit.kaggle_search`` notebooks' best public scores from Kaggle's search API (run in a child process)
- ``kgkit.kaggle_outputs`` the files and log of a kernel's last session, e.g. a run its cap stopped (child process)
- ``kgkit.kaggle_cli``  the Kaggle CLI with Kaggle's reason for a refused request (kgkit's CLI calls run through it)
- ``kgkit.discussions`` competition forums (topics, search, write-ups, threads), live or from Meta Kaggle
- ``kgkit.datasets``    dataset download with a per-file fallback when the whole-dataset archive 404s
- ``kgkit.pages``       a competition's own pages (evaluation, rules, data description) as text
- ``kgkit.safety``      credential checks for uploads; a boundary around text written by other Kaggle users
- ``kgkit.recheck``     re-draw the folds with a new seed and re-run old vs current: does the gain survive?
- ``kgkit.submission``  submission validation against sample_submission
- ``kgkit.state``       competition state (``.kaggle-gm/competition.json``)
- ``kgkit.gpu``         training on Kaggle GPUs/TPUs: quota, remote kernels, accelerator-hour accounting
- ``kgkit.kaggle_gpu_runner`` the script of a remote training kernel, which ``gpu build`` fills in (runs inside
                        Kaggle; kept in the package so that a copy made by ``kgkit vendor`` has its own version's)
- ``kgkit.budget``      training-loop guard: session deadline, resume checkpoints, learning-curve pruning

Run ``python -m kgkit --help`` for the command line interface.
"""

__version__ = "1.11.0"
