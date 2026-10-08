# Ultimate Grandmaster Kaggle Kit

A Claude Code plugin that makes Claude compete like a Kaggle Grandmaster. It gives Claude a
working method on top of model knowledge: validation it can trust, a high rate of small
experiments that are all logged, honest ensembling, offline packaging for code competitions,
and careful choice of final submissions. It covers every competition type: tabular, computer
vision, NLP/LLM, time series, audio/signal, recsys, simulation agents and combinatorial
optimisation.

```
/kaggle-grandmaster:kg-start playground-series-s6e2
```
This one command runs recon, data download, workspace setup, EDA, adversarial validation,
frozen folds, a baseline model and a validated submission file.

---

## What's inside

| Layer | Count | What it does |
|---|---|---|
| **Knowledge skills** | 19 | Playbooks Claude loads when relevant: master playbook, recon, validation, tabular, CV, NLP/LLM, time series, audio, simulation/optimisation, recsys, ensembling, metric optimisation, HPO, DL training, leaderboard boosters, code competitions, Kaggle CLI, CV↔LB debugging, final-submission selection |
| **Slash commands** | 15 | `kg-start`, `kg-recon`, `kg-eda`, `kg-cv`, `kg-baseline`, `kg-experiment`, `kg-grind`, `kg-ensemble`, `kg-submit`, `kg-status`, `kg-kernel`, `kg-final`, `kg-debug`, `kg-ideas`, `kg-postmortem` |
| **Specialist agents** | 9 | competition-analyst, solution-researcher, data-detective, validation-auditor, feature-engineer, experiment-runner, error-analyst, ensemble-architect, kernel-packager |
| **Hooks** | 4 | session brief, Kaggle-prompt → skill router, submission guard (validation + daily budget + credential protection), submission logger |
| **`kgkit` toolkit** | 11 modules | Tested Python: leak-free folds, 35+ metrics, threshold/rounding optimisers, hill climbing / stacking, adversarial validation, leak-safe features, EDA red flags, experiment ledger, submission validator, CLI |
| **Templates** | 4 + metadata | GBDT (LightGBM/XGBoost/CatBoost/HGB), timm image models, HF transformers, offline inference kernel |
| **Evals** | 7 cases | `claude plugin eval` suite with a no-plugin baseline arm |

## Install

Requirements: Claude Code, Python ≥ 3.9 with `numpy pandas scikit-learn scipy` (the GBDT/DL
libraries are only needed for the matching templates), and a POSIX `sh` for hooks (Git Bash
on Windows, which Claude Code already uses). For Kaggle access: `pip install kaggle` plus an
API token.

```bash
# from a clone of this repo
claude plugin marketplace add /path/to/ultimate-grandmaster-kaggle-kit
claude plugin install kaggle-grandmaster@ultimate-grandmaster-kaggle-kit

# or just try it for one session
claude --plugin-dir /path/to/ultimate-grandmaster-kaggle-kit/plugins/kaggle-grandmaster
```

Plugin commands are namespaced: `/kaggle-grandmaster:kg-status`. Plain `/kg-status` also works
when no other plugin uses the same name.

## How a competition flows

```
kg-start ─► recon (competition-analyst ∥ solution-researcher) ─► data ─► kgkit init
        ─► EDA + adversarial validation ─► frozen folds ─► metric impl ─► baseline + 1st submission
                                   │
             ┌─────────────────────┘
             ▼
   kg-experiment / kg-grind  ◄── kg-ideas (error-analyst, prior art, untried levers)
   (one change · frozen folds · ledger · Δ vs fold std · keep/discard)
             │
             ├─► kg-submit (validated, budgeted, LB attached to ledger) ─► CV↔LB correlation
             ├─► kg-debug when CV and LB disagree (validation-auditor)
             ▼
   kg-ensemble (same-fold OOFs, hill climbing, honest nested score) ─► kg-kernel (code comps)
             ▼
   kg-final (best honest CV + hedge, LB-noise aware) ─► kg-postmortem (lessons for next time)
```

Each workspace gets a `CLAUDE.md` with non-negotiable rules (frozen folds, log every run, one
change per experiment, trust CV and check it against LB, no leakage, validate before
submitting) and a `.kaggle-gm/` state folder:

```
.kaggle-gm/competition.json   metric, direction, task, target, code-comp limits, deadline, budget
.kaggle-gm/ledger.jsonl       one line per experiment: CV, fold scores, params, git hash, folds hash, LB
.kaggle-gm/submissions.jsonl  every submission the hooks saw
artifacts/<exp_id>/           oof.npy + test.npy, the inputs for ensembling
```

## Hooks

| Event | Behaviour |
|---|---|
| SessionStart | Exports `KGKIT_HOME`; inside a workspace it injects a brief: competition facts, deadline countdown and phase, best CV/LB, CV↔LB correlation, recent experiments, submissions today, lessons from past competitions (`~/.kaggle-gm/lessons.md`) |
| UserPromptSubmit | Detects competition work and tells Claude which 1–3 plugin skills to load before it answers. In evals this took skill use on Kaggle questions from 0 % to 100 % |
| PreToolUse (Bash/PowerShell) | **Denies** commands that would print Kaggle credentials. **Denies** `kaggle competitions submit` when the file fails validation against `sample_submission` (header, rows, ids, NaNs, index column written by mistake). **Asks** when today's submission budget is used up |
| PostToolUse | Logs submissions and reminds Claude to fetch the public score and attach it to the ledger |

Hooks never block for any other reason. They exit 0 on internal errors, and Python only starts
for commands that mention Kaggle, which keeps overhead around 70 ms.

## `kgkit` toolkit

```bash
PYTHONPATH="$KGKIT_HOME" python -m kgkit <command>     # KGKIT_HOME is set by the SessionStart hook
```

| Command | Purpose |
|---|---|
| `init <slug> --metric auc --target y --id-col id [--code-competition --runtime-hours 9 --no-internet] --deadline YYYY-MM-DD` | create a workspace |
| `status` | dashboard: best CV/LB, CV↔LB correlation, recent experiments |
| `folds train.csv --target y [--group g] [--strategy ...]` | leak-free folds + fold report (group leakage check) |
| `eda train.csv --test test.csv --target y --id-col id` | markdown EDA with red flags (leaks, ID-like columns, drift, unseen categories, duplicates, row-order correlation) |
| `adv train.csv test.csv --drop id,y` | adversarial validation with drifting-feature importances |
| `ledger list|best|show|lb|add` | experiment ledger; `ledger lb <id> <public> [private]` |
| `blend --exp 3 7 9 --truth train.csv:y --folds folds.csv:fold --method hill --sample sample_submission.csv` | hill climbing / weights / rank blending with an honest nested score; warns if members used different folds |
| `validate sub.csv sample_submission.csv` | metric-aware submission validation |
| `metrics` / `score` / `vendor` | metric registry / scoring / copy kgkit into a project or Kaggle dataset |

Python API highlights: `kgkit.cv.assign_folds` (stratified / group / stratified-group /
multilabel), `time_series_splits(gap=h)`, `purged_kfold_splits`; `kgkit.metrics.get(name)`;
`kgkit.thresholds.OptimizedRounder`; `kgkit.ensemble.hill_climb / cv_blend_score / stack`;
`kgkit.adversarial.adversarial_validation`; `kgkit.features.oof_target_encode / lag_features`;
`kgkit.experiment.Ledger().log(...)`.

## Templates

| File | Highlights |
|---|---|
| `train_gbdt.py` | LightGBM / XGBoost / CatBoost / sklearn HGB; frozen folds, seed averaging, label-metric decoding fitted on OOF (threshold / argmax / optimised rounding), log-target, id-aligned submission, `--smoke` |
| `train_image.py` | timm; AMP (bf16/fp16), EMA, cosine warmup, head LR multiplier, hflip TTA; `--folds` lets you split folds across GPUs, and the run that completes the set logs the experiment |
| `train_transformer.py` | HF AutoModel with mean/attention/CLS pooling, layer-wise LR decay, gradient accumulation, several evaluations per epoch, length-sorted inference, offline-model friendly |
| `inference_kernel.py` + `kernel/*.json` | offline Kaggle kernel: runtime test discovery, offline wheels, time-budgeted ensemble members with fallback, output validation; metadata with internet off |

## Verification

- **63 pytest tests** (one runs only when LightGBM is installed) cover kgkit, all four templates end to end (tiny synthetic data, plus an
  offline tiny BERT), and the hooks driven through `run.sh` exactly as Claude Code calls them.
- **Live Claude Code sessions** confirmed the SessionStart brief and the credential guard
  (tested against a dummy `kaggle.json`).
- **Real-data dry run** on `playground-series-s6e2` (630k rows, no submission made): EDA 6 s,
  adversarial AUC 0.501, LightGBM CV AUC 0.95521, HGB 0.95490, hill-climb blend 0.95537
  (honest nested). The run surfaced 5 bugs, all fixed (see git history).
- **Plugin evals** (`claude plugin eval`, Haiku, 3 runs per arm): the with-plugin arm scored
  100 % on all 7 cases. `ensemble-hygiene` scored 1.00 with the plugin vs 0.00 without. The
  other cases pass with or without the plugin on Haiku, so they work as regression tests.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
claude plugin validate plugins/kaggle-grandmaster
cd plugins/kaggle-grandmaster && claude plugin eval . --runs 3 --model haiku
```

Repository layout:

```
.claude-plugin/marketplace.json        marketplace manifest (this repo is installable)
plugins/kaggle-grandmaster/
  .claude-plugin/plugin.json
  skills/<name>/SKILL.md               19 knowledge skills + 15 kg-* commands
  agents/*.md                          9 subagents
  hooks/                               hooks.json, run.sh launcher, stdlib-only Python hooks
  kgkit/                               the toolkit (python -m kgkit)
  templates/                           training / inference templates, CLAUDE.md template
  evals/                               plugin eval suite
tests/                                 pytest suite
```

## Limitations

- Only the user can accept competition rules, create API tokens and tick final submissions on
  the website. The plugin reminds you at the right moments.
- Submitting uses your quota and goes out to Kaggle. Claude confirms before submitting unless
  you have explicitly authorised autonomous submissions. `kg-grind` never submits on its own.
- The domain playbooks reflect competition practice up to 2026. Recon and the
  solution-researcher agent exist so that current competition-specific findings take priority.

## License

MIT
