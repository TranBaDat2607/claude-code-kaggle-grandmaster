# Kaggle Grandmaster for Claude Code

**Turn Claude Code into a Kaggle teammate that works like a Grandmaster.**

Validation you can trust, every experiment logged and judged, honest ensembles, free Kaggle GPUs when you have
none, and final picks that hold up on the private leaderboard.

[![tests](https://github.com/TranBaDat2607/claude-code-kaggle-grandmaster/actions/workflows/tests.yml/badge.svg)](https://github.com/TranBaDat2607/claude-code-kaggle-grandmaster/actions/workflows/tests.yml)
![Claude Code plugin](https://img.shields.io/badge/Claude%20Code-plugin-d97757)
![Kaggle](https://img.shields.io/badge/Kaggle-competitions-20beff)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

```bash
/plugin install kaggle-grandmaster@claude-plugins-official
/kaggle-grandmaster:kg-start playground-series-s6e2
```

That second line goes from a competition name to a validated first submission: recon, data, EDA, adversarial
validation, frozen CV folds and a baseline.

## Why use it

- **It catches the mistakes that cost medals.** Leakage across folds, tuning on the public LB, blending predictions
  made on different folds, notebooks that time out on the hidden test set. Hooks block invalid submission files
  and any command that would print your Kaggle key, and ask before a repeated file or one over today's budget.
- **Experiments, not guesses.** One change per run, logged to a ledger, compared fold by fold with the accepted
  baseline, then kept or discarded.
- **A repo worth making public.** Every experiment keeps its phase, its command and the exact code it ran from.
  When the competition ends, `/kg-showcase` gives the repo two ways in: `solution/` for readers who want the
  result, and `journey/` for readers who want to see how the ideas developed, dead ends included.
- **Free GPUs from Kaggle.** `/kg-gpu` runs your local training script on Kaggle's GPUs and TPUs. It keeps both T4s
  busy, resumes runs that time out, stops losing ideas early and budgets your weekly quota.
- **Your Kaggle runs, beside the chat.** `/kaggle-monitor` shows each running kernel's live CPU, GPU, RAM and
  training progress, your submission scores, and the GPU quota of every account you register.
- **Any competition.** Tabular, computer vision, NLP and LLMs, time series, audio, recommender systems, simulation
  and optimisation.

## Install

From the official Claude plugin directory:

```bash
/plugin install kaggle-grandmaster@claude-plugins-official
```

Or from this repository:

```bash
/plugin marketplace add TranBaDat2607/claude-code-kaggle-grandmaster
/plugin install kaggle-grandmaster@claude-code-kaggle-grandmaster
```

**You need** Python 3.9+ with `numpy pandas scikit-learn scipy`, the Kaggle CLI (`pip install kaggle`) and a Kaggle
API token. Every merge to `main` is a release. A marketplace added from this repo doesn't update by itself: turn on
auto-update under `/plugin` → **Marketplaces**.

## Commands

| Command | What it does |
|---|---|
| `/kg-start <slug>` | New competition: recon, data, EDA, frozen folds, baseline and first submission |
| `/kg-adopt <slug>` | Already weeks in? Adopts your repo as it is and imports your submission history |
| `/kg-experiment` · `/kg-grind` | One experiment, or a loop of them, each judged against the baseline |
| `/kg-gpu` | Train on Kaggle's free GPUs and TPUs within your weekly quota |
| `/kg-ensemble` | Hill climbing and stacking on same-fold predictions, with an honest score |
| `/kg-submit` | Validate, check today's budget, submit, then record the leaderboard score |
| `/kg-kernel` | Package an offline notebook for a code competition |
| `/kg-final` | Choose the final submissions most likely to hold up on the private leaderboard |
| `/kaggle-monitor` | Open the Kaggle pane |

Also `kg-recon`, `kg-eda`, `kg-cv`, `kg-baseline`, `kg-status`, `kg-debug`, `kg-ideas`, `kg-postmortem` and
`kg-showcase`. In Claude Code, commands are written with the plugin's name (`/kaggle-grandmaster:kg-status`); the
short form works when no other plugin uses the name.

## What's inside

**22 playbook skills**, Claude loads them on its own when relevant · **18 slash commands** · **9 specialist agents** ·
**5 safety hooks** · **`kgkit`**, a tested Python toolkit (folds, 35+ metrics, ensembling, experiment ledger, Kaggle
GPU pipeline) · **training templates** for GBDTs, timm and Hugging Face transformers · **the Kaggle pane**.

## Your data and keys

- No server of its own, and it collects nothing. It talks to Kaggle under your account through the official CLI,
  plus one read-only GitHub check of whether your repo is public.
- The plugin never uses or sends your API key, and commands that would print it are blocked.
- Claude asks before each submission and each GPU push, unless you have authorised it to go ahead.

## Learn more

The [full reference](docs/reference.md) covers the workflow, each hook, every `kgkit` command, the templates, GPU
training, testing, and exactly what goes over the network.

**Develop:** `pip install -r requirements-dev.txt`, `python -m pytest tests -q -n auto`,
`claude plugin validate plugins/kaggle-grandmaster`.

## License

MIT
