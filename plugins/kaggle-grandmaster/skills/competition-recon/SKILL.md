---
name: competition-recon
description: How to analyse a Kaggle competition before modelling — metric, data, rules, timeline, public notebooks, discussions, leaks, past similar competitions and shake-up risk. Use at the start of a competition, when joining late, or when the user asks to research/understand a competition.
---

# Competition Recon

Goal: in a few hours, know more about this competition than 90% of participants. Produce
`reports/recon.md` with the sections below and fill `.kaggle-gm/competition.json`.

## 1. Pull the facts

```bash
kaggle competitions list -s "<keywords>"           # find the slug
kaggle competitions files <slug>                   # file list + sizes
kaggle competitions download <slug> -p data/ && unzip -q -o data/<slug>.zip -d data/
kaggle competitions leaderboard <slug> --show | head -30
kaggle kernels list --competition <slug> --sort-by voteCount --page-size 20
```

Use WebFetch on `https://www.kaggle.com/competitions/<slug>/overview`, `/data`, `/rules`,
`/discussion?sort=votes` (pages may need the user's browser if blocked; ask them to paste
the Overview/Evaluation/Data text if fetching fails). Record:

| Item | Why it matters |
|---|---|
| **Exact metric** + formula + any host code | You must reproduce it exactly (`src/metric.py`) |
| Prediction target and granularity | Defines the modelling unit and the CV group |
| Train/test split method (random? by time? by group/site/patient?) | Dictates the CV scheme |
| Public/private split size and how it's made | Shake-up risk; how much to trust public LB |
| Code competition? runtime, GPU/CPU, internet off? | Drives model size choices from day 1 |
| Submission limits (per day, final picks) | Plan the submission budget |
| External data / pretrained model rules; team size; merger deadline | Avoid disqualification |
| Timeline: entry/merger deadline, final deadline (UTC) | Back-plan the endgame |
| Prize / licence requirements (code must be open-sourced?) | Library/model licence choices |

## 2. Understand the data

- Read the data description line by line; map each file and column to a real-world meaning.
- How was the test set constructed? Is there a hidden test set larger than the visible
  one (code competitions often re-run on hidden data — never hardcode test ids)?
- Look at raw samples (not just summary stats). Run `python -m kgkit eda` and
  `python -m kgkit adv` (adversarial validation) on day 1.

## 3. Mine prior art (highest ROI hour of the competition)

- **Top-voted public notebooks**: note their CV scheme, CV and LB scores, key features, model
  choices. The best public notebook is your baseline-to-beat, not your solution.
- **Discussion sorted by votes and by recent**: data issues, leaks, metric quirks, "CV vs LB"
  threads, host clarifications. Re-check every few days.
- **Similar past competitions**: search "kaggle <domain> competition winning solution",
  Kaggle discussion write-ups ("1st place solution"), and `farid.one/kaggle-solutions`-style
  indexes. Extract the techniques that repeatedly won in this domain. Delegate to the
  `solution-researcher` agent for a structured digest.
- Papers / SOTA for the underlying task (Papers with Code, arXiv) when the problem is a
  well-studied academic task.

## 4. Leak & quirk hunting checklist

- IDs or file names that correlate with the target or encode time/order.
- Row order in train/test files; duplicated rows between train and test.
- Timestamps/metadata (image EXIF, file sizes, sampling rates) correlating with labels.
- Target derivable from other columns (sums, ratios) or from public external data.
- Test-set structure (e.g., known number of positives per group, sorted ordering).
Exploiting a leak is allowed unless the rules prohibit it — but report findings honestly to
the user; hosts sometimes fix leaks mid-competition, so never depend on one exclusively.

## 5. Shake-up risk assessment

Estimate: public test size (rows/groups), metric noise (AUC on 1k positives is stable;
F1 on 200 positives is not), train/test drift (adversarial AUC), and whether private data
is from a different time/source. High risk ⇒ weight CV more, prefer robust ensembles,
choose one "CV-best" and one "LB-best/hedge" final.

## 6. Recon report template (`reports/recon.md`)

```
# <slug> — recon
## TL;DR (5 bullets: what wins here, biggest risks, first 3 experiments)
## Metric (formula, direction, implementation notes, optimal post-processing)
## Data (files, unit of prediction, size, key columns, split construction)
## Rules & constraints (code comp/runtime/internet/external data/limits/deadlines)
## Validation plan (scheme, groups, why it mirrors the test split)
## Prior art (best public notebooks + scores, key discussion insights, similar past comps & winning ideas)
## Leaks / quirks
## Shake-up risk (low/med/high + reasoning)
## Initial backlog (ranked ideas)
```

Then: `python -m kgkit init <slug> --metric <m> --task <t> --target <col> --id-col <id> [--code-competition --runtime-hours 9 --no-internet] --deadline YYYY-MM-DD`.
