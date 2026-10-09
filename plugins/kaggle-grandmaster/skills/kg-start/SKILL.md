---
name: kg-start
description: Bootstrap a Kaggle competition workspace end to end — recon, data download, workspace init, EDA, adversarial validation, frozen folds, metric implementation, baseline model and a first validated submission file.
argument-hint: <competition-slug> [workspace-dir]
allowed-tools: [Bash, PowerShell, Read, Write, Edit, Glob, Grep, WebFetch, WebSearch, Agent]
---

# /kg-start — bootstrap a competition

Arguments: `$ARGUMENTS` (competition slug, optional workspace directory; default: current directory).

kgkit: use the absolute path from the session context ("kgkit home"), else `$KGKIT_HOME`, else
`${CLAUDE_PLUGIN_ROOT}`. Invoke as `PYTHONPATH=<kgkit home> python -m kgkit <cmd>`
(PowerShell: `$env:PYTHONPATH="<kgkit home>"; python -m kgkit <cmd>`). Use `python3` if `python`
is not the right interpreter on this machine.

Follow the `grandmaster-playbook` and `competition-recon` skills. Steps:

1. **Preflight**: `kaggle --version` (offer `pip install kaggle` if missing). Check auth by running
   `kaggle competitions files <slug>`; on 401/403 tell the user to create an API token and/or
   accept the competition rules on the website — do not proceed with downloads until fixed. Never
   print credentials.
2. **Workspace**: create/enter the directory; `git init` if not a repo.
3. **Recon** (in parallel with the download): launch the `competition-analyst` agent for the slug
   (and optionally `solution-researcher` for prior art). Meanwhile:
4. **Data**: `kaggle competitions download <slug> -p data/` and extract. List files and sizes.
5. **Init**: `python -m kgkit init <slug> --metric <m> --task <t> --target <col> --id-col <id>
   [--code-competition --runtime-hours H --no-internet] --deadline <date>` using facts from recon
   (ask the user only for facts you cannot determine).
6. **Metric**: write `src/metric.py` implementing the exact competition metric (reuse
   `kgkit.metrics` if identical) with a unit test against a hand-computed example.
7. **EDA & shift**: `python -m kgkit eda data/train.csv --test data/test.csv --target <y> --id-col
   <id> --out reports/eda.md` and `python -m kgkit adv ...`. For non-tabular data, launch the
   `data-detective` agent instead.
8. **Folds**: choose the scheme per `validation-strategy` (groups? time?) and write
   `data/folds.csv` with `python -m kgkit folds ...`; show the fold report.
9. **Baseline**: adapt the matching template from `${CLAUDE_PLUGIN_ROOT}/templates/`
   (`train_gbdt.py`, `train_image.py`, `train_transformer.py`) into `src/`, run it quickly
   (reduced settings if large), log to the ledger, write `subs/<exp_id>.csv`, and validate it with
   `python -m kgkit validate`. Tabular: also run `--model linear` and `--model mlp` on the same folds
   (diverse baselines, see `/kg-baseline`). Mark the main one `python -m kgkit ledger decide <id> baseline`.
10. **Backlog**: add the recon's initial ideas with `python -m kgkit backlog add ...` (gain, prob,
   cost, evidence, source) and `python -m kgkit backlog render`.
11. **Commit** code, folds definition (or seed), CLAUDE.md, `.kaggle-gm/`.

Finish with: recon TL;DR, CV scheme, baseline CV, the submission file ready (ask before
submitting, or run `/kg-submit`), and the top 5 backlog experiments.
