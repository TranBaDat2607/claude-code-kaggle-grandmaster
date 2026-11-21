---
name: kg-start
description: Bootstrap a Kaggle competition workspace end to end — recon, data download, workspace init, EDA, adversarial validation, frozen folds, metric implementation, baseline model and a first validated submission file.
argument-hint: <competition-slug> [workspace-dir]
allowed-tools:
  - Read
  - Glob
  - Grep
  - Agent
  - 'Bash(kgkit init *)'
  - 'Bash(kgkit eda *)'
  - 'Bash(kgkit adv *)'
  - 'Bash(kgkit folds *)'
  - 'Bash(kgkit audit-repo *)'
  - 'Bash(kgkit backlog *)'
  - 'Bash(kgkit ledger *)'
  - 'Bash(kgkit validate *)'
  - 'Bash(kgkit status *)'
  - 'Bash(kaggle competitions files *)'
  - 'PowerShell(kaggle competitions files *)'
  - 'Bash(kaggle competitions download *)'
  - 'PowerShell(kaggle competitions download *)'
---

# /kg-start — bootstrap a competition

Run kgkit with the Bash tool as `kgkit <command>`: the plugin puts it on PATH, also on Windows (Git Bash),
and this command pre-approves the kgkit calls it makes. Older text that says `python -m kgkit` means the same
command. Submitting, pushing or publishing still asks.

Arguments: `$ARGUMENTS` (competition slug, optional workspace directory; default: current directory).

kgkit runs with the first real Python on PATH (`python`, then `python3`); `KGKIT_PYTHON=<path>` picks another.

Follow the `grandmaster-playbook` and `competition-recon` skills. Steps:

1. **Preflight**: `kaggle --version` (offer `pip install kaggle` if missing). Check auth by running
   `kaggle competitions files <slug>`; on 401/403 tell the user to create an API token and/or
   accept the competition rules on the website — do not proceed with downloads until fixed. Never
   print credentials.
2. **Workspace**: create/enter the directory; `git init` if not a repo.
3. **Recon** (in parallel with the download): launch the `competition-analyst` agent for the slug
   (and optionally `solution-researcher` for prior art). Meanwhile:
4. **Data**: `kaggle competitions download <slug> -p data/` and extract. List files and sizes.
5. **Init**: `kgkit init <slug> --metric <m> --task <t> --target <col> --id-col <id>
   [--code-competition --runtime-hours H --no-internet] --deadline <date>` using facts from recon
   (ask the user only for facts you cannot determine). It makes `.gitignore` cover `data/`, and adds it to a
   `.gitignore` that exists. Then run `kgkit audit-repo`. It says whether the GitHub remote is public
   and checks that nothing from the competition's files is in git. Kaggle's rules forbid publishing them, and
   that covers copies and subsets too. Init also makes `work/` (scratch: probes, logs; ignored) and `studies/`
   (one folder per analysis that decides something). Name the first phase:
   `kgkit ledger phase "baselines"`: every record carries the phase it ran in, and `/kg-showcase` tells
   the journey phase by phase when the competition ends.
6. **Metric**: write `src/metric.py` implementing the exact competition metric (reuse
   `kgkit.metrics` if identical) with a unit test against a hand-computed example.
7. **EDA & shift**: `kgkit eda data/train.csv --test data/test.csv --target <y> --id-col
   <id> --out reports/eda.md` and `kgkit adv ...`. For non-tabular data, launch the
   `data-detective` agent instead.
8. **Folds**: choose the scheme per `validation-strategy` (groups? time?) and write
   `data/folds.csv` with `kgkit folds ...`; show the fold report.
9. **Baseline**: adapt the matching template from `${CLAUDE_PLUGIN_ROOT}/templates/`
   (`train_gbdt.py`, `train_image.py`, `train_transformer.py`) into `src/`, run it quickly
   (reduced settings if large), log to the ledger, write `subs/<exp_id>.csv`, and validate it with
   `kgkit validate`. Tabular: also run `--model linear` and `--model mlp` on the same folds
   (diverse baselines, see `/kg-baseline`). Mark the main one `kgkit ledger decide <id> baseline`.
10. **Backlog**: add the recon's initial ideas with `kgkit backlog add ...` (gain, prob,
   cost, evidence, source) and `kgkit backlog render`.
11. **Commit** code, folds definition (or seed), CLAUDE.md, `.kaggle-gm/` (its `remote/` and `build/` are ignored). Never commit `data/`, or anything
   made from the competition's files outside it.

Finish with: recon TL;DR, CV scheme, baseline CV, the submission file ready (ask before
submitting, or run `/kg-submit`), and the top 5 backlog experiments.
