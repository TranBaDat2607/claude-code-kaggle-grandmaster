---
name: kg-showcase
description: After a competition ends — turn the work into a public portfolio. Rebuilds the README around the result, separates own work from public work, adds a license and figures, audits a public notebook and weights, drafts a Kaggle write-up and tags the final state. Asks before every public step.
argument-hint: "[competition-slug]"
allowed-tools:
  - Read
  - Glob
  - Grep
  - Agent
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit final *)'
  - 'Bash(python -m kgkit final *)'
  - 'PowerShell(python -m kgkit final *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit ledger *)'
  - 'Bash(python -m kgkit ledger *)'
  - 'PowerShell(python -m kgkit ledger *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit audit-repo *)'
  - 'Bash(python -m kgkit audit-repo *)'
  - 'PowerShell(python -m kgkit audit-repo *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit pages *)'
  - 'Bash(python -m kgkit pages *)'
  - 'PowerShell(python -m kgkit pages *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit discussions *)'
  - 'Bash(python -m kgkit discussions *)'
  - 'PowerShell(python -m kgkit discussions *)'
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit status *)'
  - 'Bash(python -m kgkit status *)'
  - 'PowerShell(python -m kgkit status *)'
  - 'Bash(kaggle competitions leaderboard *)'
  - 'PowerShell(kaggle competitions leaderboard *)'
  - 'Bash(kaggle datasets files *)'
  - 'PowerShell(kaggle datasets files *)'
  - 'Bash(kaggle kernels status *)'
  - 'PowerShell(kaggle kernels status *)'
disable-model-invocation: true
---

# /kg-showcase

Run kgkit with the Bash tool, as `PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit <command>`: this command
pre-approves that form for the kgkit calls it makes. Submitting, pushing or publishing still asks (so does a
PowerShell `$env:` assignment).

Competition: `$ARGUMENTS` (default: from `.kaggle-gm/competition.json`). Best run after `/kg-postmortem`.

What goes public stays public: a notebook, dataset or repo can be forked, cached or indexed within minutes.
Prepare everything locally and show the user exactly what will become public. Then ask before each public step:
making a notebook or dataset public, pushing a branch or tag, opening a PR, changing the repo's visibility. Never
rewrite history, force-push or change visibility without their explicit go-ahead.

## 1. Facts, from commands only
- `python -m kgkit final`: rank, number of teams, top %, the medal zone (computed from Kaggle's table, say so),
  the winning score, the finals (or the pool Kaggle picked from when none was selected) and their private scores.
- `python -m kgkit ledger sync-lb`, then `python -m kgkit ledger predictiveness --plot docs/figures`: every score on
  record, the figures, and what predicted private. If `reports/postmortem.md` does not exist, also do
  `/kg-postmortem`'s steps 4–5 for "what the winners did".
- Own work vs public work. Sources:
  - The ledger (`python -m kgkit ledger list -n 500`): records tagged `lb-only` or `lb-blend`, or imported with
    `--source notebook`, hold other people's work.
  - `ref/*/REVIEW.md`: public notebooks studied with `kgkit kernels pull`, with their authors.
  - The final notebook's `kernel-metadata.json`: `dataset_sources`, `kernel_sources` and `model_sources` list
    every input it attaches, so every author to credit.

  Show the user the split and have them confirm it. Never guess who wrote what.
- Quote scores, ranks and dates from this output. Never retype them from memory or older docs.

## 2. README around the result
Rebuild `README.md` with [readme-template.md](readme-template.md). The first screen says what the competition
was, how it went and what is the user's own work. Keep what is still true in the old README, and move long
result tables to `docs/results.md` instead of deleting them. Write numbers as the commands print them, with the
date the standings were read.

## 3. License
- Ask which license the user wants for their own code (MIT is a common default), then add `LICENSE`.
- Read the rules (`python -m kgkit pages --page rules`). A prize winner usually owes the winning code under the
  license the rules name; quote that clause.
- Each public notebook, dataset and model the solution uses keeps its own license: list them in the README's
  License section. Non-commercial or share-alike ones (CC BY-NC, CC BY-SA) cannot be relicensed under MIT.
- Competition data is never ours to license or publish (step 8 checks the repo).

## 4. Figures
`ledger predictiveness --plot docs/figures` writes `public_vs_private.png`, `private_over_time.png` and
`cv_vs_private.png` (it needs matplotlib, and says so if it is missing). Put one in the README, with alt text
that states what it shows ("private 0.015 below public for every submission"), not "chart".

## 5. Public notebook and weights
Audit before anything goes public on Kaggle, and show the user the findings:
- **Inputs** of the notebook (`kernel-metadata.json`). A private dataset or model breaks the public notebook for
  readers: publish it first or drop it. Other users' inputs: credit them.
- **Saved outputs** in the `.ipynb`: API keys, tokens, `kaggle.json` paths, absolute paths with a username. Also
  printed competition data beyond what the competition itself shows on Kaggle.
- **A weights dataset** (`kaggle datasets files <owner>/<slug>`, or the local upload folder): weights and the code
  to load them, nothing else. No competition data: no CSV, parquet, npy or image files, no cached or pre-decoded
  inputs, no OOF or label tables keyed by the competition's IDs. No credentials, no logs.
- **Checkpoints**: load each with `torch.load(path, map_location="cpu", weights_only=True)`. One that only loads with
  `weights_only=False` pickles objects. Such a file runs code on load for every reader, and can carry label
  tables, optimizer state or a config with local paths. Re-save it as a plain `state_dict` or as safetensors.
- `dataset-metadata.json`: a description that says what the files are and how they were trained, and a license.

Then, only on the user's go-ahead:
- An existing notebook or dataset is made public from its page on the Kaggle website: no new version, no re-run.
  For a notebook that already ran, prefer this. `kaggle kernels push` with `"is_private": false` creates and runs
  a new version, which spends quota and can fail.
- A new weights dataset: `kaggle datasets create -p <dir>` (private by default). The user checks it on the
  website, then makes it public there.

## 6. Kaggle write-up draft
Write `reports/kaggle-writeup.md` with [writeup-template.md](writeup-template.md), in the user's voice and with
their numbers. Read the top write-ups of this competition first (`kgkit discussions solutions`) for its
conventions. Posting is website-only, in the competition's Writeups or Discussion section
(`https://www.kaggle.com/competitions/<slug>/writeups`): the Kaggle CLI cannot post, and never post on the
user's behalf. If Claude Code ran part of the work, offer a line saying what it did and what the user decided.
Whether to include it is the user's call.

## 7. Stale notes
Update everything that still describes the competition as running, or contradicts `kgkit final`:
- The notes in the workspace `CLAUDE.md`: the competition is over, the result, where to start reading.
- `reports/final-selection.md`: append the outcome.
- Any README or doc line that states a final pick or a score that the output of step 1 contradicts.

## 8. Audit, then tag
- `python -m kgkit audit-repo` checks every branch, the history, uncommitted files, and whether the GitHub
  remote is public. On a LEAK, stop and show the user the steps it prints.
- Commit the changes on a branch, and tag the final state locally:
  `git tag -a <slug>-final -m "<one-line result>"`.
- Ask before pushing the branch and the tag, before opening a PR, and before making the repo public
  (`gh repo edit --visibility public --accept-visibility-change-consequences`).

## 9. Report
- What changed locally.
- What is public now, with links.
- What is left for the user: post the write-up on the website, and any public step they chose to do themselves.
