---
name: kg-showcase
description: After a competition ends — turn the work into a public repository with two ways in. solution/ for readers who want the result (the final pipeline and the exact commands), journey/ for readers who want to learn from it (how the ideas developed, phase by phase, written from the ledger, with links to the code each experiment ran). Tidies the tree, rebuilds the README, adds a license and figures, audits a public notebook and weights, drafts a Kaggle write-up and tags the final state. Asks before every public step.
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
  - 'Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit backlog *)'
  - 'Bash(python -m kgkit backlog *)'
  - 'PowerShell(python -m kgkit backlog *)'
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

The public repository has two ways in, for two readers:

- **`solution/`**: the result. The final pipeline end to end, the exact commands that reproduce the finals, the
  hardware and time. For a reader who only wants the best result.
- **`journey/`**: how the ideas developed. Phase by phase: what was tried, what the numbers said, what was kept,
  the dead ends and why they failed, with a link to the code each experiment ran from. For a reader who wants to
  learn from it. Most winners' repositories show only the first; the second is where people learn the most.

The README is the landing page that sends each reader to theirs. Nothing in history is rewritten: the tree at
the end of the competition stays one tag away (`<slug>-workbench`).

What goes public stays public: a notebook, dataset or repo can be forked, cached or indexed within minutes.
Prepare everything locally and show the user exactly what will become public. Then ask before each public step:
making a notebook or dataset public, pushing a branch, a tag or the snapshot refs, opening a PR, changing the repo's
visibility. Never rewrite history, force-push or change visibility without their explicit go-ahead.

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

## 2. Freeze the workbench
Before anything moves, tag the tree as it was at the end of the competition, locally:
`git tag -a <slug>-workbench -m "the working repository at the end of <slug>"`. Every later step is an ordinary
commit on a branch; the journey's links point at commits, so moving or deleting a file never breaks them.

## 3. The journey
1. **Phases.** `python -m kgkit ledger phase` lists the phases the records carry. Records with no phase (logged
   before phases were named, or in an adopted repo): propose phase boundaries from the record dates and notes, the
   backlog and the commit messages (`git log --reverse --format="%ad %h %s" --date=short`), as a short list of
   `<name>: <first id>..<last id>` with one line on what the phase was about. Show it to the user, then apply it with
   `python -m kgkit ledger phase "<name>" --records <a>..<b>`. Four to eight phases read well.
2. **Decisions.** Records with no decision are left out of the lineage graph. For the ones the story needs (each
   phase's main experiments), propose a decision from the record's notes and the comparisons
   (`.kaggle-gm/comparisons.jsonl`), and record it with the user's agreement
   (`python -m kgkit ledger decide <ids> keep|discard|inconclusive "<why>"`; a range `12..20` decides a sweep).
3. **The generated half.** `python -m kgkit ledger journey` writes `journey/experiments.md`: every record by phase,
   its parent, the change, its scores, the decision and a link to the code it ran from (and a diff against its
   parent's), plus the lineage as a Mermaid graph. Read what it prints: approximate links (records logged from
   uncommitted changes before snapshots existed), snapshots that need pushing, records still without a phase.
4. **The story.** Write `journey/README.md` with [journey-template.md](journey-template.md), in the user's voice
   and with their numbers. Sources, all local: the ledger (notes and decision notes), `journey/experiments.md`, the
   backlog with its evidence and results (`python -m kgkit backlog list --all`), the commit messages, the notes in
   the workspace `CLAUDE.md`, `studies/*/README.md`, `reports/` (post-mortem, final selection), the figures of
   step 1. Every number comes from these; every claim links to the record, the study or the code. Dead ends get as
   much room as successes: what was expected, what happened, why it failed.

   Show the user the draft and have them correct the story: they know why they tried things.

## 4. The solution
Write `solution/README.md` with [solution-template.md](solution-template.md): the final pipeline, for a reader who
wants the result only.
- **What the finals are.** `python -m kgkit ledger lineage <final id>` for each final: the chain of kept changes
  that built it. `python -m kgkit ledger show <id>` for the members and parameters of a blend or a stack; the final
  kernel's `kernel-metadata.json` for its inputs.
- **How to reproduce them, as commands.** The records' `cmd` (the command line each one ran), in pipeline order:
  prepare the data, train each member (with its folds and seeds), blend or stack, predict. A record with no `cmd`
  (logged by hand, or before kgkit recorded commands): reconstruct it from the code at its commit and its params, and
  say it was reconstructed.
- **Hardware and time.** The records' `runtime_s`, the GPU hours in `.kaggle-gm/gpu_runs.jsonl` and in the records
  of remote runs, the inference time of the final kernel in a code competition.
- **Environment.** Pin what the pipeline imports: the installed versions (`python -m pip freeze`) of the packages
  imported by the scripts in the commands above, in `solution/requirements.txt` (or an existing `environment.yml`
  or `pyproject.toml` the repo already uses).
- **A pipeline diagram**, as a Mermaid flowchart in the README (GitHub renders it): inputs → features / labels →
  models → blend → submission.

## 5. Tidy the tree
The tree should read as a project, not a working folder. List every tracked file (`git ls-files`) and propose a home
for each, as a table grouped by folder:
- **keep**: the pipeline in `src/`, the studies, numbered notebooks, the kernels the solution or the journey cite,
  tests, docs, reports.
- **move**: an analysis the journey cites → `studies/<name>/` with a short README (question, result, the records it
  decided); a vendored kgkit in `src/kgkit` → `third_party/kgkit` (scripts then find it there: the templates look
  in `third_party/`); scripts scattered by prefix (`lb_*.py`, `tilt_*.py`) → a folder per stage or study.
- **remove**: copies of scripts (`train_v2.py`), one-off probes, scratch notebooks. Their code stays reachable: the
  journey's links point at the commits and snapshots that ran them, and `<slug>-workbench` has the whole tree.
- **untrack** (`git rm --cached`, then ignore): logs, `.kaggle-gm/remote/` and `.kaggle-gm/build/`, generated
  files, anything the audit flags.

`python -m kgkit status` prints the layout drift it sees. Show the plan; move nothing until the user approves it.
Then `git mv` (history follows the file), fix the imports and paths the moves break, and run the tests and a
`--help` of each entry point.

## 6. README: the landing page
Rebuild `README.md` with [readme-template.md](readme-template.md). The first screen says what the competition was,
how it went, what is the user's own work, and sends each reader on: **Final solution → `solution/`**, **How it
developed → `journey/`**. Keep what is still true in the old README, and move long result tables to
`docs/results.md` instead of deleting them. Write numbers as the commands print them, with the date the standings
were read.

## 7. License
- Ask which license the user wants for their own code (MIT is a common default), then add `LICENSE`.
- Read the rules (`python -m kgkit pages --page rules`). A prize winner usually owes the winning code under the
  license the rules name; quote that clause.
- Each public notebook, dataset and model the solution uses keeps its own license: list them in the README's
  License section. Non-commercial or share-alike ones (CC BY-NC, CC BY-SA) cannot be relicensed under MIT.
- Competition data is never ours to license or publish (step 11 checks the repo).

## 8. Figures
`ledger predictiveness --plot docs/figures` writes `public_vs_private.png`, `private_over_time.png` and
`cv_vs_private.png` (it needs matplotlib, and says so if it is missing). Put one in the README and the score over
time in `journey/README.md`, with alt text that states what it shows ("private 0.015 below public for every
submission"), not "chart".

## 9. Public notebook and weights
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

## 10. Kaggle write-up draft and stale notes
- Write `reports/kaggle-writeup.md` with [writeup-template.md](writeup-template.md), in the user's voice and with
  their numbers. Read the top write-ups of this competition first (`kgkit discussions solutions`) for its
  conventions. Link the repository's `solution/` and `journey/`. Posting is website-only, in the competition's
  Writeups or Discussion section (`https://www.kaggle.com/competitions/<slug>/writeups`): the Kaggle CLI cannot
  post, and never post on the user's behalf. If Claude Code ran part of the work, offer a line saying what it did and
  what the user decided. Whether to include it is the user's call.
- Update everything that still describes the competition as running, or contradicts `kgkit final`: the notes in
  the workspace `CLAUDE.md` (the competition is over, the result, where to start reading),
  `reports/final-selection.md` (append the outcome), any README or doc line that states a final pick or a score
  that the output of step 1 contradicts.

## 11. Check from a fresh clone, audit, then publish
- Commit the changes on a branch. Clone the branch into a scratch folder (`git clone --branch <branch> . <dir>`;
  `--no-local` when the repository is on a network drive) and check it as a reader would: every relative link in
  `README.md`, `solution/README.md` and `journey/README.md` opens a file of the clone; `python -m compileall -q` over
  the code folders; the tests; `--help` of each entry point in the solution's commands. Fix what fails.
- `python -m kgkit audit-repo` checks every branch, every tag, the snapshot refs (`refs/kg/*`, the code of each
  experiment), the history, uncommitted files, and whether the GitHub remote is public. On a LEAK, stop and show
  the user the steps it prints.
- Tag the final state locally: `git tag -a <slug>-final -m "<one-line result>"`.
- Ask before each push: the branch, the tags (`<slug>-workbench`, `<slug>-final`), and the snapshot refs, which the
  journey's code links need: `git push <remote> 'refs/kg/*:refs/kg/*'`. Then ask before opening a PR, and before
  making the repo public (`gh repo edit --visibility public --accept-visibility-change-consequences`).

## 12. Report
- What changed locally: the two ways in, what moved where.
- What is public now, with links.
- What is left for the user: post the write-up on the website, and any public step they chose to do themselves.
