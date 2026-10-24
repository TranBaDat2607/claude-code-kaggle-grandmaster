# Kaggle Grandmaster for Claude Code

**A Claude Code plugin that helps you win Kaggle competitions.** Install it and Claude works the way a
Kaggle Grandmaster does. It builds validation you can trust, runs and logs many small experiments,
ensembles honestly, packages offline notebooks for code competitions and picks final submissions
that hold up on the private leaderboard.

[![tests](https://github.com/TranBaDat2607/claude-code-kaggle-grandmaster/actions/workflows/tests.yml/badge.svg)](https://github.com/TranBaDat2607/claude-code-kaggle-grandmaster/actions/workflows/tests.yml)
![Claude Code plugin](https://img.shields.io/badge/Claude%20Code-plugin-d97757)
![Kaggle](https://img.shields.io/badge/Kaggle-competitions-20beff)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

It covers every type of Kaggle competition: **tabular** (LightGBM, XGBoost, CatBoost, Playground
Series), **computer vision** (classification, detection, segmentation, medical imaging), **NLP and
LLMs** (DeBERTa, LoRA fine-tuning, vLLM inference), **time series and forecasting**, **audio and
biosignals**, **recommender systems and ranking**, **simulation agents** and **combinatorial
optimisation**.

## Install

Pick one of the two options.

**Option A: from the official Claude plugin directory**

```bash
/plugin install kaggle-grandmaster@claude-plugins-official
```

You can also browse to it with `/plugin` → **Discover**, or add it from the plugin directory on claude.ai.

**Option B: from this repository**

```bash
/plugin marketplace add TranBaDat2607/claude-code-kaggle-grandmaster
/plugin install kaggle-grandmaster@claude-code-kaggle-grandmaster
```

From a shell, `claude plugin marketplace add ...` and `claude plugin install ...` do the same thing.

**Updates.** Every merge to `main` releases a new version. Marketplaces other than the official one don't
auto-update by default, so run `/plugin` → **Marketplaces** → **Enable auto-update** once, or update by
hand with `claude plugin marketplace update claude-code-kaggle-grandmaster` and
`claude plugin update kaggle-grandmaster@claude-code-kaggle-grandmaster`.

**Permissions.** The `kg-*` commands don't pre-approve any shell, file-write or web tools, so
Claude asks before each one according to your own permission settings. If you want `/kg-grind` to
run hands-off, allow the toolkit in your workspace's `.claude/settings.json`:

```json
{ "permissions": { "allow": ["Bash(python -m kgkit:*)", "Edit(./src/**)", "Write(./src/**)"] } }
```

Kaggle submissions and GPU pushes still go through the plugin's hooks, which check them either way.

## Quick start

Inside Claude Code, start a competition:

```bash
/kaggle-grandmaster:kg-start playground-series-s6e2
```

`kg-start` runs competition recon, downloads the data, sets up a workspace, runs EDA and
adversarial validation, freezes the CV folds, trains a baseline and writes a validated
submission file.

Already weeks into a competition with your own repo? Adopt it in place instead:

```bash
/kaggle-grandmaster:kg-adopt <competition-slug>
```

`kg-adopt` writes only `.kaggle-gm/` (the competition facts), imports your past Kaggle submissions
and the experiments your repo already tracks into the ledger, and marks the accepted baseline. Your
layout, folds and scripts stay as they are. The session brief suggests it when a repo has
`kernel-metadata.json` files or a competition URL but no workspace.

**Requirements:** [Claude Code](https://claude.com/claude-code); Python ≥ 3.9 with
`numpy pandas scikit-learn scipy` (LightGBM, PyTorch and similar libraries are only needed for the
matching templates); `pip install kaggle` and a Kaggle API token; a POSIX `sh` for the hooks (Git
Bash on Windows, which Claude Code already uses). To try the plugin without installing it, clone
the repo and run `claude --plugin-dir plugins/kaggle-grandmaster`.

Plugin commands are namespaced (`/kaggle-grandmaster:kg-status`). Plain `/kg-status` also works
when no other plugin uses the same name.

## Who is this for?

- Kaggle competitors who use Claude Code as an AI pair-programmer and want it to follow a winning
  process, not just write models.
- Anyone who wants an AI agent that catches the classic mistakes before they cost a medal:
  target leakage, group or time leakage in cross-validation, tuning on the public LB, blending
  OOFs from different folds, notebooks that time out on the hidden test set.
- Data scientists who want a reusable, tested toolkit for CV folds, metrics, ensembling and
  experiment tracking (`kgkit`), with or without Claude.

---

## What's inside

| Layer | Count | What it does |
|---|---|---|
| **Knowledge skills** | 21 | Playbooks Claude loads when relevant: master playbook, recon, validation, tabular, CV, NLP/LLM, time series, audio, simulation/optimisation, recsys, ensembling, metrics, tuning, DL training, boosters, code competitions, Kaggle CLI, Kaggle GPUs, CV-LB debugging, final selection, and competition strategy (choosing competitions, solo gold, teaming and merging) |
| **Slash commands** | 17 | `kg-start`, `kg-adopt`, `kg-recon`, `kg-eda`, `kg-cv`, `kg-baseline`, `kg-experiment`, `kg-grind`, `kg-gpu`, `kg-ensemble`, `kg-submit`, `kg-status`, `kg-kernel`, `kg-final`, `kg-debug`, `kg-ideas`, `kg-postmortem` |
| **Specialist agents** | 9 | competition-analyst, solution-researcher, data-detective, validation-auditor, feature-engineer, experiment-runner, error-analyst, ensemble-architect, kernel-packager |
| **Hooks** | 4 | session brief (incl. GPU quota), Kaggle-prompt → skill router, submission + GPU-push + upload guard (validation, repeated files, daily budget, weekly GPU quota, credential protection), submission / GPU-run logger |
| **`kgkit` toolkit** | 23 modules | Tested Python: leak-free folds, 35+ metrics, threshold/rounding optimisers, hill climbing / multi-level and residual stacking / library pruning, adversarial validation, leak-safe features, brute-force feature search, EDA red flags, experiment ledger with decisions + lineage, paired experiment comparison, fresh-split recheck, idea backlog, public-notebook review, competition pages and discussions, submission validation, Kaggle GPU/TPU pipeline, dataset download with per-file fallback, upload and third-party-text safety |
| **Templates** | 5 + metadata | Tabular (LightGBM/XGBoost/CatBoost/HGB + linear/kNN/SVM/MLP), timm image models, HF transformers (both resumable), offline inference kernel, remote GPU training runner |
| **Evals** | 15 cases | `claude plugin eval` suite with a no-plugin baseline arm |

## How a competition flows

```
kg-start ─► recon (competition-analyst ∥ solution-researcher) ─► data ─► kgkit init
        ─► EDA + adversarial validation ─► frozen folds ─► metric impl ─► baseline + 1st submission
                                   │
             ┌─────────────────────┘
             ▼
   kg-experiment / kg-grind  ◄── kg-ideas → kgkit backlog (error-analyst, public notebooks, prior art, levers)
   (one change · frozen folds · ledger · paired compare vs accepted baseline · decide keep/discard)
   tabular: kgkit features search (thousands of candidates, batch-screened, rechecked on a fresh split)
   kg-gpu: no local GPU? 1-fold screens on Kaggle (2 per T4x2 session, pruned) ─► full-fold runs
             │
             ├─► kg-submit (validated, budgeted, LB attached to ledger) ─► CV↔LB correlation
             ├─► kg-debug when CV and LB disagree (validation-auditor)
             ▼
   kg-ensemble (same-fold OOFs, prune, hill climbing / multi-level / residual stack, honest score) ─► kg-kernel
             ▼
   kg-final (best honest CV + hedge, LB-noise aware) ─► kg-postmortem (lessons for next time)
```

Each workspace gets a `CLAUDE.md` with non-negotiable rules (frozen folds, log every run, one
change per experiment, trust CV and check it against LB, no leakage, validate before
submitting) and a `.kaggle-gm/` state folder:

```
.kaggle-gm/competition.json   metric, direction, task, target, code-comp limits, deadline, budget
.kaggle-gm/ledger.jsonl       one line per experiment: CV, fold scores, params, git hash, folds hash, LB,
                              parent baseline, keep/discard decision
.kaggle-gm/backlog.json       ranked ideas: gain, probability, cost, evidence, source, outcome
.kaggle-gm/submissions.jsonl  every submission the hooks saw
.kaggle-gm/gpu_runs.jsonl     every remote GPU run: cap, GPU-hours used, fold scores, ledger ids
.kaggle-gm/gpu_quota.json     last `kaggle quota` snapshot (read by the hooks, no network)
artifacts/<exp_id>/           oof.npy + test.npy, the inputs for ensembling
```

## Hooks

| Event | Behaviour |
|---|---|
| SessionStart | Exports `KGKIT_HOME`; inside a workspace it injects a brief: competition facts, deadline countdown and phase, best CV/LB, accepted baseline, backlog top, CV↔LB correlation, recent experiments, submissions today, lessons from past competitions (`~/.kaggle-gm/lessons.md`). Outside a workspace it recognises existing Kaggle work (`kernel-metadata.json`, competition URLs) and suggests `/kg-adopt <slug>` |
| UserPromptSubmit | Detects competition work and tells Claude which 1–3 plugin skills to load before it answers. In evals this took skill use on Kaggle questions from 0 % to 100 % |
| PreToolUse (Bash/PowerShell) | **Denies** commands that would print Kaggle credentials: a file reader on `kaggle.json`, `~/.kaggle/access_token` or `credentials.json`, Python that prints more than their `username` field, `KAGGLE_KEY` / `KAGGLE_API_TOKEN`, `kaggle auth print-access-token`, and Kaggle CLI calls with `VERBOSE` on (the CLI then prints the bearer token); text that only mentions those files passes, and `kgkit whoami` gives the username. **Denies** a `kaggle datasets create / version` or `kaggle models instances ... create` whose folder holds credential-like files (`.env`, `kaggle.json`, `access_token`, keys) or links outside it. **Denies** each `kaggle competitions submit` in a command whose file fails validation against `sample_submission` (header, rows, ids, NaNs, index column written by mistake). **Asks** when the same file (SHA-256) was already submitted, when the command's submits would go over today's budget, or when Kaggle reported fewer submissions left today for the team than the command makes. A submit quoted in a heredoc that is only text (an issue body `cat` saves, a commit message for `git` or `gh`) is not read as one |
| PreToolUse (GPU/TPU push) | **Asks** before each GPU or TPU `kaggle kernels push` in a command when the cached weekly quota of that accelerator, minus kernels still running, is exhausted or smaller than the push's `-t` cap |
| PostToolUse | Logs each submit in a command that Kaggle replied to, its replies matched in output order (a submit in a loop once per reply; one with no reply, quoted in text or after a step that failed, is not logged unless the command runs in the background), with the file's SHA-256 and the team's remaining count Kaggle printed after it, and reminds Claude to fetch the public score and attach it to the ledger (for code competitions: grading reruns the notebook and takes hours, estimated from its measured runtime). Explains a refused submit (`400 Bad Request`, usually `FAILED_PRECONDITION` from a non-versioned dataset). Points out the Kaggle CLI's failures that exit 0 (`Dataset version creation error`, `Kernel push error`, `Could not submit to competition`). Explains a refused kernel push (a `400` from `SaveKernel`, which the CLI prints without Kaggle's reason), naming a machine shape Kaggle does not take, and names the inputs Kaggle left out of a push it accepted (the run starts without them). Logs every GPU/TPU kernel push in a command so quota accounting includes them (a re-push replaces the kernel's older reservation; a push whose folder it cannot tell, after a `cd` with a variable or in a loop, is reported instead of logged; a push in a command running in the background is logged as pending, its whole cap reserved, until `kgkit gpu status` or `gpu wait` sees Kaggle run it) |

Hooks never block for any other reason. They exit 0 on internal errors, and Python only starts
for commands that mention Kaggle (or `VERBOSE`), which keeps overhead around 70 ms. Relative paths
(`-p`, `-f`) are read from the folder the command's own `cd` steps lead to at that point.

## `kgkit` toolkit

```bash
PYTHONPATH="$KGKIT_HOME" python -m kgkit <command>     # KGKIT_HOME is set by the SessionStart hook
```

| Command | Purpose |
|---|---|
| `init <slug> --metric auc --target y --id-col id [--code-competition --runtime-hours 9 --no-internet] --deadline YYYY-MM-DD` | create a workspace |
| `adopt [slug] [--metric ... --code-competition --no-internet] [--submissions-file subs.json]` | adopt an existing repo mid-competition: writes only `.kaggle-gm/competition.json` and imports your Kaggle submission history so today's submissions count against the budget; slug and internet setting detected from `kernel-metadata.json` / competition URLs; safe to re-run |
| `status` | dashboard: accepted baseline, best CV/LB, CV↔LB correlation, recent experiments, backlog top, CV-overfitting warning |
| `folds train.csv --target y [--group g] [--strategy ...]` | leak-free folds + fold report (group leakage check) |
| `eda train.csv --test test.csv --target y --id-col id` | markdown EDA with red flags (leaks, ID-like columns, drift, unseen categories, duplicates, row-order correlation) |
| `adv train.csv test.csv --drop id,y` | adversarial validation with drifting-feature importances |
| `ledger list\|best\|show\|lb\|add` | experiment ledger; `ledger lb <id> <public> [private]` |
| `ledger compare <new> [<base>] --truth train.csv:y --folds folds.csv:fold [--fold 0] [--groups ...]` | paired per-fold test + paired OOF bootstrap against the accepted baseline → KEEP / DISCARD / INCONCLUSIVE (also works on unassembled 1-fold screens in `artifacts/<name>/`) |
| `ledger decide <id> keep\|discard\|inconclusive\|baseline` · `baseline` · `lineage` | record decisions; the accepted baseline is the last KEEP, not the best CV; lineage drives reverse ablation |
| `ledger import <name> oof.npy --test test.npy --truth ... --folds ... --source teammate` | bring a teammate's or public notebook's predictions in, CV recomputed on your folds |
| `backlog add\|list\|set\|render\|fidelity` | ranked idea backlog (`--cost` hands-on hours, `--gpu-hours` unattended Kaggle GPU time ranked against the quota left); `fidelity` checks that cheap screens rank ideas like full CV |
| `features search train.csv --target y --folds folds.csv:fold --recheck-seed 7` | generate count / target encodings, groupby aggregates and pairwise arithmetic, screen in batches, recheck on a fresh split |
| `kernels top [--sort-by scoreDescending]` · `kernels pull <owner/slug>` | top public notebooks with Kaggle's recorded best public score (search API; `--no-score` skips it), flagging titles that claim a score the record does not back up; pull with a `REVIEW.md` (CV scheme, models, public score on record, scores printed in outputs, external inputs, and the lines to read before running it locally: shell, network, credential paths, `exec`/`eval`) and a local script with Kaggle paths remapped |
| `pages [--page evaluation\|rules\|data-description] [--save reports/pages]` | the competition's own pages as text, from `kaggle competitions pages` (CLI 2.2.4+); the website's pages are rendered by JavaScript, so WebFetch often gets nothing |
| `discussions sync\|top\|search\|solutions\|read` | competition forum, live through the Kaggle CLI (2.2.4+) for active and finished competitions, or from Meta Kaggle (Kaggle's daily forum export, finished competitions only): topic index, title search, "Nth place solution" write-ups of any past competition, full threads. Forum and page text is printed inside `<untrusted-content-…>` blocks, as data rather than instructions |
| `recheck --seed 7 --run "<old>" --run "<current>" --exp <old_id> <cur_id>` | re-draw the folds with the frozen recipe and a new seed, re-run both pipelines on it, report how much of the gain survived |
| `blend --exp 3 7 9 --truth train.csv:y --folds folds.csv:fold --method hill --sample sample_submission.csv` | hill climbing / weights / rank / `stack --levels N` / `residual --base <id>`, `--prune N` for large libraries, honest nested score; warns if members used different folds |
| `validate sub.csv sample_submission.csv` | metric-aware submission validation |
| `metrics` / `score` / `vendor` | metric registry / scoring / copy kgkit into a project or Kaggle dataset |
| `gpu quota\|plan\|build\|push\|status\|wait\|collect\|release\|cancel\|log` | train on Kaggle GPUs and TPUs: live quota (GPU + TPU) and what holds it, budget to the deadline, remote training kernels, push of any kernel folder with a cap, ledger import, GPU/TPU-hour accounting ([below](#training-on-kaggle-gpus)) |
| `datasets pull <owner/slug> <dir> [--files GLOB]` | download a dataset; falls back to one request per file when the whole-dataset archive returns 404 |
| `whoami` · `config [key=value ...]` | your Kaggle username (never the key) · show or set `competition.json` fields, e.g. `validation=holdout+lb holdout_n=58` |

Python API highlights: `kgkit.cv.assign_folds` (stratified / group / stratified-group /
multilabel), `time_series_splits(gap=h)`, `purged_kfold_splits`; `kgkit.metrics.get(name)`;
`kgkit.thresholds.OptimizedRounder`; `kgkit.ensemble.hill_climb / cv_blend_score / stack /
multi_level_stack / residual_stack / prune_library`; `kgkit.compare.compare_folds / paired_bootstrap`;
`kgkit.featsearch.feature_search / materialize`; `kgkit.backlog.Backlog`;
`kgkit.adversarial.adversarial_validation`; `kgkit.features.oof_target_encode / lag_features`;
`kgkit.experiment.Ledger().log(...)`.

## Templates

| File | Highlights |
|---|---|
| `train_gbdt.py` | LightGBM / XGBoost / CatBoost / sklearn HGB, plus linear / kNN / SVM / MLP pipelines for diverse baselines; frozen folds, seed averaging, label-metric decoding fitted on OOF (threshold / argmax / optimised rounding), log-target, id-aligned submission, `--smoke` |
| `train_image.py` | timm; AMP (bf16/fp16), EMA, cosine warmup, head LR multiplier, hflip TTA; `--folds` lets you split folds across GPUs, and the run that completes the set logs the experiment; checkpoints before a session deadline and resumes, learning-curve pruning, `--assemble` |
| `train_transformer.py` | HF AutoModel with mean/attention/CLS pooling, layer-wise LR decay, gradient accumulation, several evaluations per epoch, length-sorted inference, offline-model friendly; same deadline/resume/pruning/`--assemble` support |
| `kaggle_gpu_runner.py` | generated by `kgkit gpu build`: the kernel that runs your training command on Kaggle (code bundle embedded, inputs linked into `data/` or one folder each, one (job, fold) per GPU, deadline, resume, assembly, `kg_run.json` report) |
| `inference_kernel.py` + `kernel/*.json` | offline Kaggle kernel: runtime test discovery, offline wheels, time-budgeted ensemble members with fallback, output validation; metadata with internet off |

## Training on Kaggle GPUs

No local GPU? `kgkit gpu` (and `/kg-gpu`, skill `kaggle-gpu`) runs your local training command on
Kaggle's free GPUs. It treats the weekly quota as the budget: the aim is the most leaderboard per
GPU-hour, not the fastest wall clock.

```bash
python -m kgkit gpu quota                    # live quota, reset countdown, "use it or lose it" warning
python -m kgkit gpu plan                     # hours until the deadline, final-week reserve, measured costs
# screen two ideas on fold 0 in ONE session (one per T4), stopping losers early
python -m kgkit gpu build --name lr3e4 --folds 0 --hours 2 --prune-against baseline --dataset me/knee-png-512 \
    --cmd "python src/train_image.py --name lr3e4 --lr 3e-4 --folds {fold}" \
    --extra-job res448 "python src/train_image.py --name res448 --img-size 448 --folds {fold}"
python -m kgkit gpu push kernels/lr3e4-train  # quota check + -t cap + run log (confirm first: spends quota)
python -m kgkit gpu wait <user>/lr3e4-train --collect    # background; collect = ledger + GPU-hours + diagnosis
# decide each screen against a control screen of the baseline at the same reduced settings
python -m kgkit ledger compare artifacts/lr3e4 artifacts/control --truth data/train.csv:y --folds data/folds.csv:fold --fold 0
```

What it does for you:
- **Both T4s busy.** One process per (job, fold), one per GPU. Two folds or two screens share a quota hour.
- **Same code, same folds.** Your `src/`, `kgkit`, `competition.json` and `data/folds.csv` are embedded
  in the kernel. Inputs are linked into `./data`, so the command runs unchanged and the imported ledger
  records carry the same `folds_hash`. With `--input-layout nested` each dataset and kernel output gets its
  own folder (`data/ext/<slug>/`, `data/ext/kout-<slug>/`), so libraries with the same file names can be
  stacked; `--dataset <owner>/<slug>=data/<folder>` places one input. A name two inputs share is never
  dropped silently: the run log and `collect` say which input kept it.
- **Never loses work.** `KG_DEADLINE` makes the templates checkpoint before the session limit.
  `--resume-from <kernel>` skips finished folds and continues timed-out ones.
- **Stops losers early.** `--prune-against baseline` compares the best-so-far score with the baseline's learning
  curve at the same fraction of the schedule, using the baseline's fold std as the margin.
- **Measures itself.** `collect` imports ledger records with fresh ids, records GPU-hours, and flags an idle
  second GPU, input-bound runs (< 60 % utilisation), deadline hits and pruned folds.
- **Guards the quota.** `push` refuses runs whose cap exceeds the plannable hours, re-checks runs it had
  as in flight (finished ones free their reservation), refuses while an attached kernel is still running
  (inputs are bound at push time), and warns when your account's batch-session slots look full. The hook
  asks before a raw GPU/TPU `kaggle kernels push` when the quota is exhausted.
- **Says why a push fails.** `push` refuses a machine shape Kaggle is not known to take (such as
  `NvidiaTeslaT4Highmem`, which `kaggle kernels pull -m` can record) before anything reaches Kaggle, names the
  shape to ask for, and warns when Kaggle accepts a push but leaves inputs out. kgkit's Kaggle calls print
  Kaggle's reason for a refused request (`Kaggle's reason: ...`), which the Kaggle CLI drops.
- **Any kernel.** `gpu push <dir> --hours H` takes notebook, inference and diagnostics kernels too.
- **TPU when the GPU quota is gone.** `--accelerator TpuV5E8` runs one process per TPU chip on the separate
  20 h/week TPU quota; the skill covers the torch_xla port and the 10-24 h queue.

## Verification

- **147 pytest tests** (one runs only when LightGBM is installed) cover kgkit (including paired comparisons, the backlog, feature search, multi-level/residual stacking, notebook review, competition pages, upload checks and repo adoption), all templates end to end (tiny synthetic data, plus an
  offline tiny BERT), the remote GPU runner executed in a simulated `/kaggle` layout (build, run on two
  fake GPUs, resume, prune, collect), and the hooks driven through `run.sh` exactly as Claude Code calls them.
- **Live Claude Code sessions** confirmed the SessionStart brief and the credential guard
  (tested against a dummy `kaggle.json`).
- **Real-data dry run** on `playground-series-s6e2` (630k rows, no submission made): EDA 6 s,
  adversarial AUC 0.501, LightGBM CV AUC 0.95521, HGB 0.95490, hill-climb blend 0.95537
  (honest nested). The run surfaced 5 bugs, all fixed (see git history).
- **Plugin evals** (`claude plugin eval`, Haiku, 3 runs per arm): the with-plugin arm scored
  100 % on all 10 cases. `ensemble-hygiene`, `gpu-quota-budget` and `team-merge` (no private sharing of
  OOF files before a merge; shared folds after) scored 1.00 with the plugin vs 0.00 without. The other
  cases pass with or without the plugin on Haiku, so they work as regression tests, as does
  `untrusted-notebook-injection` (1.00 in both arms).

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
claude plugin validate plugins/kaggle-grandmaster
cd plugins/kaggle-grandmaster && claude plugin eval . --runs 3 --model haiku
```

Releases are automatic. On every push to `main`, `.github/workflows/release.yml` runs
`scripts/bump_version.py`. That bumps the version in `plugin.json`, `marketplace.json` and
`kgkit/__init__.py`, commits `Release X.Y.Z`, then tags `kaggle-grandmaster--vX.Y.Z` and creates a GitHub
release whose notes GitHub generates from the merged PRs. The bump is a patch
by default. The PR labels `release:minor` and `release:major` select a bigger bump, and `no-release` skips
the release. Don't edit the versions by hand.

Repository layout:

```
.claude-plugin/marketplace.json        marketplace manifest (this repo is installable)
plugins/kaggle-grandmaster/
  .claude-plugin/plugin.json
  skills/<name>/SKILL.md               21 knowledge skills + 17 kg-* commands
  agents/*.md                          9 subagents
  hooks/                               hooks.json, run.sh launcher, stdlib-only Python hooks
  kgkit/                               the toolkit (python -m kgkit)
  templates/                           training / inference templates, CLAUDE.md template
  evals/                               plugin eval suite
scripts/bump_version.py                version check / release bump (used by CI)
tests/                                 pytest suite
```

## Data and network access

The plugin has no server of its own and collects nothing. All network traffic goes to Kaggle under
your own account, through the official `kaggle` CLI and the API token you set up yourself. The hooks
make no network calls.

**Exchanged with Kaggle (kaggle.com)**
- Competition data: `kaggle competitions files / download / leaderboard / submissions /
  submission-limits / pages` (`kgkit pages`).
- Submissions: `kaggle competitions submit` uploads your submission file. Claude confirms before it
  submits, and the hooks block files that fail validation.
- GPU training (`/kg-gpu`, `kgkit gpu`): `kaggle kernels push` uploads a private kernel containing
  your workspace's `src/`, `kgkit`, `competition.json` and `data/folds.csv`. `kaggle kernels status /
  logs / output` fetch its results, and `kaggle quota` reads your weekly GPU quota.
- Code-competition notebooks and model weights (`/kg-kernel`): `kaggle kernels push` and
  `kaggle datasets create / version`, private unless you make them public.
- Public notebooks: `kaggle kernels list / pull` (`kgkit kernels top / pull`).
- Forums: `kaggle competitions topics` / `topic-messages` (CLI 2.2.4+), and for finished competitions
  `kaggle datasets download kaggle/meta-kaggle`, Kaggle's public forum export (`kgkit discussions --source meta`).

**Web reads.** `/kg-start`, `/kg-recon`, `/kg-ideas`, `/kg-grind`, `/kg-postmortem` and the
`competition-analyst` and `solution-researcher` agents use Claude's WebSearch and WebFetch to read
public competition pages, past solution write-ups and papers. They don't upload your files, and Claude
asks before each call unless you have allowed these tools.

**Credentials.** The plugin never uses or sends your Kaggle API key. `kgkit gpu` takes only your
Kaggle username, from `KAGGLE_USERNAME` or the `username` field of `kaggle.json`, to name your
kernels. The PreToolUse hook blocks commands that would print a credential (`kaggle.json`,
`~/.kaggle/access_token`, `credentials.json`, `KAGGLE_KEY`, `kaggle auth print-access-token`, `VERBOSE`
Kaggle CLI calls) and dataset or model uploads whose folder holds credential-like files. `kgkit gpu build`
leaves credential-like files and links outside the project out of the kernel it uploads.

**Text from Kaggle.** Competition pages, forum posts and public notebooks are written by other people.
The skills and agents treat them as data, never as instructions; `kgkit pages` and `kgkit discussions`
print them inside `<untrusted-content-…>` blocks, and `kgkit kernels pull` lists a notebook's shell,
network, credential-path and `exec`/`eval` lines before you run it locally.

**Stored on your machine only**
- `.kaggle-gm/` in the workspace: experiment ledger, idea backlog, submission log, GPU-run log and
  the last quota snapshot.
- `artifacts/` in the workspace: out-of-fold and test predictions.
- `~/.kaggle-gm/meta-kaggle/`: the cached forum export (only with `discussions --source meta`).
- `~/.kaggle-gm/lessons.md`: optional lessons from past competitions, read by the session brief.

## Limitations

- Only the user can accept competition rules, create API tokens and tick final submissions on
  the website. The plugin reminds you at the right moments.
- Submitting uses your quota and goes out to Kaggle. Claude confirms before submitting unless
  you have explicitly authorised autonomous submissions. `kg-grind` never submits on its own.
- Pushing a GPU kernel spends your weekly Kaggle quota. Claude confirms before `kgkit gpu push` unless
  you have authorised remote runs for the task. Quota and session limits are read live (`kaggle quota`,
  Kaggle CLI >= 2.2) and may change on Kaggle's side.
- The domain playbooks reflect competition practice up to 2026. Recon and the
  solution-researcher agent exist so that current competition-specific findings take priority.

## License

MIT
