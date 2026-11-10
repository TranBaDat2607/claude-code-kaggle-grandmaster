# Kaggle Grandmaster: full reference

Everything the [README](../README.md) leaves out: the workflow, the workspace files, each hook, the Kaggle pane,
every `kgkit` command, the templates, GPU training, how it is tested, and what goes over the network.

- [Permissions](#permissions)
- [What's inside](#whats-inside)
- [How a competition flows](#how-a-competition-flows)
- [Hooks](#hooks)
- [Kaggle pane (`/kaggle-monitor`)](#kaggle-pane-kaggle-monitor)
- [`kgkit` toolkit](#kgkit-toolkit)
- [Templates](#templates)
- [Compute efficiency](#compute-efficiency)
- [Training on Kaggle GPUs](#training-on-kaggle-gpus)
- [Verification](#verification)
- [Development](#development)
- [Data and network access](#data-and-network-access)
- [Limitations](#limitations)

## Permissions

The `kg-*` commands don't pre-approve any shell, file-write or web tools, so
Claude asks before each one according to your own permission settings. If you want `/kg-grind` to
run hands-off, allow the toolkit in your workspace's `.claude/settings.json`:

```json
{ "permissions": { "allow": ["Bash(python -m kgkit:*)", "Edit(./src/**)", "Write(./src/**)"] } }
```

Kaggle submissions and GPU pushes still go through the plugin's hooks, which check them either way.

## What's inside

| Layer | Count | What it does |
|---|---|---|
| **Knowledge skills** | 22 | Playbooks Claude loads when relevant: master playbook, recon, validation, tabular, CV, NLP/LLM, time series, audio, simulation/optimisation, recsys, ensembling, metrics, tuning, DL training, boosters, code competitions, Kaggle CLI, Kaggle GPUs, compute efficiency (speed defaults by domain, from winning solutions), CV-LB debugging, final selection, and competition strategy (choosing competitions, solo gold, teaming and merging) |
| **Slash commands** | 18 | `kg-start`, `kg-adopt`, `kg-recon`, `kg-eda`, `kg-cv`, `kg-baseline`, `kg-experiment`, `kg-grind`, `kg-gpu`, `kg-ensemble`, `kg-submit`, `kg-status`, `kg-kernel`, `kg-final`, `kg-debug`, `kg-ideas`, `kg-postmortem`, `kg-showcase` |
| **Specialist agents** | 9 | competition-analyst, solution-researcher, data-detective, validation-auditor, feature-engineer, experiment-runner, error-analyst, ensemble-architect, kernel-packager |
| **Hooks** | 5 | session brief (incl. GPU quota), Kaggle-prompt → skill router, submission + GPU-push + upload guard (validation, repeated files, daily budget, weekly GPU quota, credential protection), submission / GPU-run logger, compute defaults when model code is written |
| **Kaggle pane** | 1 mod | `/kaggle-monitor`: running kernels with live CPU / GPU / RAM / disk and what each GPU is doing, recent kernels and submissions, GPU/TPU quota for every account, multi-account token management ([below](#kaggle-pane-kaggle-monitor)) |
| **`kgkit` toolkit** | 25 modules | Tested Python: leak-free folds, 35+ metrics, threshold/rounding optimisers, hill climbing / multi-level and residual stacking / library pruning, adversarial validation, leak-safe features, brute-force feature search, EDA red flags, experiment ledger with decisions + lineage, paired experiment comparison, fresh-split recheck, idea backlog, public-notebook review, competition pages and discussions, submission validation, Kaggle GPU/TPU pipeline, speed measurement (input-bound / compute-bound / idle, speed-up of a change) and slow-pattern checks, dataset download with per-file fallback, upload and third-party-text safety |
| **Templates** | 5 + metadata | Tabular (LightGBM/XGBoost/CatBoost/HGB + linear/kNN/SVM/MLP, GPU option), timm image models, HF transformers (both resumable, with `--bench`), offline inference kernel, Kaggle-pane monitor cell for any notebook |
| **Evals** | 48 cases | `claude plugin eval` suite with a no-plugin baseline arm |

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
             ▼
   kg-showcase (two ways in: solution/ = the final pipeline + the commands that reproduce it,
                journey/ = how the ideas developed, phase by phase, from the ledger with links to each
                experiment's code; README around the result, own vs public work, license, figures, write-up
                draft; audited before anything goes public, asks before each public step)
```

Each workspace gets a `CLAUDE.md` with non-negotiable rules (frozen folds, log every run, one
change per experiment, trust CV and check it against LB, no leakage, validate before
submitting, keep the repo readable) and a layout meant to go public when the competition ends:

```
src/          the pipeline: training / inference scripts; a variant is a flag of its script, not a copy
studies/      one folder per analysis a decision rests on (script + README)
notebooks/    exploration, numbered
kernels/      Kaggle notebooks / datasets you write
third_party/  code that is not yours (a vendored kgkit)
work/         scratch, ignored: probes, logs, downloads
data/ artifacts/ subs/   ignored; reports/ markdown
```

Every record keeps what the journey needs: the phase it ran in (`ledger phase`), the command that ran it (`cmd`),
and its exact code: the commit, or, when the tree had uncommitted changes, a snapshot commit under `refs/kg/<id>`
(`git_snapshot`; written from a temporary index, so the working tree and the staging area are not touched; runs of
the same code share one; `KG_NO_SNAPSHOT=1` turns them off). The refs stay local until `/kg-showcase` pushes them,
after `kgkit audit-repo` has read them. The session brief and `kgkit status` ask for a phase and for decisions, and
name drift from the layout: copies of scripts in `src/`, a crowded `src/`, logs or generated files tracked by git.
The `.kaggle-gm/` state folder:

```
.kaggle-gm/competition.json   metric, direction, task, target, code-comp limits, deadline, budget
.kaggle-gm/ledger.jsonl       one line per experiment: CV, fold scores, params, git hash, folds hash, LB,
                              parent baseline, keep/discard decision
.kaggle-gm/backlog.json       ranked ideas: gain, probability, cost, evidence, source, outcome
.kaggle-gm/submissions.jsonl  every submission the hooks saw
.kaggle-gm/kaggle_submissions.json  Kaggle's list with public and private scores, as `ledger sync-lb` last read it
.kaggle-gm/gpu_runs.jsonl     every remote GPU run: cap, GPU-hours used, fold scores, ledger ids
.kaggle-gm/gpu_quota.json     last `kaggle quota` snapshot (read by the hooks, no network)
.kaggle-gm/build/, remote/    kernels `gpu build` writes, outputs `gpu collect` downloads (ignored by git)
artifacts/<exp_id>/           oof.npy + test.npy, the inputs for ensembling
```

## Hooks

| Event | Behaviour |
|---|---|
| SessionStart | Exports `KGKIT_HOME`; inside a workspace it injects a brief: competition facts, deadline countdown and phase, best CV/LB, accepted baseline, backlog top, CV↔LB correlation, recent experiments, the phase the work is in (or a nudge to name one) and records with no decision, drift from the workspace layout, submissions today, lessons from past competitions (`~/.kaggle-gm/lessons.md`), and a vendored kgkit (`third_party/kgkit`) whose version is not the plugin's, with the command that refreshes it. It also shows a warning while the last `kgkit audit-repo` found competition data in git or data files `git add -A` would commit, and a nudge to run the audit in a repo with a GitHub remote that was never audited (read from the cached result: no scan and no network call at session start). After the deadline the brief is short: the final standing and finals as the last `kgkit final` read them, whether `reports/postmortem.md` exists, the repo audit, and whether lessons for the competition are in `~/.kaggle-gm/lessons.md`; the CV, submission and GPU lines are dropped. Outside a workspace it recognises existing Kaggle work (`kernel-metadata.json`, competition URLs) and suggests `/kg-adopt <slug>` |
| UserPromptSubmit | Detects competition work and tells Claude which 1–3 plugin skills to load before it answers. In evals this took skill use on Kaggle questions from 0 % to 100 %. For a slash-only command it cannot load ("show my work" after the deadline → `/kg-showcase`), it tells Claude to suggest the command. After the deadline its fallback is `kg-postmortem`, not the playbook. On the deadline's day, while fewer finals are selected than allowed, it says once per session how many are (`kgkit final`) |
| PreToolUse (Bash/PowerShell) | **Denies** commands that would print Kaggle credentials: a file reader on `kaggle.json`, `~/.kaggle/access_token` or `credentials.json`, Python that prints more than their `username` field, `KAGGLE_KEY` / `KAGGLE_API_TOKEN`, `kaggle auth print-access-token`, and Kaggle CLI calls with `VERBOSE` on (the CLI then prints the bearer token); text that only mentions those files passes, and `kgkit whoami` gives the username. **Denies** a `kaggle datasets create / version` or `kaggle models instances ... create` whose folder holds credential-like files (`.env`, `kaggle.json`, `access_token`, keys) or links outside it. **Denies** each `kaggle competitions submit` in a command whose file fails validation against `sample_submission` (header, rows, ids, NaNs, index column written by mistake). **Asks** when the same file (SHA-256) was already submitted, when the command's submits would go over today's budget, or when Kaggle reported fewer submissions left today for the team than the command makes. A submit quoted in a heredoc that is only text (an issue body `cat` saves, a commit message for `git` or `gh`) is not read as one |
| PostToolUse (Write/Edit) | When a `.py` file written in a workspace trains or runs a model, it gives Claude that domain's compute defaults once per session, with the path of the `compute-efficiency` reference (subagents without the Skill tool can read it), and names slow patterns not reported yet: fp32 training, a DataLoader without workers, no `cudnn.benchmark` for a CNN, XGBoost `exact`/`gpu_hist`, row-wise DataFrame loops, prediction without `inference_mode`, bf16 chosen with `torch.cuda.is_bf16_supported()` (also True on a T4 or P100, where bf16 is emulated), a Hugging Face model fine-tuned in its checkpoint's fp16/bf16 (the transformers 5 default) (`kgkit speedcheck`). A new script written as a copy of one beside it (`train_v2.py`) gets a note, once: a variant belongs behind a flag. Other writes start no Python |
| PreToolUse (GPU/TPU push) | **Asks** before each GPU or TPU `kaggle kernels push` in a command when the cached weekly quota of that accelerator, minus kernels still running, is exhausted or smaller than the push's `-t` cap |
| PreToolUse (any kernel push) | **Edits** the code file of each `kaggle kernels push` the command runs (not one quoted in a message or `grep`): adds the hardware monitor cell (`[kgmon]` lines for `/kaggle-monitor`) to a Python script or notebook that does not mention kgmon yet, and tells Claude so. The push itself is never blocked |
| PostToolUse | Logs each submit in a command that Kaggle replied to, its replies matched in output order (a submit in a loop once per reply; one with no reply, quoted in text or after a step that failed, is not logged unless the command runs in the background), with the file's SHA-256 and the team's remaining count Kaggle printed after it, and reminds Claude to fetch the public score and attach it to the ledger (for code competitions: grading reruns the notebook and takes hours, estimated from its measured runtime). Explains a refused submit (`400 Bad Request`, usually `FAILED_PRECONDITION` from a non-versioned dataset). Points out the Kaggle CLI's failures that exit 0 (`Dataset version creation error`, `Kernel push error`, `Could not submit to competition`). Explains a refused kernel push (a `400` from `SaveKernel`, which the CLI prints without Kaggle's reason), naming a machine shape Kaggle does not take or a source of 1 MB or more, and names the inputs Kaggle left out of a push it accepted (the run starts without them). Logs every GPU/TPU kernel push in a command so quota accounting includes them (a re-push replaces the kernel's older reservation; a push whose folder it cannot tell, after a `cd` with a variable or in a loop, is reported instead of logged; a push in a command running in the background is logged as pending, its whole cap reserved, until `kgkit gpu status` or `gpu wait` sees Kaggle run it). Once per session, a command that writes a log outside `work/` (`> train.log`, `tee reports/x.log`) gets a note: logs go in `work/` |

Hooks never block for any other reason. They exit 0 on internal errors, and Python only starts
for commands that mention Kaggle (or `VERBOSE`), which keeps overhead around 70 ms. Relative paths
(`-p`, `-f`) are read from the folder the command's own `cd` steps lead to at that point.

## Kaggle pane (`/kaggle-monitor`)

The plugin also ships a mod, `hooks/monitor/`: a TypeScript hooks module that Claude Code loads beside the Python
hooks and that opens a pane beside the chat, so you do not have to keep Kaggle open in a browser. Run
`/kaggle-monitor` (or `/kaggle-monitor quota`, `accounts`, `use <alias>`, `refresh`) to open it; it never opens by
itself. Keys: `1` `2` `3` switch tabs, `r` refreshes, `u` opens or closes every running kernel's card.

| Tab | Shows |
|---|---|
| Activity | Every kernel running on any registered account, as a card you open or close by its title line (one running kernel opens, two or more start closed; your choice is kept across sessions). Closed, it is one line of CPU, RAM, each GPU's use and the training progress, with its newest log line under it. Open, it shows the whole run so far (the log is replayed from the start): a sparkline of CPU, RAM and each GPU with the value now, the average and the peak, GPU memory, disk, progress (`step 820/858`, an epoch, a tqdm bar), what it is doing (job and fold on each GPU) and its last three log lines. Each card's title has the session's run time and limit (`limit 12h · 28m left`), see below; the other kernels of the last 24 h (running blue, complete green, failed red); the workspace's latest submissions and scores. A notification when a kernel finishes or a submission is scored. `c` opens *This computer*: local CPU, RAM, disk, NVIDIA GPUs and your training processes, each described from its command line |
| Quotas | GPU and TPU hours used and left, and the reset time, for every account, and the total across them |
| Accounts | The accounts of `~/.kaggle/kg_accounts.json`: **Use** points this session's Kaggle CLI and kgkit at that account's token file (`KAGGLE_API_TOKEN`), tells Claude which account it is, and is remembered for the next session; **Add** checks a pasted API token (or a token file) with Kaggle, then saves it to `~/.kaggle/access_token_<alias>`, readable by you alone; the registry holds only paths. Where the Kaggle CLI is, found from its install without contacting Kaggle |

Kaggle's API reports no resources of a running session, so the kernel prints them: kernels built by
`kgkit gpu build` print a `[kgmon] {...}` line every 30 s (`KG_MONITOR_S`, `0` turns it off). Before any other
`kaggle kernels push` Claude runs, the PreToolUse hook adds `templates/kaggle_monitor_cell.py` to the kernel's code
(the first cell of a notebook, the top of a script) and tells Claude it changed the file; a code file that mentions
kgmon anywhere is left alone, so a `# kgmon: off` comment opts out. For a kernel you push yourself, paste the cell in
as its first cell (`kgmon_doing("fold 2")` adds what it is doing).
The pane follows each running kernel's live log (`kaggle kernels logs -f`, Kaggle CLI 2.2 or later). Only batch runs
(Save Version, `kaggle kernels push`) have a log the API streams; an interactive editor session does not.

Kaggle's status of a kernel lags: it can say RUNNING for minutes after the script has ended, after the outputs are
saved, or after the session limit has cancelled it. So each running card also keeps a session clock:

- **The limit**: the push's `-t` / `--hours` as `.kaggle-gm/gpu_runs.jsonl` records it (`kgkit gpu push`, or a GPU or
  TPU `kaggle kernels push` the hook saw), else Kaggle's longest session (12 h GPU or CPU, 9 h TPU). It counts from
  the kernel's own clock (the `t0` of its `[kgmon]` lines), else from Kaggle's last-run time for a GPU or CPU session,
  or, for a TPU session, from when it was first seen running (TPU sessions queue for hours); a TPU kernel never seen
  running shows its limit with no countdown. In the last half hour (a quarter of a limit under 2 h), while the script
  still runs, the card turns yellow and warns, once with a notification.
- **SAVING**: the log shows the notebook converted to HTML (`[NbConvertApp] ... to html`): the script has finished
  (or stopped on an error) and Kaggle is saving the outputs. Near the limit it warns that a save the limit cuts off
  can lose them.
- **PAST LIMIT**: the clock has passed the limit: Kaggle has probably cancelled the session; the run time stops at
  the limit.
- **ENDED**: Kaggle has closed the log (a reader replays it and the stream ends by itself): the session is over.
- **Outputs**: once the script has ended or the session looks over, the pane lists the files Kaggle serves for the
  session every few minutes (kgkit's `kaggle_outputs`, under the Kaggle CLI's Python; nothing is downloaded): whether
  `kaggle kernels output` would get anything. The notification when the status finally changes says how many files
  were readable. A cancelled version usually cannot be attached to another kernel: download its outputs and upload
  them as a dataset.

Mods need a Claude Code release that loads hooks modules; elsewhere the pane is simply absent and the rest of the
plugin works as before.

## `kgkit` toolkit

```bash
PYTHONPATH="$KGKIT_HOME" python -m kgkit <command>     # KGKIT_HOME is set by the SessionStart hook
```

`python -m kgkit vendor third_party` copies the package into the project so training scripts can `import kgkit` (the
templates look in `third_party/`); every command also runs from that copy, at the copy's version. Vendor again (with
the plugin's kgkit) to update it.

| Command | Purpose |
|---|---|
| `init <slug> --metric auc --target y --id-col id [--code-competition --runtime-hours 9 --no-internet] --deadline YYYY-MM-DD` | create a workspace |
| `adopt [slug] [--metric ... --code-competition --no-internet] [--submissions-file subs.json]` | adopt an existing repo mid-competition: writes only `.kaggle-gm/competition.json` and imports your Kaggle submission history so today's submissions count against the budget; slug and internet setting detected from `kernel-metadata.json` / competition URLs; safe to re-run |
| `audit-repo [--data DIR] [--fix-gitignore]` | is competition data in git, and is the GitHub remote public? Kaggle's rules forbid publishing Competition Data, and a `.gitignore` that names only the downloaded files lets in a file made from them. So this compares contents with the competition's files on your machine (the untracked files in `data/`, or `--data`): exact copies, rows (re-saved CSVs and CRLF included), passages of text (template sentences and IDs don't count), and label columns equal to the competition's for the same IDs. It checks every branch, tag and snapshot ref (`refs/kg/*`; deleted files stay in history) and what `git add -A` would commit. A table keyed by the competition's IDs whose values are your own (predictions, model-made labels) is listed as a judgment call. It prints the cleanup steps (it never rewrites history), exits 1 on a leak, and `--fix-gitignore` adds `data/` to `.gitignore`. `init` and `adopt` run the same checks |
| `status` | dashboard: accepted baseline, best CV/LB, CV↔LB correlation, recent experiments, backlog top, CV-overfitting warning, and a warning when Kaggle would auto-select a `public-lb-fitted` file as a final or, in the last 3 days, when no selection is confirmed |
| `final` | asks Kaggle which submissions are selected as finals (the website selects them; there is no API for that), shows what auto-selection would take when none is, flags `public-lb-fitted` files, records the selection (`final_picks`) and when Kaggle confirmed it (only that read silences the status warning; `config final_picks=` is just an intended list), prints the website steps and the deadline in UTC and local time. After the deadline it reports the outcome instead: the finals' private scores (with none selected, the pool Kaggle picked from and which of them the team score matches), the team's private and public rank, the medal zone computed from Kaggle's progression table, and the best private score of any submission |
| `folds train.csv --target y [--group g] [--strategy ...]` | leak-free folds + fold report (group leakage check) |
| `eda train.csv --test test.csv --target y --id-col id` | markdown EDA with red flags (leaks, ID-like columns, drift, unseen categories, duplicates, row-order correlation) |
| `adv train.csv test.csv --drop id,y` | adversarial validation with drifting-feature importances |
| `ledger list\|best\|show\|lb\|add` | experiment ledger; `ledger lb <id> <public> [private]` |
| `ledger sync-lb [--dry-run]` | one read of Kaggle's submission list attaches the public and private scores to every record a submission names (the ledger id in its description or file name, or the number the description starts with); keeps a hand-typed score that agrees to the digits Kaggle shows, lists the submissions no record names, saves the list for `predictiveness` |
| `ledger predictiveness [--top 10] [--plot DIR]` | after the deadline: Spearman of CV and public score with private (95% bootstrap intervals) over every submission and among the top k by public score and by CV, the private score each way of picking the finals gave, and the public-to-private drop; `--plot` writes the public-vs-private, scores-over-time and CV-vs-private figures (needs matplotlib) |
| `ledger compare <new> [<base>] --truth train.csv:y --folds folds.csv:fold [--fold 0] [--groups ...]` | paired per-fold test + paired OOF bootstrap against the accepted baseline → KEEP / DISCARD / INCONCLUSIVE (also works on unassembled 1-fold screens in `artifacts/<name>/`) |
| `ledger decide <ids> keep\|discard\|inconclusive\|baseline` · `baseline` · `lineage` | record decisions (`<ids>`: an id, a list `3,5`, or a range `12..20` for a sweep); the accepted baseline is the last KEEP, not the best CV; lineage drives reverse ablation |
| `ledger phase [<name>] [--records 12..20]` | name the phase the work is in (every record logged from then on carries it), put past records in a phase, or list the phases |
| `ledger journey [--out journey/experiments.md] [--repo-url URL]` | write every record by phase: change, parent, CV, Δ, public / private, decision, a link to the code it ran from and a diff against its parent's (GitHub links from the remote), the commands, and the lineage as a Mermaid graph. `/kg-showcase` writes `journey/README.md`, the story, around it |
| `ledger tag <id> <tag>... [--remove]` | tag a logged record, e.g. `public-lb-fitted` (a file tilted toward the public labels: display only; kept out of best CV, the baseline and the CV↔LB correlation; never accepted as KEEP/baseline) |
| `ledger import <name> oof.npy --test test.npy --truth ... --folds ... --source teammate` | bring a teammate's or public notebook's predictions in, CV recomputed on your folds |
| `backlog add\|list\|set\|render\|fidelity` | ranked idea backlog (`--cost` hands-on hours, `--gpu-hours` unattended Kaggle GPU time ranked against the quota left); `fidelity` checks that cheap screens rank ideas like full CV |
| `features search train.csv --target y --folds folds.csv:fold --recheck-seed 7` | generate count / target encodings, groupby aggregates and pairwise arithmetic, screen in batches, recheck on a fresh split; `--model xgb --device cuda` screens on the GPU |
| `kernels top [--sort-by scoreDescending]` · `kernels pull <owner/slug>` | top public notebooks with Kaggle's recorded best public score (search API; `--no-score` skips it), flagging titles that claim a score the record does not back up; pull with a `REVIEW.md` (CV scheme, models, public score on record, scores printed in outputs, external inputs, and the lines to read before running it locally: shell, network, credential paths, `exec`/`eval`) and a local script with Kaggle paths remapped |
| `pages [--page evaluation\|rules\|data-description] [--save reports/pages]` | the competition's own pages as text, from `kaggle competitions pages` (CLI 2.2.4+); the website's pages are rendered by JavaScript, so WebFetch often gets nothing |
| `discussions sync\|top\|search\|solutions\|read` | competition forum, live through the Kaggle CLI (2.2.4+) for active and finished competitions, or from Meta Kaggle (Kaggle's daily forum export, finished competitions only): topic index, title search, "Nth place solution" write-ups of any past competition, full threads. Forum and page text is printed inside `<untrusted-content-…>` blocks, as data rather than instructions |
| `bench --run "<cmd>" [--run "<cmd>"]` | time commands while sampling the GPUs (nvidia-smi): wall time, utilisation and memory per GPU, a verdict (GPU idle / under-used / busy, a second GPU left idle), the speed-up of each run against the first and against the last bench of the same command (`reports/bench.jsonl`). The image and text templates' `--bench` splits the data loader from the GPU step (input-bound vs compute-bound) |
| `speedcheck src/train.py` | slow patterns in a training or inference script and the compute defaults of its domain (the checks the write hook runs) |
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
| `train_gbdt.py` | LightGBM / XGBoost / CatBoost / sklearn HGB, plus linear / kNN / SVM / MLP pipelines for diverse baselines; frozen folds, seed averaging, label-metric decoding fitted on OOF (threshold / argmax / optimised rounding), log-target, id-aligned submission, `--smoke`, `--device cuda` (XGBoost / CatBoost on the GPU, LightGBM's GPU build when one works) |
| `train_image.py` | timm; AMP (bf16/fp16), channels_last, cudnn.benchmark, fused AdamW, `--compile` (Linux, sm_70+), `--bench`, EMA, cosine warmup, head LR multiplier, hflip TTA; `--folds` lets you split folds across GPUs, and the run that completes the set logs the experiment; checkpoints before a session deadline and resumes, learning-curve pruning, `--assemble` |
| `train_transformer.py` | HF AutoModel with mean/attention/CLS pooling, layer-wise LR decay, gradient accumulation, length-bucketed training batches, fused AdamW, `--bench`, several evaluations per epoch, length-sorted inference, offline-model friendly; same deadline/resume/pruning/`--assemble` support |
| `kgkit/kaggle_gpu_runner.py` | filled in by `kgkit gpu build`, and part of the package so that a vendored kgkit (`kgkit vendor third_party`) builds kernels too: the kernel that runs your training command on Kaggle (code bundle embedded, inputs linked into `data/` or one folder each, one (job, fold) per GPU, deadline, resume, assembly, `kg_run.json` report) |
| `kaggle_monitor_cell.py` | the first cell for a notebook kgkit did not build: prints the session's CPU, RAM, disk and GPUs, and when its clock started, as `[kgmon]` lines for the Kaggle pane |
| `inference_kernel.py` + `kernel/*.json` | offline Kaggle kernel: runtime test discovery, offline wheels, time-budgeted ensemble members with fallback, output validation; metadata with internet off |

## Compute efficiency

Quota hours and runtime limits are fixed, so a faster loop buys more experiments and a faster kernel carries a
bigger ensemble. The `compute-efficiency` skill holds the speed defaults per domain (images and 3D, text and LLMs,
tabular and large dataframes, audio, recsys and time series, inference kernels), what runs on which hardware (T4 x2,
P100, TPU, CPU-only kernels, Windows), and the evidence: about 40 techniques from top solutions' write-ups and code.
It reaches Claude in three ways. The router loads it for speed questions and for requests that build a model; each
domain skill carries its own short list of defaults; and a hook gives the defaults, and any slow pattern, when a
training or inference script is written in a workspace.

```bash
python src/train_image.py --bench     # data loader alone vs GPU step vs real loop: INPUT-BOUND or COMPUTE-BOUND
python -m kgkit bench --run "python src/train_gbdt.py --smoke" --run "python src/train_gbdt.py --smoke --device cuda"
python -m kgkit speedcheck src/train_image.py
```

Each bench is appended to `reports/bench.jsonl`, and the next one prints its speed-up and the settings that changed,
so a change is measured in minutes before it spends quota.

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
python -m kgkit gpu push .kaggle-gm/build/lr3e4-train  # quota check + -t cap + run log (confirm first: spends quota)
python -m kgkit gpu wait <user>/lr3e4-train --collect    # background; collect = ledger + GPU-hours + diagnosis
# decide each screen against a control screen of the baseline at the same reduced settings
python -m kgkit ledger compare artifacts/lr3e4 artifacts/control --truth data/train.csv:y --folds data/folds.csv:fold --fold 0
```

What it does for you:
- **Both T4s busy.** One process per (job, fold), one per GPU. Two folds or two screens share a quota hour.
- **Same code, same folds.** Your `src/`, `kgkit`, `competition.json` and `data/folds.csv` are embedded
  in the kernel, the split packed column by column (700k rows of `id,fold` take ~0.3 MB) and checked cell for
  cell in the session before anything runs; a split too big even packed comes from a dataset (`--split-from`).
  Inputs are linked into `./data`, so the command runs unchanged and the imported ledger records carry the
  same `folds_hash`. With `--input-layout nested` each dataset and kernel output gets its
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
  `NvidiaTeslaT4Highmem`, which `kaggle kernels pull -m` can record) and a kernel source of 1 MB or more (both
  answered with a bare 400) before anything reaches Kaggle, and names the shape to ask for or the biggest parts
  of the script (`build` warns about them). It warns when Kaggle accepts a push but leaves inputs out. kgkit's
  Kaggle calls print Kaggle's reason for a refused request (`Kaggle's reason: ...`), which the Kaggle CLI drops.
- **Any kernel.** `gpu push <dir> --hours H` takes notebook, inference and diagnostics kernels too.
- **TPU when the GPU quota is gone.** `--accelerator TpuV5E8` runs one process per TPU chip on the separate
  20 h/week TPU quota; the skill covers the torch_xla port and the 10-24 h queue.

## Verification

- **696 pytest tests** (some run only when LightGBM, PyTorch, timm or transformers are installed) cover kgkit (including paired comparisons, the backlog, feature search, multi-level/residual stacking, notebook review, competition pages, upload checks and repo adoption), all templates end to end (tiny synthetic data, plus an
  offline tiny BERT), the remote GPU runner executed in a simulated `/kaggle` layout (build, run on two
  fake GPUs, resume, prune, collect, `[kgmon]` resource lines), and the hooks driven through `run.sh` exactly as Claude Code calls them.
  The Kaggle pane's own tests (`claude plugin test`) draw it on the terminal and desktop surfaces and parse the
  runner's `[kgmon]` lines.
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
python -m pytest tests -q -n auto   # in parallel (pytest-xdist); without -n auto in one process
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
  skills/<name>/SKILL.md               22 knowledge skills + 18 kg-* commands
  agents/*.md                          9 subagents
  hooks/                               hooks.json, run.sh launcher, stdlib-only Python hooks
  hooks/monitor/                       the Kaggle pane mod (TypeScript hooks module) and its tests
  types/index.d.ts                     the pane's state types
  kgkit/                               the toolkit (python -m kgkit) and the remote GPU kernel runner
  templates/                           training / inference templates, CLAUDE.md template
  evals/                               plugin eval suite
scripts/bump_version.py                version check / release bump (used by CI)
tests/                                 pytest suite
```

## Data and network access

The plugin has no server of its own and collects nothing. Its network traffic goes to Kaggle under
your own account, through the official `kaggle` CLI and the API token you set up yourself, apart from
one read-only question to GitHub (below). The hooks make no network calls.

**Asked of GitHub (github.com).** `kgkit audit-repo` (also run by `kgkit adopt`) asks whether your
repository's GitHub remote is public: `gh repo view <owner>/<repo> --json visibility,forkCount` with your own
`gh` login, or, without `gh`, an anonymous `GET https://api.github.com/repos/<owner>/<repo>`. Only the
repository's name is sent; `--offline` (or `KGKIT_OFFLINE=1`) skips it.

**Exchanged with Kaggle (kaggle.com)**
- Competition data: `kaggle competitions files / download / leaderboard / submissions /
  submission-limits / pages` (`kgkit pages`).
- Submissions: `kaggle competitions submit` uploads your submission file. Claude confirms before it
  submits, and the hooks block files that fail validation.
- GPU training (`/kg-gpu`, `kgkit gpu`): `kaggle kernels push` uploads a private kernel containing
  your workspace's `src/`, `kgkit`, `competition.json` and `data/folds.csv`. `kaggle kernels status /
  logs / output` fetch its results, and `kaggle quota` reads your weekly GPU quota.
- Code-competition notebooks and model weights (`/kg-kernel`): `kaggle kernels push` and
  `kaggle datasets create / version`, private unless you make them public. After the deadline
  `/kg-showcase` audits them and asks before making any of them public.
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
- `.kaggle-gm/` in the workspace: experiment ledger, idea backlog, submission log, GPU-run log,
  the last quota snapshot, and the last competition-data audit (`audit.json`: paths and counts, no
  contents).
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
