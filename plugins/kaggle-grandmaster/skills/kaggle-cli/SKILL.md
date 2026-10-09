---
name: kaggle-cli
description: Using the official Kaggle CLI/API from Claude Code — authentication, downloading competition data, submitting (and reading scores), listing submissions and leaderboards, pushing/pulling notebooks (kernels), creating and versioning datasets and models, and fetching episode replays. Use whenever interacting with kaggle.com programmatically.
---

# Kaggle CLI / API

## Authentication

- Needs an API token: either `~/.kaggle/kaggle.json` (`{"username": ..., "key": ...}`, chmod 600
  on Unix; `C:\Users\<user>\.kaggle\kaggle.json` on Windows) or env vars `KAGGLE_USERNAME` /
  `KAGGLE_KEY` (newer CLI versions also accept a `KAGGLE_API_TOKEN` access token).
- **Never print, commit, or paste the key.** If auth fails, ask the user to create a token at
  kaggle.com → Settings → API and place it themselves.
- You must have joined the competition and accepted its rules on the website before downloading
  or submitting (403 otherwise) — only the user can do that.
- Check installation: `kaggle --version`; install with `pip install kaggle`.

## Competitions

```bash
kaggle competitions list -s "keyword"                 # search
kaggle competitions files <slug>                      # files
kaggle competitions download <slug> -p data/          # all files (zip) ; -f <file> for one
unzip -q -o data/<slug>.zip -d data/                  # (PowerShell: Expand-Archive data/<slug>.zip data/)
kaggle competitions submit <slug> -f subs/x.csv -m "0007_lgbm_te | CV 0.81234"
kaggle competitions submissions <slug> --format json  # history with public scores (poll after submitting)
kaggle competitions leaderboard <slug> --show         # top of public LB; --download for full csv
```
Submission messages should always contain the ledger experiment id and CV so the LB score can be
attached back (`python -m kgkit ledger lb <id> <score>`). Code competitions: submit via the
notebook: `kaggle competitions submit <slug> -k <user>/<kernel> -v <version> -f submission.csv -m "..."`
(verified against Kaggle CLI 2.2), or through the website.

## Notebooks (kernels)

```bash
kaggle kernels list --competition <slug> --sort-by voteCount --page-size 20
kaggle kernels pull <user>/<kernel> -p notebooks/ref/ -m     # -m also pulls metadata
kaggle kernels output <user>/<kernel> -p artifacts/ref/      # download a public notebook's outputs
kaggle kernels init -p kernels/infer                          # creates kernel-metadata.json
kaggle kernels push -p kernels/infer -t 21600 --accelerator NvidiaTeslaT4   # -t caps the run (s)
kaggle kernels status <user>/<kernel>
kaggle kernels logs <user>/<kernel>                           # execution log (-f to follow)
kaggle quota --format json                                    # weekly GPU/TPU quota + refreshAt
```

`kernel-metadata.json` keys: `id`, `title`, `code_file`, `language` (python), `kernel_type`
(script|notebook), `is_private`, `enable_gpu`, `enable_tpu`, `enable_internet`, `machine_shape`
(`NvidiaTeslaT4` = 2x T4, `NvidiaTeslaP100`, `Tpu1VmV38`; `--accelerator` overrides it),
`dataset_sources`, `competition_sources`, `kernel_sources`, `model_sources`.

To *study* a public notebook, prefer `python -m kgkit kernels pull <owner>/<slug>`. It pulls the
notebook with its metadata into `ref/<slug>/`, writes a `REVIEW.md` (CV scheme, model families, printed
scores, external inputs, with `--check-access` testing that attached datasets are reachable) and a
local script with Kaggle paths mapped to the workspace. `kgkit kernels top` lists the competition's top
notebooks.

## Discussions

The CLI has no discussion commands. `python -m kgkit discussions sync|top|search|solutions|read` reads
the forum from Meta Kaggle (`kaggle datasets download kaggle/meta-kaggle -f ForumTopics.csv`, Kaggle's
official daily export; cached in `~/.kaggle-gm/meta-kaggle`). It works for any competition slug,
including finished ones (`discussions solutions --competition <slug>` for write-ups). The title must slugify
to the id's slug. `kaggle kernels output` pages through all output files (`--page-size 200` means fewer
requests), and `--file-pattern <regex>` filters them, e.g. to skip weights.

For GPU **training** runs use `python -m kgkit gpu build/push/wait/collect` (skill `kaggle-gpu`). It
adds the quota check, a `-t` cap, deadline-aware checkpoints, both T4s busy and ledger import.

## Datasets & models

```bash
kaggle datasets init -p kernels/weights                 # dataset-metadata.json (title, id user/slug, licenses)
kaggle datasets create -p kernels/weights --dir-mode zip
kaggle datasets version -p kernels/weights -m "fold models v3" --dir-mode zip
kaggle datasets download <owner>/<slug> -p data/ext --unzip
kaggle models instances versions create ...              # Kaggle Models (for model hubs) — see `kaggle models -h`
```
Keep weight datasets private until the competition rules require publishing.

## Simulation competitions

Episodes/replays of top agents can be listed and downloaded through the Kaggle API endpoints for
episodes (used for imitation learning); respect rate limits and cache downloads.

## Etiquette & limits

- Respect daily submission limits; the plugin's hook warns before submitting and validates files.
- Poll status endpoints with a delay (≥ 30–60 s), not tight loops.
- Submitting is an outward-facing action that consumes the user's quota — confirm with the user
  unless they've explicitly authorised autonomous submissions.
