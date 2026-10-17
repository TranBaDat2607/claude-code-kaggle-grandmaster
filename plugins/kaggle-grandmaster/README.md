# kaggle-grandmaster

Claude Code plugin for competing at Kaggle Grandmaster level. See the repository README for
full documentation.

## Install

From the official Claude plugin directory:

```
/plugin install kaggle-grandmaster@claude-plugins-official
```

Or from the GitHub repository:

```
/plugin marketplace add TranBaDat2607/claude-code-kaggle-grandmaster
/plugin install kaggle-grandmaster@claude-code-kaggle-grandmaster
```

Each merge to the repository's `main` publishes a new version. To receive them, turn on auto-update for
this marketplace (`/plugin` → **Marketplaces**) or run `claude plugin update kaggle-grandmaster@claude-code-kaggle-grandmaster`.

## Use


Start a competition: `/kaggle-grandmaster:kg-start <competition-slug>`. Already mid-competition in your own
repo: `/kaggle-grandmaster:kg-adopt <competition-slug>` writes only `.kaggle-gm/` and imports past
submissions and experiments, leaving your layout alone.

- `skills/`: 21 knowledge playbooks (including `competition-strategy`: choosing competitions, the solo
  gold, teaming and merging) + 17 `kg-*` slash commands
- `agents/`: 9 specialist subagents
- `hooks/`: session brief (accepted baseline, backlog top, GPU quota), prompt→skill router, submission +
  GPU-quota + upload guard (credentials never printed or uploaded, repeated files caught), submission /
  GPU-run logger
- `kgkit/`: Python toolkit (`python -m kgkit --help`):
  - experiment loop: `ledger compare` (paired per-fold test + OOF bootstrap), `ledger decide / baseline /
    lineage / import`, `backlog`, `recheck` (does the gain survive a re-drawn fold split?)
  - modelling: `features search` (thousands of generated features, batch-screened), `blend` with hill
    climbing, multi-level and residual stacking, `--prune`
  - prior art: `pages` (the competition's pages as text), `kernels top / pull` (notebook review with the
    lines to read before running it + local script), `discussions sync / top / search / solutions / read`
    (forum, live via the Kaggle CLI)
  - `gpu` for quota-aware training on Kaggle GPUs and TPUs (reservations released when runs finish,
    any kernel folder pushable with a cap, busy-session and input checks); `datasets pull` (per-file
    fallback when a dataset archive 404s); `whoami`; `config` (e.g. `validation=holdout+lb`)
- `templates/`: tabular (GBDTs + linear / kNN / SVM / MLP), image and transformer training, remote GPU
  training runner, offline inference kernel, workspace CLAUDE.md
- `evals/`: `claude plugin eval` suite (15 cases)

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
  logs / output` fetch its results, `kaggle quota` reads your weekly GPU and TPU quota, and `kaggle
  kernels list -m` + `status` count your busy sessions before a push.
- Datasets: `kaggle datasets files / download` (`kgkit datasets pull`).
- Code-competition notebooks and model weights (`/kg-kernel`): `kaggle kernels push` and
  `kaggle datasets create / version`, private unless you make them public.
- Public notebooks: `kaggle kernels list / pull` (`kgkit kernels top / pull`).
- Forums: `kaggle competitions topics` / `topic-messages` (CLI 2.2.4+), and for finished competitions
  `kaggle datasets download kaggle/meta-kaggle`, Kaggle's public forum export (`kgkit discussions --source meta`).

**Web reads.** `/kg-start`, `/kg-recon`, `/kg-ideas`, `/kg-grind`, `/kg-postmortem` and the
`competition-analyst` and `solution-researcher` agents use Claude's WebSearch and WebFetch to read
public competition pages, past solution write-ups and papers. They don't upload your files, and Claude
asks before each call unless you have allowed these tools.

**Credentials.** The plugin never uses or sends your Kaggle API key. `kgkit gpu` and `kgkit whoami`
take only your Kaggle username, from `KAGGLE_USERNAME`, the `username` field of `kaggle.json`, or the
`username` line of `kaggle config view` (token logins), to name your kernels. The PreToolUse hook blocks
commands that would print a credential (a file reader on `kaggle.json`, `~/.kaggle/access_token` or
`credentials.json`, Python printing more than their username field, `KAGGLE_KEY`, `kaggle auth
print-access-token`, `VERBOSE` Kaggle CLI calls) and dataset or model uploads whose folder holds
credential-like files. Text that only mentions those files (an issue body saved with a heredoc, a commit
message) is not blocked. `kgkit` drops
`VERBOSE` / `VERBOSE_OUTPUT` from the environment of its own Kaggle CLI calls. `kgkit gpu build`
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
