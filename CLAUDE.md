# Developing the kaggle-grandmaster plugin

This repo is a Claude Code marketplace (`.claude-plugin/marketplace.json`) containing one plugin at
`plugins/kaggle-grandmaster/`.

## Commands
- Tests: `python -m pytest tests -q -n auto` (in parallel with pytest-xdist, from `requirements-dev.txt`: about 3 min
  instead of 7 on 12 cores; drop `-n auto` for one process; template tests need torch/timm/transformers; LightGBM
  test is skipped if absent)
- Validate manifests/skills/agents: `claude plugin validate plugins/kaggle-grandmaster`
- Inventory + token cost: `claude --plugin-dir plugins/kaggle-grandmaster plugin details kaggle-grandmaster`
- Evals (see `plugins/kaggle-grandmaster/evals/README.md`), from `plugins/kaggle-grandmaster`:
  advice cases, any OS incl. native Windows: `claude plugin eval . --tag portable --model haiku --judge-model sonnet --no-publish`;
  workflow cases, Linux/macOS/WSL2: `claude plugin eval . --tag shell --scaffold --allow-tools Bash --model haiku --judge-model sonnet --no-publish`.
  Runs use your plan's usage (or API billing): batch large runs with `--case` / `--tag`.

## Conventions
- `kgkit/state.py`, `kgkit/gpu.py`, `kgkit/budget.py`, `kgkit/backlog.py`, `kgkit/adopt.py`, `kgkit/safety.py`,
  `kgkit/evalset.py`, `kgkit/finals.py`, `kgkit/audit.py`, `kgkit/progress.py`, `kgkit/speedcheck.py`, `kgkit/bench.py`,
  `kgkit/kaggle_gpu_runner.py` and everything under `hooks/` must stay **standard-library only** (hooks import them, the runner executes inside Kaggle; they must be fast and never fail closed).
  `kgkit/kaggle_search.py`, `kgkit/kaggle_submissions.py`, `kgkit/kaggle_outputs.py` and `kgkit/kaggle_cli.py` too, at import time: they run under the Kaggle CLI's
  interpreter, which may have nothing else installed (they import kagglesdk / the CLI inside `main`).
- kgkit reads no file outside `kgkit/` that a command needs: `kgkit vendor` copies only the package (into a
  project's `src/`, or a Kaggle dataset), and every command must work from such a copy with files of the copy's
  own version. A file kgkit code fills in or runs belongs in the package (`kgkit/kaggle_gpu_runner.py`); a
  user-facing template stays in `templates/` and is found with `cli._plugin_file` (beside kgkit, else the plugin at
  `$KGKIT_HOME`), and the command says what it skipped when neither has it.
- `hooks/monitor/` is the exception to the standard-library rule: a TypeScript hooks module (a mod, listed under
  `modules` in `hooks.json`) for the `/kaggle-monitor` pane. It may pass `$` only to functions declared at the top of
  a file, writes `$.state` only under the plugin's own name (`kaggle-grandmaster`), and never reads a token's value:
  accounts are token files, given to the CLI as `KAGGLE_API_TOKEN=<path>`. Check it with
  `claude plugin validate plugins/kaggle-grandmaster` and `claude plugin test plugins/kaggle-grandmaster`
  (`tests/test_monitor_mod.py` runs the latter when `claude` is on PATH). The `[kgmon]` line the runner and
  `templates/kaggle_monitor_cell.py` print is its input: change all three together.
- Hooks exit 0 on any internal error; only deny/ask for credentials (printed, or in an upload folder), invalid or
  byte-identical repeat submissions, an exhausted daily budget, or a GPU push that the cached weekly GPU quota
  cannot cover.
- Text written by other Kaggle users (pages, forum posts, notebooks) is printed through `kgkit.safety.wrap`.
- Templates and `kgkit/kaggle_gpu_runner.py` print ASCII only (Windows consoles are often cp1252).
- Skills: knowledge skills are model-invoked; `kg-*` skills are slash commands. Side-effectful commands
  (`kg-submit`, `kg-grind`, `kg-kernel`, `kg-gpu`, `kg-showcase`) set `disable-model-invocation: true`.
- A `kg-*` command's `allowed-tools` pre-approves only the kgkit and Kaggle CLI subcommands it runs. Each kgkit
  one is listed three times: `Bash(PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit <sub> *)`, the bare
  `Bash(python -m kgkit <sub> *)`, and the `PowerShell(...)` form of the bare one. The body says to run kgkit in
  the first form. An allow rule can't match the session brief's literal path, or a `$env:` assignment in
  PowerShell. Never a blanket `kgkit *`, and never a command that submits, pushes or publishes
  (`tests/test_skills.py` enforces this).
- When a new skill is added, add it to the `no-plugin-skill` grader of every `negative` eval case
  (`tests/test_evals.py` enforces this). If the model should load it from a plain-language prompt (a knowledge
  skill, or a command like `kg-adopt`), add a route in `hooks/prompt_router.py`. For a slash-only command, add a
  `HINTS` entry instead: the router then tells Claude to suggest the command. The router ignores prompts that are
  themselves slash commands. If it encodes non-obvious judgement, add an eval case under `evals/` (tagged
  `capability` until it passes reliably, then `regression`).
- Don't bump versions by hand. Every push to `main` runs `.github/workflows/release.yml`, which runs
  `scripts/bump_version.py` to bump `plugin.json`, `marketplace.json` and `kgkit/__init__.py` (patch by default;
  PR labels `release:minor` / `release:major` / `no-release`), tags and makes a GitHub release.
  `tests/test_version.py` checks the three stay in sync.
- `CHANGELOG.md` is local-only (gitignored); keep notes there under `## Unreleased` if useful. GitHub release
  notes are generated from the merged PRs, so give each PR a title and description that say what changed.
