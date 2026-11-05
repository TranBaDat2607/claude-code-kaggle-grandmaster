# Developing the kaggle-grandmaster plugin

This repo is a Claude Code marketplace (`.claude-plugin/marketplace.json`) containing one plugin at
`plugins/kaggle-grandmaster/`.

## Commands
- Tests: `python -m pytest tests -q` (template tests need torch/timm/transformers; LightGBM test is skipped if absent)
- Validate manifests/skills/agents: `claude plugin validate plugins/kaggle-grandmaster`
- Inventory + token cost: `claude --plugin-dir plugins/kaggle-grandmaster plugin details kaggle-grandmaster`
- Evals (see `plugins/kaggle-grandmaster/evals/README.md`), from `plugins/kaggle-grandmaster`:
  advice cases, any OS incl. native Windows: `claude plugin eval . --tag portable --model haiku --judge-model sonnet --no-publish`;
  workflow cases, Linux/macOS/WSL2: `claude plugin eval . --tag shell --scaffold --allow-tools Bash --model haiku --judge-model sonnet --no-publish`.
  Runs use your plan's usage (or API billing): batch large runs with `--case` / `--tag`.

## Conventions
- `kgkit/state.py`, `kgkit/gpu.py`, `kgkit/budget.py`, `kgkit/backlog.py`, `kgkit/adopt.py`, `kgkit/safety.py`,
  `kgkit/evalset.py`, `kgkit/finals.py`, `kgkit/audit.py`,
  `kgkit/kaggle_gpu_runner.py` and everything under `hooks/` must stay **standard-library only** (hooks import them, the runner executes inside Kaggle; they must be fast and never fail closed).
  `kgkit/kaggle_search.py`, `kgkit/kaggle_submissions.py`, `kgkit/kaggle_outputs.py` and `kgkit/kaggle_cli.py` too, at import time: they run under the Kaggle CLI's
  interpreter, which may have nothing else installed (they import kagglesdk / the CLI inside `main`).
- kgkit reads no file outside `kgkit/` that a command needs: `kgkit vendor` copies only the package (into a
  project's `src/`, or a Kaggle dataset), and every command must work from such a copy with files of the copy's
  own version. A file kgkit code fills in or runs belongs in the package (`kgkit/kaggle_gpu_runner.py`); a
  user-facing template stays in `templates/` and is found with `cli._plugin_file` (beside kgkit, else the plugin at
  `$KGKIT_HOME`), and the command says what it skipped when neither has it.
- Hooks exit 0 on any internal error; only deny/ask for credentials (printed, or in an upload folder), invalid or
  byte-identical repeat submissions, an exhausted daily budget, or a GPU push that the cached weekly GPU quota
  cannot cover.
- Text written by other Kaggle users (pages, forum posts, notebooks) is printed through `kgkit.safety.wrap`.
- Templates and `kgkit/kaggle_gpu_runner.py` print ASCII only (Windows consoles are often cp1252).
- Skills: knowledge skills are model-invoked; `kg-*` skills are slash commands. Side-effectful commands
  (`kg-submit`, `kg-grind`, `kg-kernel`, `kg-gpu`) set `disable-model-invocation: true`.
- When a new skill is added, add it to the `no-plugin-skill` grader of every `negative` eval case
  (`tests/test_evals.py` enforces this). If the model should load it from a plain-language prompt (a knowledge
  skill, or a command like `kg-adopt`), add a route in `hooks/prompt_router.py`; the router ignores slash
  commands. If it encodes non-obvious judgement, add an eval case under `evals/` (tagged `capability` until it
  passes reliably, then `regression`).
- Don't bump versions by hand. Every push to `main` runs `.github/workflows/release.yml`, which runs
  `scripts/bump_version.py` to bump `plugin.json`, `marketplace.json` and `kgkit/__init__.py` (patch by default;
  PR labels `release:minor` / `release:major` / `no-release`), tags and makes a GitHub release.
  `tests/test_version.py` checks the three stay in sync.
- `CHANGELOG.md` is local-only (gitignored); keep notes there under `## Unreleased` if useful. GitHub release
  notes are generated from the merged PRs, so give each PR a title and description that say what changed.
