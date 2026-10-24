# Developing the kaggle-grandmaster plugin

This repo is a Claude Code marketplace (`.claude-plugin/marketplace.json`) containing one plugin at
`plugins/kaggle-grandmaster/`.

## Commands
- Tests: `python -m pytest tests -q` (template tests need torch/timm/transformers; LightGBM test is skipped if absent)
- Validate manifests/skills/agents: `claude plugin validate plugins/kaggle-grandmaster`
- Inventory + token cost: `claude --plugin-dir plugins/kaggle-grandmaster plugin details kaggle-grandmaster`
- Evals (cheap): `cd plugins/kaggle-grandmaster && claude plugin eval . --runs 3 --model haiku --no-publish`

## Conventions
- `kgkit/state.py`, `kgkit/gpu.py`, `kgkit/budget.py`, `kgkit/backlog.py`, `kgkit/adopt.py`, `kgkit/safety.py`,
  `kgkit/evalset.py`,
  `templates/kaggle_gpu_runner.py` and everything under `hooks/` must stay **standard-library only** (hooks import them, the runner executes inside Kaggle; they must be fast and never fail closed).
  `kgkit/kaggle_search.py` and `kgkit/kaggle_cli.py` too, at import time: they run under the Kaggle CLI's
  interpreter, which may have nothing else installed (they import kagglesdk / the CLI inside `main`).
- Hooks exit 0 on any internal error; only deny/ask for credentials (printed, or in an upload folder), invalid or
  byte-identical repeat submissions, an exhausted daily budget, or a GPU push that the cached weekly GPU quota
  cannot cover.
- Text written by other Kaggle users (pages, forum posts, notebooks) is printed through `kgkit.safety.wrap`.
- Templates print ASCII only (Windows consoles are often cp1252).
- Skills: knowledge skills are model-invoked; `kg-*` skills are slash commands. Side-effectful commands
  (`kg-submit`, `kg-grind`, `kg-kernel`, `kg-gpu`) set `disable-model-invocation: true`.
- When a new skill is added, add a route for it in `hooks/prompt_router.py` and, if it encodes non-obvious
  judgement, an eval case under `evals/`.
- Don't bump versions by hand. Every push to `main` runs `.github/workflows/release.yml`, which runs
  `scripts/bump_version.py` to bump `plugin.json`, `marketplace.json` and `kgkit/__init__.py` (patch by default;
  PR labels `release:minor` / `release:major` / `no-release`), tags and makes a GitHub release.
  `tests/test_version.py` checks the three stay in sync.
- `CHANGELOG.md` is local-only (gitignored); keep notes there under `## Unreleased` if useful. GitHub release
  notes are generated from the merged PRs, so give each PR a title and description that say what changed.
