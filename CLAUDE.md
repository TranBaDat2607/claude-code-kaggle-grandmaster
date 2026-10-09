# Developing the kaggle-grandmaster plugin

This repo is a Claude Code marketplace (`.claude-plugin/marketplace.json`) containing one plugin at
`plugins/kaggle-grandmaster/`.

## Commands
- Tests: `python -m pytest tests -q` (template tests need torch/timm/transformers; LightGBM test is skipped if absent)
- Validate manifests/skills/agents: `claude plugin validate plugins/kaggle-grandmaster`
- Inventory + token cost: `claude --plugin-dir plugins/kaggle-grandmaster plugin details kaggle-grandmaster`
- Evals (cheap): `cd plugins/kaggle-grandmaster && claude plugin eval . --runs 3 --model haiku --no-publish`

## Conventions
- `kgkit/state.py`, `kgkit/gpu.py`, `kgkit/budget.py`, `templates/kaggle_gpu_runner.py` and everything under `hooks/` must
  stay **standard-library only** (hooks import them, the runner executes inside Kaggle; they must be fast and never fail closed).
- Hooks exit 0 on any internal error; only deny/ask for credentials, invalid submissions, an exhausted daily budget,
  or a GPU push that the cached weekly GPU quota cannot cover.
- Templates print ASCII only (Windows consoles are often cp1252).
- Skills: knowledge skills are model-invoked; `kg-*` skills are slash commands. Side-effectful commands
  (`kg-submit`, `kg-grind`, `kg-kernel`, `kg-gpu`) set `disable-model-invocation: true`.
- When a new skill is added, add a route for it in `hooks/prompt_router.py` and, if it encodes non-obvious
  judgement, an eval case under `evals/`.
- Keep versions in `plugin.json`, `marketplace.json` and `kgkit/__init__.py` in sync.
