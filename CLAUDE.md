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
  `kgkit/snapshot.py`, `kgkit/layout.py`, `kgkit/repoguard.py`, `kgkit/filelock.py`, `kgkit/grind.py`, `kgkit/permissions.py`, `kgkit/journey.py`, `kgkit/monitor.py`, `kgkit/kaggle_monitor_cell.py`, `kgkit/localrun.py`, `kgkit/vault.py` (at import time: numpy and pandas only inside its functions),
  `kgkit/kaggle_gpu_runner.py` and everything under `hooks/` must stay **standard-library only** (hooks import them, the runner executes inside Kaggle; they must be fast and never fail closed).
  `kgkit/kaggle_search.py`, `kgkit/kaggle_submissions.py`, `kgkit/kaggle_outputs.py` and `kgkit/kaggle_cli.py` too, at import time: they run under the Kaggle CLI's
  interpreter, which may have nothing else installed (they import kagglesdk / the CLI inside `main`).
- kgkit reads no file outside `kgkit/` that a command needs: `kgkit vendor` copies only the package (into a
  project's `src/`, or a Kaggle dataset), and every command must work from such a copy with files of the copy's
  own version. A file kgkit code fills in or runs belongs in the package (`kgkit/kaggle_gpu_runner.py`); a
  user-facing template stays in `templates/` and is found with `cli._plugin_file` (beside kgkit, else the plugin at
  `$KGKIT_HOME`), and the command says what it skipped when neither has it.
- `hooks/monitor/` is the exception to the standard-library rule: a TypeScript hooks module (a mod, listed under
  `modules` in `hooks.json`) for the `/kaggle-monitor` pane and the `/kg-update` command. It may pass `$` only to
  functions declared at the top of a file, writes `$.state` only under the plugin's own name (`kaggle-grandmaster`), and never reads a token's value:
  accounts are token files, given to the CLI as `KAGGLE_API_TOKEN=<path>`. Check it with
  `claude plugin validate plugins/kaggle-grandmaster` and `claude plugin test plugins/kaggle-grandmaster`
  (`tests/test_monitor_mod.py` runs the latter when `claude` is on PATH). The `[kgmon]` line the runner and
  `kgkit/kaggle_monitor_cell.py` print is its input, and so is the `KG_BAD_MACHINE` line (`kgkit.gpu.machine_problem`
  reads both too): change all of them together.
- Hooks exit 0 on any internal error; only deny/ask for credentials (printed, or in an upload folder), invalid or
  byte-identical repeat submissions, an exhausted daily budget, a GPU push that the cached weekly GPU quota
  cannot cover, a read of the sealed vault's labels (`~/.kaggle-gm/vault/`, `kgkit.vault.names_vault`: a
  command naming it, `hooks/pre_file.py` for the file tools), or a `git add`/`commit`/`push` that would send
  credentials or competition data (deny) or other never-public files (ask; `kgkit/repoguard.py`, whose lists the
  git hooks share).
- Text written by other Kaggle users (pages, forum posts, notebooks) is printed through `kgkit.safety.wrap`.
- Templates and `kgkit/kaggle_gpu_runner.py` print ASCII only (Windows consoles are often cp1252).
- Skills and agents follow `docs/skill-authoring.md` (Anthropic's skill guidance as rules, checked by
  `tests/test_skill_standard.py` in the `skill-standard` CI job): description says what and when, trigger words first,
  body under 500 lines, supporting files one link from `SKILL.md`, and a listing budget that only goes down.
- Skills: knowledge skills are model-invoked; `kg-*` skills are slash commands. Side-effectful commands
  (`kg-submit`, `kg-grind`, `kg-kernel`, `kg-gpu`, `kg-showcase`) set `disable-model-invocation: true`.
  `/kg-update` is not a skill: the hooks module declares it and runs `python -m kgkit.update --apply` (no model turn).
  Claude Code refuses a mod command named like one of the plugin's skills, so the two never share a name.
- Claude runs kgkit as `kgkit <cmd>`: `bin/kgkit` is on the Bash tool's PATH while the plugin is enabled (it runs
  the plugin's own kgkit with the first real Python), in subagents too. Skills, agents and hook messages never put a
  path in the command (`PYTHONPATH=`, `$KGKIT_HOME`: `tests/test_skills.py` checks). `bin/kgkit` is committed
  executable (mode 100755) and with LF like every text file (no `.gitattributes`: the directory rejects it).
- A `kg-*` command's `allowed-tools` pre-approves only the kgkit and Kaggle CLI subcommands it runs, one rule per
  kgkit subcommand: `Bash(kgkit <sub> *)`. Never a blanket `kgkit *`, and never a command that submits, pushes or publishes
  (`tests/test_skills.py` enforces this).
- Agents have no Skill tool and inherit no skills: an agent preloads the one or two skills it relies on (`skills:`,
  bare names: Claude Code resolves them in the agent's own plugin) and reads any other one by its path
  (`${CLAUDE_PLUGIN_ROOT}/skills/<name>/SKILL.md`). `model: sonnet` for reading, summarising and assembling agents,
  `inherit` for those that judge, write training code or can let a leak through. `tests/test_agents.py` holds the map.
- When a new skill is added, add it to the `no-plugin-skill` grader of every `negative` eval case
  (`tests/test_evals.py` enforces this). If the model should load it from a plain-language prompt (a knowledge
  skill, or a command like `kg-adopt`), add a route in `hooks/prompt_router.py`: descriptions alone do not reliably
  load skills (measured with `scripts/router_ab.py`: the expected skill loaded in 7% of eval runs without the router
  and 91% with it on Haiku, 50% and 98% on Sonnet). A route that names what the user is doing (phase, CV-LB gap,
  deciding an experiment, teaming, research) goes in `SITUATION`. For a slash-only command, add a `HINTS` entry instead: the router then tells Claude to suggest the
  command. The router ignores prompts that are themselves slash commands. If it encodes non-obvious judgement, add an
  eval case under `evals/` (tagged `capability` until it passes reliably, then `regression`);
  `tests/test_router_ab.py` checks the router names one of each case's expected skills.
- Don't bump versions by hand. Every push to `main` runs `.github/workflows/release.yml`, which runs
  `scripts/bump_version.py` to bump `plugin.json`, `marketplace.json` and `kgkit/__init__.py` (patch by default;
  PR labels `release:minor` / `release:major` / `no-release`), tags and makes a GitHub release.
  `tests/test_version.py` checks the three stay in sync.
- `CHANGELOG.md` is local-only (gitignored); keep notes there under `## Unreleased` if useful. GitHub release
  notes are generated from the merged PRs, so give each PR a title and description that say what changed.
