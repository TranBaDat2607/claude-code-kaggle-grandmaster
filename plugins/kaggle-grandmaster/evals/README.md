# kaggle-grandmaster eval suite

Cases for [`claude plugin eval`](https://code.claude.com/docs/en/plugin-evals): each directory is one case, a
realistic request plus graders. Every case runs in a fresh, isolated Claude Code session with only this plugin
loaded, and by default again without it, so the report shows what the plugin adds (`WITH`, `W/OUT`, `Δ`).

## What is in the suite

| Kind | Cases | Checks |
|---|---|---|
| Advice from UAT bugs | 18 cases from real UAT bugs: validation, experiment noise, final selection, ensembling, GPU/TPU quota, code competitions, teaming, prompt injection, showing the work after the deadline | The judgement the plugin was fixed to give. These should stay near 100% |
| Advice per domain skill | 16 cases, one or more per domain skill: time series, NLP/LLM, recsys, audio, simulation, optimisation, tuning, tabular, vision, DL training, recon, strategy, Kaggle CLI, metrics, boosters | What the skills teach that a model without them often misses |
| Compute efficiency (`compute`) | 3 cases: an image run that is input-bound, a 50M-row tabular job, LLM inference over a 9-hour limit | Hardware-aware speed advice: measure first, fix the actual bottleneck, know what runs on which Kaggle GPU (no RAPIDS on a P100, no bf16 on a T4) |
| Should not trigger (`negative`) | 3 cases: "competition" in a business sense, a game leaderboard, production ML at work | No plugin skill loads. Scored in both arms, so over-triggering shows as a negative `Δ` |
| Workflow (`shell`) | 5 cases in a seeded competition workspace: status report, invalid submission, exhausted daily budget, a credential request, a keep/discard decision | What Claude does with the real files, hooks and ledger, not only what it says |

How this follows Anthropic's guidance (the [plugin eval docs](https://code.claude.com/docs/en/plugin-evals) and
[Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)):

- **Cases come from real failures.** The regression cases are bugs found in UAT. Each new capability case
  checks a non-obvious point that its skill makes.
- **Each case has one grader on the result and one on the process.** The result grader is an `llm` rubric with
  explicit PASS/FAIL. The process grader is `skill-fired` (`tool_used: Skill`) or, for commands that skills may
  not run, `hook-fired`. Process graders are indicators: they are not scored in a with/without run.
- **One judge per dimension.** A rubric with several independent requirements is split into graders, which also
  gives partial credit.
- **Deterministic graders where possible.** The shell cases use regex and `tool_used` checks for anything a
  pattern can decide, for example that the API key never appears and that no Python-API submit was attempted.
- **The set is balanced.** Should-trigger and should-not-trigger cases are both present.
- **Outcomes are graded, not paths.** A workflow case passes whether the hook blocked the submit or Claude
  caught the bad file first.
- **Every run is isolated.** The scaffold builds each run's workspace from scratch, and Claude cannot read the
  eval directory.

## Tags

Of the 45 cases, 35 are `regression` and 10 are `capability` (`multilabel-f1-thresholds`, `optim-santa-no-cv`,
`showcase-after-deadline`, the three `compute` cases and four workflow cases). Many regression cases also score 1.0 without the plugin on
Sonnet, the model already knows the point. There they guard against the plugin making answers worse rather than measure what it adds; the cases
with a positive `Δ` are the ones that show the plugin's value.

Every case carries exactly one of each pair, plus an area tag (`validation`, `gpu`, `safety`, `time-series`, ...):

- `portable`: read-only tools only (`Read`, `Glob`, `Grep`, `Skill`). Runs anywhere, including native Windows.
- `shell`: needs Bash, a `--scaffold` workspace and Claude Code's OS sandbox. Runs on Linux, macOS, or WSL2 on
  Windows. Native Windows has no sandbox backend, so Claude Code refuses shell-granting runs there.
- `regression`: should pass about 100% of the time. A drop means something broke.
- `capability`: newer or harder cases. A capability case that passes reliably on both Haiku and Sonnet over
  several full runs gets its tag changed to `regression` (and a case that is always 1.0 with **and** without the
  plugin should be made harder or dropped: it measures nothing).

## Running

Every run and every `llm` judge call is a real model call on your account: on a claude.ai plan it counts
against your usage limits, with an API key it is billed. Runs that hit a usage or rate limit score 0, so run
large suites in batches (`--case`, `--tag`) and re-run after the limit resets. From the plugin root
(`plugins/kaggle-grandmaster`):

```bash
# Advice + negative cases (any OS, including native Windows PowerShell)
claude plugin eval . --tag portable --model haiku --judge-model sonnet --no-publish

# Workflow cases (Linux / macOS / WSL2): seed the workspace, grant Bash (it runs inside the sandbox)
claude plugin eval . --tag shell --scaffold --allow-tools Bash --model haiku --judge-model sonnet --no-publish

# The same on Sonnet (the model most users run)
claude plugin eval . --tag portable --model sonnet --judge-model sonnet --no-publish

# Iterate on one case: one run, no baseline arm (`--case` takes one glob; the last one given wins)
claude plugin eval . --case ws-keep-or-discard --scaffold --allow-tools Bash --runs 1 --ablation none --model haiku
```

Pin `--judge-model sonnet` for runs you compare or report. The default judge is Haiku, and on this suite it
failed several correct answers. Keep the judge the same across runs, or score differences can come from the
judge rather than the plugin.

Run the `portable` and `shell` sets as separate commands: `--allow-tools Bash` applies to every case in a run,
and the advice cases are meant to be answered without a shell.

On Windows, run the `portable` command in PowerShell. Run the `shell` command inside WSL2 with Claude Code
installed there (on Linux and WSL2 the sandbox needs `bubblewrap` and `socat`: `sudo apt install bubblewrap
socat`).

### Python inside the shell cases

The sandbox that confines Bash in an eval run cannot read your home directory, apart from the plugin itself. A
Python installed under `$HOME` (conda, pyenv, uv, a venv) therefore fails to start inside the run, and Claude
cannot run `python -m kgkit ...` there. The cases are written for this: hooks run outside the sandbox and use
your normal Python, the session-start brief carries the workspace state, and the graders check outcomes (Claude
can read `.kaggle-gm/` files instead). If you want Claude to be able to run kgkit inside these runs, put a
Python with numpy and pandas outside `$HOME` first on `PATH` for the shell command.

The scaffold itself (`_fixtures/make_workspace.sh`) runs outside the sandbox with the Python on your `PATH`,
which needs numpy and pandas (kgkit's own requirements). It refuses to run anywhere but an eval run's temporary
workspace, and writes the fake `kaggle.json` only into the run's temporary `HOME`.

## Reading the report

- `Δ > 0` on a case: the plugin helps there. `Δ ≈ 0` with `skill-fired` failing: the skill did not load on this
  phrasing (check the skill's `description` and the route in `hooks/prompt_router.py`). `Δ ≈ 0` with the skill
  firing: the model already knows this, or the rubric is too easy.
- `Δ < 0` with `skill-fired` passing: suspect the judge first (`--judge-model sonnet`), then the skill text.
- A negative case with `no-plugin-skill` failing: the plugin over-triggers on that wording.
- Read the transcripts of failing runs (`--keep-temp` keeps them) before changing a rubric: a failure should be
  fair, and a rubric should not reject a correct answer that is phrased differently.

## Adding a case

1. `mkdir evals/<name>/graders` with a `prompt.md` (frontmatter `description`, `tags`, `max_turns`,
   `allowed_tools`; body phrased the way a user would type it, without naming the skill) and at least one
   result grader with concrete PASS and FAIL conditions.
2. Add `graders/skill-fired.md` naming the skill(s) that should produce the answer (copy one from another case).
3. A shell case also needs `case.yaml` with `scaffold_script: setup.sh`, and a `setup.sh` that calls
   `../_fixtures/make_workspace.sh` (add an option there if the case needs other state).
4. A new skill must also be added to the `no-plugin-skill` grader of every `negative` case, and given a route in
   `hooks/prompt_router.py`. `python -m pytest tests/test_evals.py` checks the layout, tags, skill names and the
   negative cases' skill lists.
5. Run it once with `--runs 1 --ablation none`, read the transcript, then run it with the baseline arm.
