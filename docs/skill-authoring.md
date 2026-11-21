# Skill authoring standard

The rules every `SKILL.md` and agent file in `plugins/kaggle-grandmaster/` follows. They come from Anthropic's
official guidance, cited in the **Source** column:

- **[BP]** [Skill authoring best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)
  (Claude Platform docs)
- **[CC]** [Extend Claude with skills](https://code.claude.com/docs/en/skills) (Claude Code docs)

The rest of the repo's conventions (`kgkit <cmd>` with no paths, scoped `allowed-tools`, agent models, eval cases for
new skills) are in [CLAUDE.md](../CLAUDE.md) and are not repeated here.

**Enforced by** names the test that fails in CI when a rule breaks. `review` means a person checks it in the PR,
because no pattern can decide it. The CI job `skill-standard` (`.github/workflows/tests.yml`) runs these tests on
every PR, and so does the full `pytest` job.

Contents: [How Claude sees a skill](#how-claude-sees-a-skill) · [Frontmatter](#frontmatter) ·
[Description](#description) · [Listing budget](#listing-budget) · [Body](#body) ·
[Supporting files](#supporting-files) · [Agents](#agents) · [Testing a change](#testing-a-change) ·
[PR checklist](#pr-checklist)

## How Claude sees a skill

1. **At session start** only `name` and `description` (+ `when_to_use`) of each skill Claude may load on its own
   go into the skill listing. That listing has a budget of about 1 % of the model's context window. Over budget,
   Claude Code keeps every name but drops the descriptions of the least-used skills [CC]. A skill without a
   description is rarely chosen.
2. **When the skill is invoked** its body enters the conversation once and stays there across turns, so every line
   is a recurring cost [CC].
3. **Supporting files** cost nothing until Claude reads them. Claude may read only the top of a long file
   (`head -100`) [BP].
4. **After auto-compaction** Claude Code re-attaches the first 5,000 tokens of each invoked skill, within 25,000
   tokens for all skills together, most recent first [CC].

The rules below follow from these four facts.

## Frontmatter

| ID | Rule | Source | Enforced by |
|---|---|---|---|
| F1 | The file starts with `---` on line 1, and the frontmatter is valid YAML that parses to a mapping. Quote a value that contains `: `. Invalid YAML loads the skill **with no fields at all**, silently (#120) | CC | `test_skills.py::test_frontmatter_parses_with_a_name_and_description` |
| F2 | Only fields Claude Code knows. A misspelt field is ignored without an error | CC | `test_skill_standard.py::test_frontmatter_fields_are_known` |
| F3 | `name` equals the folder name (an agent's: the file name), has at most 64 characters, uses lowercase letters, digits and hyphens only, and contains neither `anthropic` nor `claude` | BP | `test_skills.py` (folder), `test_skill_standard.py::test_name_follows_the_rules` |
| F4 | A command with side effects (submits, pushes, publishes, spends quota, edits the plugin) sets `disable-model-invocation: true`, so only the user starts it | CC | `test_skills.py::test_allowed_tools_are_scoped_and_matchable` |
| F5 | Knowledge skills have noun-phrase names by domain (`time-series`, `validation-strategy`); commands are `kg-<verb or noun>`. Never vague (`helper`, `utils`) | BP | review |

## Description

The description is what Claude uses to choose a skill from possibly 100+ others [BP].

| ID | Rule | Source | Enforced by |
|---|---|---|---|
| D1 | Not empty, at most **1,024** characters (the Agent Skills limit; Claude Code cuts `description` + `when_to_use` at 1,536 in the listing) | BP, CC | `test_skill_standard.py::test_description_length` |
| D2 | No XML tags or angle-bracket placeholders (`<name>`): the platform rejects them, Claude Code shows them escaped | BP, CC | `test_skill_standard.py::test_description_has_no_xml` |
| D3 | Third person ("Diagnoses a CV-LB gap...") or an imperative; never *I*, *you* or *we*: the text is injected into the system prompt | BP | `test_skill_standard.py::test_description_point_of_view` |
| D4 | It says **what** the skill does and **when** to use it: a skill Claude may load on its own has a trigger clause (`Use when ...`, `when the user ...`, `before/after ...`) | BP | `test_skill_standard.py::test_description_says_when` |
| D5 | The key use case and the words a user would type come **first**: the listing may cut the end | CC | review |
| D6 | Specific terms (`OOF`, `adversarial validation`, `QWK`), not vague ones ("Helps with competitions") | BP | review |
| D7 | No inventory of subtopics: the body holds those. One sentence on what, one on when, is the shape | BP, CC | review, and the budget below |

Good: `Diagnoses a gap between cross-validation and the public leaderboard. Use when CV and LB disagree, a score jumps
unexpectedly, or a submission scores far from its CV.`

Avoid: `Systematic diagnosis — adversarial validation, fold audit, leak checks, metric parity, post-processing
parity, distribution shift, LB noise estimation, ...` (a list, no trigger, and the trigger words come last).

## Listing budget

| ID | Rule | Source | Enforced by |
|---|---|---|---|
| B1 | The total listing text (`description` + `when_to_use`) of the skills Claude may load on its own stays under a cap that only goes **down**. On a 200K-context model the whole listing (every plugin, every built-in skill) gets about 2,000 tokens, roughly 5,400 characters at Claude Code's ~2.7 characters per token. This plugin's target is **4,000 characters** (#130) | CC | `test_skill_standard.py::test_listing_fits_the_budget` |
| B2 | A command that the user always starts by name, and that neither the prompt router nor another skill loads, is a candidate for `disable-model-invocation: true`: it then costs no listing budget [CC]. Update `SIDE_EFFECTS` in `tests/test_skills.py` with it | CC | review |

Check a real session with `/context` (the Skills row) or `/doctor` on a Sonnet model, not only the 1M-context one.

## Body

| ID | Rule | Source | Enforced by |
|---|---|---|---|
| C1 | Under **500 lines**, and at most 13,500 characters so compaction re-attaches it whole (5,000 tokens) | BP, CC | `test_skill_standard.py::test_body_under_500_lines`, `test_compaction.py::test_skills_survive_compaction_whole` |
| C2 | The most important instructions first: `## Guardrails` (when present) within the first 4,000 characters | CC | `test_compaction.py::test_skills_survive_compaction_whole` |
| C3 | Only what Claude does not already know. State what to do, not how or why at length. For each paragraph ask: "Does this justify its token cost?" | BP, CC | review |
| C4 | One default, with an escape hatch ("Use X. For Y, use Z instead"), not a menu of equal options | BP | review |
| C5 | One term per concept throughout (always "fold", never also "split" / "partition" for the same thing) | BP | review |
| C6 | Nothing that goes stale on a date ("before August 2025, use..."). Describe the current method, and put an old one under an "Old patterns" heading | BP | `test_skill_standard.py::test_no_dated_instructions` |
| C7 | Forward slashes in every path (`references/tabular.md`), even for Windows users | BP | `test_skill_standard.py::test_paths_use_forward_slashes` |
| C8 | Multi-step work as numbered steps; a quality-critical step has a feedback loop ("run the check, fix, run it again; continue only when it passes") | BP | review |
| C9 | Free (prose) where many approaches are valid; exact (a command to run unchanged) where a step is fragile: submissions, folds, pushes | BP | review |

## Supporting files

| ID | Rule | Source | Enforced by |
|---|---|---|---|
| R1 | Every supporting file is linked **directly** from its skill's `SKILL.md` (one level deep). A reference may also point to a sibling, as long as `SKILL.md` links that sibling too | BP | `test_skill_standard.py::test_supporting_files_are_linked_from_skill_md` |
| R2 | Every relative link in a skill or reference resolves. Output templates (`*-template.md`) are exempt: their links are for the repo they generate | BP | `test_skill_standard.py::test_relative_links_resolve` |
| R3 | A reference file over 100 lines opens with a table of contents (a `Contents:` line or heading in its first 15 lines) | BP | `test_skill_standard.py::test_long_references_have_contents` |
| R4 | Descriptive file names grouped by domain (`references/vision.md`), never `doc2.md` | BP | review |
| R5 | For a script, say whether Claude **runs** it ("Run `kgkit eda`") or **reads** it; prefer a kgkit command to code Claude writes for a deterministic step | BP | review |

## Agents

Agent files follow F1-F3 (the name is the file name), D1-D3 and C7. Their descriptions tell Claude when to delegate,
so D4 does not apply. Models, tools and preloaded skills are set by CLAUDE.md and checked by `tests/test_agents.py`.

## Testing a change

- **Evals first** [BP]: a new judgement a skill teaches gets an eval case under `plugins/kaggle-grandmaster/evals/`
  (tagged `capability` until it passes reliably). Evals, not reading, decide whether a change worked.
- **Every model you ship for** [BP]: what Opus infers, Haiku may need spelled out. Run the portable evals on Haiku
  (see CLAUDE.md) after shortening a description or a body.
- **Triggering is not success** [CC]: a skill that loads and gives the wrong advice still fails.

## PR checklist

For any PR that adds or changes a skill or an agent:

- [ ] `python -m pytest tests/test_skill_standard.py tests/test_skills.py tests/test_compaction.py -q` passes
- [ ] The description: trigger words first, what + when, no inventory (D5-D7)
- [ ] The body: only what Claude doesn't know, one default per choice, one term per concept (C3-C5)
- [ ] Steps numbered, with a check-fix-repeat loop where a mistake is costly (C8)
- [ ] A new skill is in every `negative` eval's `no-plugin-skill` grader and in `hooks/prompt_router.py` (CLAUDE.md)
- [ ] New judgement has an eval case; the portable evals still pass on Haiku
