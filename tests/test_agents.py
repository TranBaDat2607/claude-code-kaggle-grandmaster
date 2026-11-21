"""#124: the plugin's agents get the skills they rely on, and a model chosen for their work.

A subagent inherits no skills from the conversation that starts it, and these agents have no Skill tool, so a
skill their instructions point to reached them only when it was preloaded (``skills:``, its full text injected at
startup) or read from its file. Reading and assembling agents run on Sonnet; agents that judge, write training
code or can let a leak through run on the session's model."""

import re
from pathlib import Path

import pytest
import yaml

PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "kaggle-grandmaster"
AGENTS = sorted((PLUGIN / "agents").glob("*.md"))
SKILLS = {p.parent.name for p in (PLUGIN / "skills").glob("*/SKILL.md")}

PRELOAD = {
    "competition-analyst": ["competition-recon", "competition-strategy"],
    "solution-researcher": ["competition-recon"],
    "data-detective": ["validation-strategy"],
    "validation-auditor": ["validation-strategy", "cv-lb-debugging"],
    "error-analyst": ["grandmaster-playbook"],
    "feature-engineer": ["tabular-mastery"],
    "experiment-runner": ["grandmaster-playbook"],
    "ensemble-architect": ["ensembling"],
    "kernel-packager": ["code-competitions"],
}
MODEL = {  # sonnet: reading, summarising, assembling; inherit: judgement, training code, leak-sensitive work
    "competition-analyst": "sonnet", "solution-researcher": "sonnet", "data-detective": "sonnet",
    "kernel-packager": "sonnet",
    "validation-auditor": "inherit", "error-analyst": "inherit", "ensemble-architect": "inherit",
    "experiment-runner": "inherit", "feature-engineer": "inherit",
}
PRELOAD_BUDGET = 20_000  # characters of preloaded SKILL.md per agent, paid every time it starts


def _agent(path: Path) -> tuple[dict, str]:
    m = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", path.read_text(encoding="utf-8").replace("\r\n", "\n"), re.S)
    assert m, f"{path.name}: no frontmatter"
    return yaml.safe_load(m.group(1)), m.group(2)


def _skill_fields(name: str) -> dict:
    text = (PLUGIN / "skills" / name / "SKILL.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    return yaml.safe_load(re.match(r"\A---\n(.*?)\n---\n", text, re.S).group(1))


def test_every_agent_is_covered():
    assert {p.stem for p in AGENTS} == set(PRELOAD) == set(MODEL)


@pytest.mark.parametrize("path", AGENTS, ids=lambda p: p.stem)
def test_agent_preloads_its_skills_and_runs_on_its_model(path):
    fields, _ = _agent(path)
    assert fields.get("skills") == PRELOAD[path.stem], f"{path.stem}: skills {fields.get('skills')!r}"
    assert fields.get("model") == MODEL[path.stem], f"{path.stem}: model {fields.get('model')!r}"


@pytest.mark.parametrize("path", AGENTS, ids=lambda p: p.stem)
def test_preloaded_skills_exist_can_be_preloaded_and_fit_the_budget(path):
    fields, _ = _agent(path)
    size = 0
    for name in fields.get("skills") or []:
        assert name in SKILLS, f"{path.stem}: no skill {name!r}"
        # the subagent docs: a skill with disable-model-invocation cannot be preloaded
        assert _skill_fields(name).get("disable-model-invocation") is not True, f"{path.stem}: {name} is slash-only"
        size += len((PLUGIN / "skills" / name / "SKILL.md").read_text(encoding="utf-8"))
    assert size <= PRELOAD_BUDGET, f"{path.stem}: {size} characters of skills preloaded at every start"


@pytest.mark.parametrize("path", AGENTS, ids=lambda p: p.stem)
def test_every_skill_an_agent_names_reaches_it(path):
    """Preloaded, or read from its file (``${CLAUDE_PLUGIN_ROOT}`` is substituted in agent files): an agent without
    the Skill tool cannot load one by name."""
    fields, body = _agent(path)
    preloaded = set(fields.get("skills") or [])
    named = {m for m in re.findall(r"`([a-z][a-z0-9-]+)`", body) if m in SKILLS}
    for name in sorted(named - preloaded):
        assert f"${{CLAUDE_PLUGIN_ROOT}}/skills/{name}/SKILL.md" in body, (
            f"{path.stem} names the `{name}` skill but neither preloads it nor gives its path")
    for name in re.findall(r"\$\{CLAUDE_PLUGIN_ROOT\}/skills/([a-z0-9-]+)/", body):
        assert name in SKILLS, f"{path.stem}: path to a skill that does not exist: {name}"
