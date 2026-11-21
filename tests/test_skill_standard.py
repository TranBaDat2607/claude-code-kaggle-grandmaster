"""docs/skill-authoring.md: the mechanical rules of the skill authoring standard (IDs in each docstring).

Anthropic's skill guidance ([BP] platform.claude.com .../agent-skills/best-practices, [CC] code.claude.com/docs/en/skills)
turned into checks. Rules already enforced elsewhere (YAML parses, name = folder, side-effect commands,
compaction size, Guardrails first) stay in test_skills.py and test_compaction.py.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
PLUGIN = REPO / "plugins" / "kaggle-grandmaster"
STANDARD = REPO / "docs" / "skill-authoring.md"
FRONT = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.S)

SKILLS = sorted((PLUGIN / "skills").glob("*/SKILL.md"))
AGENTS = sorted((PLUGIN / "agents").glob("*.md"))
SUPPORTING = sorted(p for p in (PLUGIN / "skills").rglob("*.md") if p.name != "SKILL.md")

# F2: the frontmatter fields Claude Code reads (code.claude.com/docs/en/skills, .../sub-agents)
SKILL_FIELDS = {
    "name", "description", "when_to_use", "argument-hint", "arguments", "disable-model-invocation",
    "user-invocable", "allowed-tools", "disallowed-tools", "model", "effort", "context", "agent", "background",
    "hooks", "paths", "shell", "metadata", "license", "compatibility",
}
AGENT_FIELDS = {
    "name", "description", "tools", "disallowedTools", "model", "permissionMode", "mcpServers", "hooks",
    "maxTurns", "skills", "initialPrompt", "memory", "effort", "background", "isolation", "color",
}
NAME_MAX = 64  # F3 [BP]
DESC_MAX = 1024  # D1 [BP]: the Agent Skills limit, stricter than Claude Code's 1,536 listing cut
BODY_LINES_MAX = 500  # C1 [BP], [CC]
TOC_AFTER_LINES = 100  # R3 [BP]

# B1: total listing text of the skills Claude may load on its own. Lower it as descriptions shrink, never raise it.
# The target is 4,000 characters (#130); a 200K-context model has ~5,400 for every plugin and built-in skill together.
LISTING_CAP = 11_200

# D4: model-invocable commands whose description doesn't yet say when to use it. #130 rewrites them; remove each
# name here when its description gains a trigger clause (test_trigger_exemptions_are_current fails until you do).
NO_TRIGGER_YET = {
    "kg-baseline", "kg-cv", "kg-debug", "kg-eda", "kg-ensemble", "kg-experiment", "kg-final", "kg-ideas",
    "kg-recon", "kg-start", "kg-status",
}

XML = re.compile(r"<[A-Za-z/!][^>]*>")
FIRST_OR_SECOND_PERSON = re.compile(r"^(I|You|We)\b|\b(I can|I will|I'll|you can|you should|we can|Use this to)\b",
                                    re.I)
TRIGGER = re.compile(r"\bwhen(ever)?\b|\bUse (it |this )?(for|on|after|before|during|to|if)\b|\bafter\b|\bbefore\b",
                     re.I)
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATED = re.compile(rf"\b(before|after|until|by|as of|since) ({MONTHS}) 20\d\d\b")
BACKSLASH_PATH = re.compile(r"(?<![\\\w])[\w.-]+(\\[\w.-]+)+\.(py|md|sh|json|csv|ipynb|txt|yaml|toml)\b")
LINK = re.compile(r"\]\(([^)\s#]+)(?:#[^)]*)?\)")


def _front(path: Path) -> tuple[dict, str]:
    m = FRONT.match(path.read_text(encoding="utf-8").replace("\r\n", "\n"))
    assert m, f"{path}: no frontmatter"
    return yaml.safe_load(m.group(1)) or {}, m.group(2)


def _rel(path: Path) -> str:
    return path.relative_to(PLUGIN).as_posix()


def _listing(fields: dict) -> str:
    return " ".join(str(fields.get(k) or "").strip() for k in ("description", "when_to_use")).strip()


def _model_invocable() -> dict[str, dict]:
    out = {}
    for path in SKILLS:
        fields, _ = _front(path)
        if fields.get("disable-model-invocation") is not True:
            out[path.parent.name] = fields
    return out


@pytest.mark.parametrize("path", SKILLS + AGENTS, ids=_rel)
def test_frontmatter_fields_are_known(path):
    """F2: Claude Code ignores a misspelt field without an error."""
    fields, _ = _front(path)
    known = SKILL_FIELDS if path.name == "SKILL.md" else AGENT_FIELDS
    assert not set(fields) - known, f"{_rel(path)}: unknown frontmatter field(s) {sorted(set(fields) - known)}"


@pytest.mark.parametrize("path", SKILLS + AGENTS, ids=_rel)
def test_name_follows_the_rules(path):
    """F3: at most 64 characters, lowercase letters, digits and hyphens, no reserved word."""
    name = str(_front(path)[0].get("name", ""))
    assert len(name) <= NAME_MAX, f"{_rel(path)}: name is {len(name)} characters (max {NAME_MAX})"
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name), f"{_rel(path)}: name {name!r} is not lowercase-hyphenated"
    assert not re.search("anthropic|claude", name), f"{_rel(path)}: name {name!r} uses a reserved word"


@pytest.mark.parametrize("path", SKILLS + AGENTS, ids=_rel)
def test_description_length(path):
    """D1: not empty, at most 1,024 characters."""
    text = _listing(_front(path)[0])
    assert text, f"{_rel(path)}: empty description"
    assert len(text) <= DESC_MAX, f"{_rel(path)}: description is {len(text)} characters (max {DESC_MAX})"


@pytest.mark.parametrize("path", SKILLS + AGENTS, ids=_rel)
def test_description_has_no_xml(path):
    """D2: no XML tags or <placeholders>."""
    m = XML.search(_listing(_front(path)[0]))
    assert not m, f"{_rel(path)}: description contains {m.group(0)!r}"


@pytest.mark.parametrize("path", SKILLS + AGENTS, ids=_rel)
def test_description_point_of_view(path):
    """D3: third person or imperative, never I / you / we (it is injected into the system prompt)."""
    m = FIRST_OR_SECOND_PERSON.search(_listing(_front(path)[0]))
    assert not m, f"{_rel(path)}: description says {m.group(0)!r}: write it in the third person"


@pytest.mark.parametrize("name", sorted(_model_invocable()), ids=str)
def test_description_says_when(name):
    """D4: a skill Claude may load on its own says when to use it."""
    if name in NO_TRIGGER_YET:
        pytest.skip("exempt until #130 rewrites its description")
    text = _listing(_model_invocable()[name])
    assert TRIGGER.search(text), f"{name}: the description says what, not when: add 'Use when ...'"


def test_trigger_exemptions_are_current():
    """D4: the exemption list only shrinks; a fixed or removed skill must leave it."""
    skills = _model_invocable()
    stale = sorted(n for n in NO_TRIGGER_YET if n not in skills or TRIGGER.search(_listing(skills[n])))
    assert not stale, f"remove {stale} from NO_TRIGGER_YET: they now say when (or are no longer listed)"


def test_listing_fits_the_budget():
    """B1: the skill listing this plugin adds stays under a cap that only goes down."""
    total = sum(len(_listing(f)) for f in _model_invocable().values())
    assert total <= LISTING_CAP, (f"model-invocable descriptions total {total} characters (cap {LISTING_CAP}): "
                                  f"shorten one, or set disable-model-invocation on a command only users start")


@pytest.mark.parametrize("path", SKILLS, ids=_rel)
def test_body_under_500_lines(path):
    """C1: the body stays under 500 lines; detail goes to references/."""
    lines = _front(path)[1].count("\n")
    assert lines < BODY_LINES_MAX, f"{_rel(path)}: body is {lines} lines: move detail to references/"


@pytest.mark.parametrize("path", SKILLS + SUPPORTING, ids=_rel)
def test_no_dated_instructions(path):
    """C6: nothing that goes stale on a date; describe the current method."""
    m = DATED.search(path.read_text(encoding="utf-8"))
    assert not m, f"{_rel(path)}: {m.group(0)!r} goes stale: describe the current method (old ones under 'Old patterns')"


@pytest.mark.parametrize("path", SKILLS + SUPPORTING + AGENTS, ids=_rel)
def test_paths_use_forward_slashes(path):
    """C7: forward slashes in every path, even for Windows users."""
    m = BACKSLASH_PATH.search(path.read_text(encoding="utf-8"))
    assert not m, f"{_rel(path)}: {m.group(0)!r}: use forward slashes"


@pytest.mark.parametrize("path", SUPPORTING, ids=_rel)
def test_supporting_files_are_linked_from_skill_md(path):
    """R1: every supporting file is one link away from its own SKILL.md."""
    skill_dir = next(p for p in path.parents if (p / "SKILL.md").is_file())
    linked = {(skill_dir / target).resolve() for target in LINK.findall((skill_dir / "SKILL.md").read_text(encoding="utf-8"))}
    assert path.resolve() in linked, f"{_rel(path)} is not linked from {_rel(skill_dir / 'SKILL.md')}"


@pytest.mark.parametrize("path", SKILLS + [p for p in SUPPORTING if not p.name.endswith("-template.md")], ids=_rel)
def test_relative_links_resolve(path):
    """R2: every relative link resolves (output templates link into the repo they generate, so they are exempt)."""
    for target in LINK.findall(path.read_text(encoding="utf-8")):
        if re.match(r"[a-z]+:", target) or "<" in target:
            continue
        assert (path.parent / target).exists(), f"{_rel(path)}: link {target!r} does not resolve"


@pytest.mark.parametrize("path", SUPPORTING, ids=_rel)
def test_long_references_have_contents(path):
    """R3: a reference over 100 lines opens with a table of contents, for Claude's partial reads."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= TOC_AFTER_LINES:
        return
    head = "\n".join(lines[:15])
    assert re.search(r"^(#+ )?(Contents|Table of contents)\b", head, re.M | re.I), (
        f"{_rel(path)}: {len(lines)} lines with no 'Contents' in the first 15: add a table of contents")


def test_the_standard_names_tests_that_exist():
    """The standard's 'Enforced by' column points to real tests."""
    text = STANDARD.read_text(encoding="utf-8")
    for module, func in set(re.findall(r"`(test_\w+)\.py::(test_\w+)`", text)):
        source = (REPO / "tests" / f"{module}.py").read_text(encoding="utf-8")
        assert re.search(rf"^def {func}\(", source, re.M), f"docs/skill-authoring.md cites {module}.py::{func}, which is gone"
