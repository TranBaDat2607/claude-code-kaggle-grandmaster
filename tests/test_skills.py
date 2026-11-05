"""The kg-* commands' frontmatter (#70): `allowed-tools` pre-approves the kgkit and Kaggle CLI calls each command
makes, scoped to their subcommands, in the form a permission rule can match, and never one that submits, pushes or
publishes."""

import re
from pathlib import Path

import pytest

PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "kaggle-grandmaster"
COMMANDS = sorted(p.parent.name for p in (PLUGIN / "skills").glob("kg-*/SKILL.md"))
FRONT = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.S)
BASE = {"Read", "Glob", "Grep", "Agent"}
PINNED = 'Bash(PYTHONPATH="${{CLAUDE_PLUGIN_ROOT}}" python -m kgkit {} *)'
RUN_FORM = 'PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python -m kgkit <command>'
# what a turn must never run unprompted: a submission, a push, an upload, anything public, quota spent
OUTWARD = re.compile(r"\bsubmit\b|\bpush\b|datasets (create|version)|models? (create|instances)|\bgit\b|\bgh\b|"
                     r"\bvendor\b|\brecheck\b|visibility")
SIDE_EFFECTS = {"kg-submit", "kg-grind", "kg-kernel", "kg-gpu", "kg-showcase"}


def _skill(name: str) -> tuple[dict, list[str], str]:
    text = (PLUGIN / "skills" / name / "SKILL.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    m = FRONT.match(text)
    assert m, f"{name}: no frontmatter"
    fields, rules, key = {}, [], None
    for line in m.group(1).splitlines():
        if line.startswith("  - ") and key == "allowed-tools":
            rules.append(line[4:].strip().strip("'"))
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields, rules, m.group(2)


def test_there_are_kg_commands():
    assert len(COMMANDS) == 18 and "kg-showcase" in COMMANDS


@pytest.mark.parametrize("name", COMMANDS)
def test_allowed_tools_are_scoped_and_matchable(name):
    fields, rules, body = _skill(name)
    assert rules, f"{name}: allowed-tools must be a YAML list of rules"
    assert BASE & set(rules) >= {"Read", "Glob", "Grep"}
    tools = [r for r in rules if r not in BASE]
    for r in tools:
        assert re.fullmatch(r"(Bash|PowerShell)\(.+ \*\)", r), f"{name}: {r!r} is not a scoped Bash/PowerShell rule"
        assert not OUTWARD.search(r), f"{name}: {r!r} would run an outward-facing command without a prompt"
        assert not re.search(r"kgkit \*\)$", r), f"{name}: {r!r} pre-approves every kgkit command (gpu push too)"
    subs = {m.group(1) for r in tools for m in [re.fullmatch(r"Bash\(python -m kgkit (.+) \*\)", r)] if m}
    assert subs, f"{name}: no kgkit command pre-approved"
    for sub in subs:  # the form the session brief and this skill run, and a vendored copy in Bash and PowerShell
        assert PINNED.format(sub) in rules and f"PowerShell(python -m kgkit {sub} *)" in rules, (name, sub)
    for r in tools:
        m = re.fullmatch(r"Bash\(kaggle (.+) \*\)", r)
        if m:
            assert f"PowerShell(kaggle {m.group(1)} *)" in rules, (name, r)
    assert RUN_FORM in body, f"{name}: say which form is pre-approved, or every kgkit call prompts"
    assert (fields.get("disable-model-invocation") == "true") == (name in SIDE_EFFECTS), name


def test_showcase_files_and_the_brief_point_to_it(tmp_path):
    from kgkit import state as S

    fields, _, body = _skill("kg-showcase")
    for ref in re.findall(r"\]\(([\w.-]+\.md)\)", body):
        assert (PLUGIN / "skills" / "kg-showcase" / ref).is_file(), ref
    assert "readme-template.md" in body and "writeup-template.md" in body
    assert "audit-repo" in body and "kgkit final" in body and "ask before each public step" in body.lower()
    st = S.CompetitionState(slug="demo", deadline="2020-01-01")
    assert "ENDED — run /kg-postmortem, then /kg-showcase" in S.deadline_line(st)
    _, _, post = _skill("kg-postmortem")
    assert "ledger sync-lb" in post and "kgkit final" in post and "predictiveness" in post and "/kg-showcase" in post
