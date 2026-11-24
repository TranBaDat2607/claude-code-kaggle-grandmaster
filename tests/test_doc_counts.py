"""The counts the docs state match the repo, so they can't drift (as test_version.py does for versions).

Each check finds a number in a doc by a pattern and compares it with what is on disk. A count that changes with
every PR (pytest tests) is not stated at all.
"""

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PLUGIN = REPO / "plugins" / "kaggle-grandmaster"
EVALS = PLUGIN / "evals"
REFERENCE = REPO / "docs" / "reference.md"
README = REPO / "README.md"
PLUGIN_README = PLUGIN / "README.md"
EVALS_README = EVALS / "README.md"


def _skills(commands: bool) -> int:
    return sum(p.parent.name.startswith("kg-") == commands for p in (PLUGIN / "skills").glob("*/SKILL.md"))


def _mod_commands() -> int:
    """The kg-* slash commands the hooks module declares (`/kg-update`): commands the person types like the skills."""
    src = "\n".join(p.read_text(encoding="utf-8") for p in (PLUGIN / "hooks" / "monitor").glob("*.ts*")
                    if ".test." not in p.name)
    return len(set(re.findall(r"name: '(kg-[a-z-]+)'", src)))


def _commands() -> int:
    return _skills(commands=True) + _mod_commands()


def _agents() -> int:
    return len(list((PLUGIN / "agents").glob("*.md")))


def _hooks() -> int:
    config = json.loads((PLUGIN / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    scripts = {
        re.search(r"run\.sh\"? (\S+\.py)", h["command"]).group(1)
        for groups in config["hooks"].values()
        for group in groups
        for h in group["hooks"]
    }
    return len(scripts)


def _modules() -> int:
    return sum(not p.name.startswith("__") for p in (PLUGIN / "kgkit").glob("*.py"))


def _case_tags() -> list[str]:
    out = []
    for prompt in sorted(EVALS.glob("*/prompt.md")):
        m = re.search(r"^tags:\s*(.*)$", prompt.read_text(encoding="utf-8"), re.M)
        out.append(m.group(1) if m else "")
    return out


def _cases() -> int:
    return len(_case_tags())


def _tagged(tag: str) -> int:
    return sum(bool(re.search(rf"\b{tag}\b", tags)) for tags in _case_tags())


# (doc, pattern with one number group, actual count)
CLAIMS = [
    (REFERENCE, r"\| \*\*Knowledge skills\*\* \| (\d+) \|", lambda: _skills(commands=False)),
    (REFERENCE, r"\| \*\*Slash commands\*\* \| (\d+) \|", _commands),
    (REFERENCE, r"\| \*\*Specialist agents\*\* \| (\d+) \|", _agents),
    (REFERENCE, r"\| \*\*Hooks\*\* \| (\d+) \|", _hooks),
    (REFERENCE, r"\| \*\*`kgkit` toolkit\*\* \| (\d+) modules \|", _modules),
    (REFERENCE, r"\| \*\*Evals\*\* \| (\d+) cases \|", _cases),
    (REFERENCE, r"(\d+) knowledge skills \+ \d+ kg-\* commands", lambda: _skills(commands=False)),
    (REFERENCE, r"\d+ knowledge skills \+ (\d+) kg-\* commands", lambda: _skills(commands=True)),
    (REFERENCE, r"agents/\*\.md\s+(\d+) subagents", _agents),
    (README, r"\*\*(\d+) playbook skills\*\*", lambda: _skills(commands=False)),
    (README, r"\*\*(\d+) slash commands\*\*", _commands),
    (README, r"\*\*(\d+) specialist agents\*\*", _agents),
    (README, r"\*\*(\d+) safety hooks\*\*", _hooks),
    (PLUGIN_README, r"`claude plugin eval` suite \((\d+) cases\)", _cases),
    (EVALS_README, r"Of the (\d+) cases, \d+ are `regression`", _cases),
    (EVALS_README, r"Of the \d+ cases, (\d+) are `regression`", lambda: _tagged("regression")),
    (EVALS_README, r"are `regression` and (\d+) are `capability`", lambda: _tagged("capability")),
]


@pytest.mark.parametrize(
    "doc,pattern,actual", CLAIMS, ids=[f"{d.relative_to(REPO).as_posix()}:{p[:40]}" for d, p, _ in CLAIMS]
)
def test_doc_count_matches_repo(doc, pattern, actual):
    m = re.search(pattern, doc.read_text(encoding="utf-8"))
    assert m, f"{doc.relative_to(REPO)}: the count this test checks is gone (pattern {pattern!r}); update CLAIMS"
    assert int(m.group(1)) == actual(), f"{doc.relative_to(REPO)} says {m.group(1)}, the repo has {actual()}"


def test_evals_readme_kinds_add_up():
    text = EVALS_README.read_text(encoding="utf-8")
    table = text.split("## What is in the suite", 1)[1].split("\n\n", 2)[1]
    rows = re.findall(r"^\|[^|]+\| (\d+) cases", table, re.M)
    assert rows and sum(map(int, rows)) == _cases(), f"evals/README.md table rows {rows} != {_cases()} cases"


@pytest.mark.parametrize("doc", [REFERENCE, README, PLUGIN_README], ids=lambda p: p.relative_to(REPO).as_posix())
def test_no_pytest_count(doc):
    # it changes with nearly every PR: describe the suite instead of counting it
    assert not re.search(r"\b\d+ (pytest )?tests\b", doc.read_text(encoding="utf-8")), doc
