"""The `claude plugin eval` suite under plugins/kaggle-grandmaster/evals stays well-formed.

These checks need no model: case layout, tags, grader types, skill names that exist, should-not-trigger cases
that list every skill, and the shell fixture that seeds a competition workspace.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "kaggle-grandmaster"
EVALS = PLUGIN / "evals"
SKILLS = {p.parent.name for p in (PLUGIN / "skills").glob("*/SKILL.md")}
GRADER_TYPES = {"regex", "tool_used", "tool_order", "file_exists", "llm", "baseline"}
READ_ONLY_TOOLS = {"Read", "Glob", "Grep", "NotebookRead", "Skill", "AskUserQuestion", "Agent", "TodoWrite"}
FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)
SKILL_MATCH = re.compile(r"\(\?:kaggle-grandmaster:\)\?\(\?:([\w|-]+)\)")


def cases() -> list[Path]:
    return sorted(p.parent for p in EVALS.glob("*/prompt.md"))


def front(path: Path) -> dict[str, str]:
    m = FRONT.match(path.read_text(encoding="utf-8"))
    assert m, f"{path} has no frontmatter"
    out = {}
    for line in m.group(1).splitlines():
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


def as_list(value: str) -> list[str]:
    return [v.strip() for v in value.strip("[]").split(",") if v.strip()]


def graders(case: Path) -> dict[str, dict[str, str]]:
    return {g.stem: front(g) for g in sorted((case / "graders").glob("*.md"))}


def test_suite_has_cases():
    assert len(cases()) >= 40


@pytest.mark.parametrize("case", cases(), ids=lambda p: p.name)
def test_case_layout(case):
    fm = front(case / "prompt.md")
    tags = as_list(fm.get("tags", ""))
    assert len({"portable", "shell"} & set(tags)) == 1, f"{case.name}: tag it portable or shell"
    assert len({"regression", "capability"} & set(tags)) == 1, f"{case.name}: tag it regression or capability"
    assert fm.get("description"), f"{case.name}: add a description"
    tools = set(as_list(fm.get("allowed_tools", "")))
    if "portable" in tags:  # runs on native Windows: no shell, no scaffold
        assert tools <= READ_ONLY_TOOLS, f"{case.name}: portable cases use read-only tools only"
        assert not (case / "case.yaml").exists()
    else:
        assert "Bash" in tools
        yaml = (case / "case.yaml").read_text(encoding="utf-8")
        assert f"name: {case.name}\n" in yaml and 'schema_version: "1.1"' in yaml
        script = re.search(r"scaffold_script:\s*(\S+)", yaml).group(1)
        assert "_fixtures/make_workspace.sh" in (case / script).read_text(encoding="utf-8")

    gs = graders(case)
    assert gs, f"{case.name}: no graders"
    for name, g in gs.items():
        assert g.get("type") in GRADER_TYPES, f"{case.name}/{name}: type {g.get('type')!r}"
    # each case says whether the plugin produced the result (a skill, or a hook for side-effecting commands),
    # except should-not-trigger cases, whose check is that no skill fires
    if "negative" in tags:
        assert gs.get("no-plugin-skill", {}).get("arm") == "both"
    else:
        assert "skill-fired" in gs or "hook-fired" in gs, f"{case.name}: add a skill-fired grader"
        assert any(g["type"] in ("llm", "regex") and g.get("arm") != "with-only" for g in gs.values()), (
            f"{case.name}: no scored grader on the result")


@pytest.mark.parametrize("case", cases(), ids=lambda p: p.name)
def test_skill_names_exist(case):
    for name, g in graders(case).items():
        if g.get("tool") != "Skill":
            continue
        m = SKILL_MATCH.search(g.get("input_match", ""))
        assert m, f"{case.name}/{name}: input_match should name skills"
        named = set(m.group(1).split("|"))
        assert named <= SKILLS, f"{case.name}/{name}: unknown skills {named - SKILLS}"


def test_negative_cases_list_every_skill():
    negatives = [c for c in cases() if "negative" in as_list(front(c / "prompt.md")["tags"])]
    assert len(negatives) >= 3
    for case in negatives:
        g = graders(case)["no-plugin-skill"]
        assert (g["min"], g["max"]) == ("0", "0")
        named = set(SKILL_MATCH.search(g["input_match"]).group(1).split("|"))
        assert named == SKILLS, f"{case.name}: add the new skills {SKILLS - named} to its no-plugin-skill grader"


@pytest.mark.parametrize("case", cases(), ids=lambda p: p.name)
def test_regex_patterns_compile(case):
    for name, g in graders(case).items():
        for key in ("pattern", "input_match"):
            if key in g:
                re.compile(g[key].strip("'"))


needs_bash = pytest.mark.skipif(sys.platform == "win32" or not shutil.which("bash"),
                                reason="the fixture is a bash script (Linux/macOS/WSL)")


def _run_fixture(cwd: Path, home: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "HOME": str(home),
           "PATH": os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])}
    return subprocess.run(["bash", str(EVALS / "_fixtures" / "make_workspace.sh"), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)


@needs_bash
def test_fixture_refuses_outside_an_eval_workspace(tmp_path):
    r = _run_fixture(tmp_path, tmp_path, "--creds")
    assert r.returncode != 0 and "refusing" in r.stderr
    assert not any(tmp_path.iterdir())


@needs_bash
def test_fixture_builds_the_workspace(tmp_path):
    ws, home = tmp_path / "claude-eval-ws", tmp_path / "claude-eval-home"
    ws.mkdir()
    home.mkdir()
    r = _run_fixture(ws, home, "--candidate", "--budget-used", "--creds")
    assert r.returncode == 0, r.stderr

    sys.path.insert(0, str(PLUGIN))
    from kgkit import state as S

    st = S.load(ws)
    assert st.slug == "kgeval-binary-auc" and st.daily_submissions == 5
    ledger = (ws / ".kaggle-gm" / "ledger.jsonl").read_text(encoding="utf-8")
    assert "0002_lgbm-f2" in ledger and "0003_lgbm-f3" in ledger
    assert S.submissions_today(ws) == 5
    assert '"KEEP"' in (ws / ".kaggle-gm" / "comparisons.jsonl").read_text(encoding="utf-8")
    assert (home / ".kaggle" / "kaggle.json").is_file()
    # the invalid file stays invalid: the case grading relies on it
    sub = (ws / "subs" / "sub_0003.csv").read_text(encoding="utf-8").splitlines()
    assert sub[0] == "id,prediction" and len(sub) == 500

    # never overwrite an existing credential file
    ws2 = tmp_path / "claude-eval-ws2"
    ws2.mkdir()
    r = _run_fixture(ws2, home, "--creds")
    assert r.returncode != 0 and "not overwriting" in r.stderr
