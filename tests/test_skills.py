"""The kg-* commands' frontmatter (#70): `allowed-tools` pre-approves the kgkit and Kaggle CLI calls each command
makes, scoped to their subcommands, in the form a permission rule can match, and never one that submits, pushes or
publishes."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "kaggle-grandmaster"
COMMANDS = sorted(p.parent.name for p in (PLUGIN / "skills").glob("kg-*/SKILL.md"))
FRONT = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.S)
BASE = {"Read", "Glob", "Grep", "Agent"}
RUN_FORM = "Run kgkit with the Bash tool as `kgkit <command>`"  # bin/kgkit, on the Bash tool's PATH (#122)
OLD_FORMS = re.compile(r"PYTHONPATH=|\$KGKIT_HOME|kgkit home|\$env:PYTHONPATH")
# what a turn must never run unprompted: a submission, a push, an upload, anything public, quota spent
OUTWARD = re.compile(r"\bsubmit\b|\bpush\b|datasets (create|version)|models? (create|instances)|\bgit\b|\bgh\b|"
                     r"\bvendor\b|\brecheck\b|\bbench\b|\bpermissions\b|visibility")
SIDE_EFFECTS = {"kg-submit", "kg-grind", "kg-kernel", "kg-gpu", "kg-showcase", "kg-update"}


def _front(path: Path) -> tuple[dict, str]:
    """The YAML frontmatter of a skill or agent file, parsed as YAML (as Claude Code does), and the body."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    m = FRONT.match(text)
    assert m, f"{path}: no frontmatter"
    try:
        fields = yaml.safe_load(m.group(1))
    except yaml.YAMLError as e:  # Claude Code then loads the file with empty metadata (#120)
        pytest.fail(f"{path}: frontmatter is not valid YAML ({e}); quote a value that contains ': '")
    assert isinstance(fields, dict), f"{path}: frontmatter is not a mapping"
    return fields, m.group(2)


def _skill(name: str) -> tuple[dict, list[str], str]:
    fields, body = _front(PLUGIN / "skills" / name / "SKILL.md")
    return fields, list(fields.get("allowed-tools") or []), body


FILES = sorted([*(PLUGIN / "skills").glob("*/SKILL.md"), *(PLUGIN / "agents").glob("*.md")])
DESC_CAP = 1536  # Claude Code truncates a description longer than this in the skill listing


@pytest.mark.parametrize("path", FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_frontmatter_parses_with_a_name_and_description(path):
    """#120: an unquoted ': ' in a description made the YAML invalid, and Claude Code silently loaded the skill with
    no name or description, so Claude never chose it on its own."""
    fields, _ = _front(path)
    expected = path.parent.name if path.name == "SKILL.md" else path.stem
    assert fields.get("name") == expected, f"{path}: name {fields.get('name')!r}, expected {expected!r}"
    desc = fields.get("description")
    assert isinstance(desc, str) and desc.strip(), f"{path}: no description"
    assert len(desc) <= DESC_CAP, f"{path}: description is {len(desc)} characters (listing cap {DESC_CAP})"


@pytest.mark.skipif(shutil.which("claude") is None, reason="needs the claude CLI on PATH")
def test_claude_plugin_validate_passes():
    """The authoritative check: Claude Code's own validator (it parses frontmatter the way the runtime does)."""
    r = subprocess.run([shutil.which("claude"), "plugin", "validate", str(PLUGIN)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=300)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "Validation failed" not in out, out[-4000:]


def test_there_are_kg_commands():
    assert len(COMMANDS) == 19 and "kg-showcase" in COMMANDS and "kg-update" in COMMANDS


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
    subs = {m.group(1) for r in tools for m in [re.fullmatch(r"Bash\(kgkit (.+) \*\)", r)] if m}
    assert subs, f"{name}: no kgkit command pre-approved"
    stale = [r for r in tools if "kgkit" in r and not re.fullmatch(r"Bash\(kgkit .+ \*\)", r)]
    assert not stale, f"{name}: kgkit is the command on PATH now (#122), not {stale}"
    for r in tools:
        m = re.fullmatch(r"Bash\(kaggle (.+) \*\)", r)
        if m:
            assert f"PowerShell(kaggle {m.group(1)} *)" in rules, (name, r)
    assert RUN_FORM in body, f"{name}: say which form is pre-approved, or every kgkit call prompts"
    assert (fields.get("disable-model-invocation") is True) == (name in SIDE_EFFECTS), name


INJECT = re.compile(r"!`([^`]+)`")  # Claude Code runs it when the skill loads and puts its output in the text
STATE_FIRST = ("kg-status", "kg-experiment", "kg-ideas", "kg-final")


@pytest.mark.parametrize("name", STATE_FIRST)
def test_state_arrives_with_the_skill(name):
    """#127: the workspace state is injected when the command loads, so its first turn is not a `kgkit status`."""
    _, rules, body = _skill(name)
    cmds = INJECT.findall(body)
    assert any(c.startswith("kgkit status") for c in cmds), f"{name}: inject the state with !`kgkit status ... || true`"


@pytest.mark.parametrize("name", COMMANDS)
def test_injected_commands_are_pre_approved_and_never_fail_the_load(name):
    """A `!` command that is not pre-approved asks before the skill loads; one that fails stops it loading."""
    _, rules, body = _skill(name)
    for cmd in INJECT.findall(body):
        assert cmd.rstrip().endswith("|| true"), f"{name}: !`{cmd}` must end with `|| true`"
        first = cmd.split("||")[0].strip()
        assert any(re.fullmatch(r"Bash\((.+) \*\)", r) and first.startswith(r[5:-3] + " ") for r in rules), (
            f"{name}: !`{cmd}` is not pre-approved by allowed-tools")
        assert not OUTWARD.search(first), f"{name}: !`{cmd}` would run an outward-facing command at load"


@pytest.mark.parametrize("name", ["kg-experiment", "kg-grind"])
def test_an_experiment_is_finished_in_one_call(name):
    """#127: compare, decide and close the backlog item with `kgkit exp finish`, not three separate calls."""
    _, rules, body = _skill(name)
    assert "Bash(kgkit exp *)" in rules and "kgkit exp finish" in body, name
    assert "kgkit ledger compare" not in body and "kgkit ledger decide <new>" not in body, name


@pytest.mark.parametrize("path", [*FILES, *sorted((PLUGIN / "skills").glob("*/references/*.md")),
                                  PLUGIN / "templates" / "CLAUDE.competition.md"],
                         ids=lambda p: f"{p.parent.name}/{p.name}")
def test_kgkit_is_run_as_the_command_on_path(path):
    """#122: one way to run kgkit. A path in the command (the brief's literal one, $KGKIT_HOME, ${CLAUDE_PLUGIN_ROOT})
    never matched every allow rule, and $KGKIT_HOME is unset in PowerShell and absent from subagents' shells."""
    text = path.read_text(encoding="utf-8")
    bad = [line.strip() for line in text.splitlines() if OLD_FORMS.search(line)]
    assert not bad, f"{path.relative_to(PLUGIN)}: {bad[:3]}"


def test_bin_kgkit_runs_the_plugins_kgkit_from_anywhere(tmp_path):
    launcher = PLUGIN / "bin" / "kgkit"
    data = launcher.read_bytes()
    assert data.startswith(b"#!/bin/sh")  # committed with LF (test_repo_files); Git Bash also runs a CRLF checkout
    mode = subprocess.run(["git", "ls-files", "-s", str(launcher)], capture_output=True, text=True,
                          cwd=PLUGIN).stdout.split()
    assert mode and mode[0] == "100755", f"bin/kgkit must be executable in git: {mode}"
    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("needs sh")
    r = subprocess.run([sh, str(launcher), "--help"], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "usage: kgkit" in r.stdout, r.stdout + r.stderr


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
