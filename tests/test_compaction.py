"""#125: after Claude Code compacts a conversation, each skill comes back with only its first 5,000 tokens (25,000
in all, the most recent first). So the plugin's skills stay under that size with their guardrails near the top, the
router forgets what it suggested in that session (so the next prompt loads the skills again), and the re-injected
session brief says so and shows a grind in progress."""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from kgkit import grind as GR
from kgkit import state as S

PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "kaggle-grandmaster"
SH = shutil.which("sh")
SKILL_CAP = 13_500  # characters: Claude Code's estimate is ~2.7 characters a token, 5,000 tokens survive compaction
GUARDRAILS_WITHIN = 4_000  # characters from the top: well inside what survives


def _hook(script: str, payload: dict) -> dict | None:
    r = subprocess.run([SH, str(PLUGIN / "hooks" / "run.sh"), script], input=json.dumps(payload),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    out = r.stdout.strip()
    return json.loads(out)["hookSpecificOutput"] if out else None


@pytest.mark.parametrize("path", sorted((PLUGIN / "skills").glob("*/SKILL.md")), ids=lambda p: p.parent.name)
def test_skills_survive_compaction_whole(path):
    text = path.read_text(encoding="utf-8")
    assert len(text) <= SKILL_CAP, (f"{path.parent.name}: {len(text)} characters; after compaction only about the "
                                    f"first {SKILL_CAP} come back: move detail to references/")
    m = re.search(r"^## Guardrails", text, re.M)
    if m:
        assert m.start() <= GUARDRAILS_WITHIN, f"{path.parent.name}: Guardrails at character {m.start()}: put them first"


def test_skill_references_exist():
    for path in (PLUGIN / "skills").glob("*/SKILL.md"):
        for ref in re.findall(r"\]\((references/[\w.-]+\.md)(?:#[\w-]+)?\)", path.read_text(encoding="utf-8")):
            assert (path.parent / ref).is_file(), f"{path.parent.name}: {ref} is missing"


@pytest.mark.skipif(SH is None, reason="needs a POSIX sh")
def test_a_compaction_resets_the_router_and_the_brief_says_why(tmp_path):
    S.CompetitionState(slug="demo", metric="auc", deadline="2099-01-01").save(tmp_path)
    sid = "compact-test-session"
    seen = Path(tempfile.gettempdir()) / "kaggle-gm-router" / f"{sid}.json"
    seen.parent.mkdir(parents=True, exist_ok=True)
    seen.write_text(json.dumps(["grandmaster-playbook", "kaggle-gpu"]))
    other = seen.with_name("another-session.json")
    other.write_text(json.dumps(["ensembling"]))
    out = _hook("session_start.py", {"hook_event_name": "SessionStart", "source": "startup", "session_id": sid,
                                     "cwd": str(tmp_path)})
    assert seen.is_file() and "compacted" not in out["additionalContext"]  # a fresh start keeps it
    out = _hook("session_start.py", {"hook_event_name": "SessionStart", "source": "compact", "session_id": sid,
                                     "cwd": str(tmp_path)})
    assert not seen.exists() and other.is_file()  # only this session's memory is cleared
    assert "compacted" in out["additionalContext"] and "Skill tool" in out["additionalContext"]
    other.unlink()


@pytest.mark.skipif(SH is None, reason="needs a POSIX sh")
def test_the_brief_shows_a_grind_in_progress(tmp_path):
    S.CompetitionState(slug="demo", metric="auc", deadline="2099-01-01").save(tmp_path)
    GR.start(tmp_path, "old-session", hours=4)
    ctx = _hook("session_start.py", {"hook_event_name": "SessionStart", "source": "startup", "session_id": "new",
                                     "cwd": str(tmp_path)})["additionalContext"]
    assert "grind in progress" in ctx.lower() and "/kg-grind resume" in ctx


def test_kg_grind_keeps_its_state_in_the_workspace():
    body = (PLUGIN / "skills" / "kg-grind" / "SKILL.md").read_text(encoding="utf-8")
    for cmd in ("kgkit grind start", "kgkit grind status", "kgkit grind note", "kgkit grind stop", "--resume"):
        assert cmd in body, cmd
    assert "${CLAUDE_SESSION_ID}" in body  # the session that drives it
    assert "'Bash(kgkit grind *)'" in (PLUGIN / "skills" / "kg-grind" / "SKILL.md").read_text(encoding="utf-8")
