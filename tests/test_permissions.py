"""#123: a kg-* command's allowed-tools end at the user's next message and never reach its subagents, so the
bookkeeping kgkit calls get allow rules of their own, written once into the git-ignored settings.local.json."""

import json
import subprocess

import pytest

from kgkit import permissions as PM
from kgkit import state as S


def test_the_rules_hold_bookkeeping_only():
    rules = PM.rules()
    assert "Bash(kgkit status)" in rules and "Bash(kgkit ledger *)" in rules and "Bash(kgkit gpu quota *)" in rules
    for never in PM.NEVER:  # quota, publishing, commands it would run for us, the frozen split
        assert not any(r.startswith(f"Bash(kgkit {never}") for r in rules), never
    assert not any("kaggle " in r or r == "Bash(kgkit *)" for r in rules)
    assert len(rules) == len(set(rules))


def test_write_merges_once_and_keeps_every_other_setting(tmp_path):
    settings = tmp_path / ".claude" / "settings.local.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"permissions": {"allow": ["Bash(git status)"], "deny": ["Bash(rm *)"]},
                                    "env": {"X": "1"}}))
    path, added = PM.write(tmp_path)
    data = json.loads(path.read_text())
    assert path == settings and added == PM.rules()
    assert data["env"] == {"X": "1"} and data["permissions"]["deny"] == ["Bash(rm *)"]
    assert data["permissions"]["allow"][0] == "Bash(git status)"
    assert PM.write(tmp_path)[1] == [] and PM.missing(tmp_path) == []
    assert json.loads(settings.read_text()) == data  # a second run changes nothing


def test_write_never_overwrites_a_settings_file_it_cannot_read(tmp_path):
    settings = tmp_path / ".claude" / "settings.local.json"
    settings.parent.mkdir()
    settings.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        PM.write(tmp_path)
    assert settings.read_text() == "{not json"


def test_status_says_when_the_rules_are_missing(tmp_path):
    S.CompetitionState(slug="demo", metric="auc").save(tmp_path)
    assert "kgkit permissions --write" in PM.status_line(tmp_path)
    PM.write(tmp_path)
    assert PM.status_line(tmp_path) == ""


def test_the_settings_file_is_git_ignored(tmp_path, monkeypatch):
    empty = tmp_path / "empty-gitconfig"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))  # no global ignore file: the repo's .gitignore must do it
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    ws = tmp_path / "ws"
    ws.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    (ws / ".gitignore").write_text("data/\n")
    PM.write(ws)
    assert subprocess.run(["git", "check-ignore", "-q", ".claude/settings.local.json"], cwd=ws).returncode == 0
    PM.write(ws)
    assert (ws / ".gitignore").read_text().count("settings.local.json") == 1


def test_the_long_commands_offer_the_rules(plugin_root):
    for name in ("kg-start", "kg-adopt", "kg-grind", "kg-status"):
        body = (plugin_root / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
        assert "kgkit permissions --write" in body, name
