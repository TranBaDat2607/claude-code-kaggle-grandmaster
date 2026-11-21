"""/kg-update: update the plugin to the latest release without leaving the conversation. `kgkit update-check` compares
the running version with the latest GitHub release, finds this session's install in `claude plugin list --json`, and
prints the exact `claude plugin update` command; the slash command runs it (the user approves), and the user types
/reload-plugins. After the update, update-check says the new version waits for /reload-plugins and where it runs
from, and which `kgkit vendor` copies in the workspace are older."""

import json
import re
import subprocess
import uuid
from pathlib import Path

import pytest

from kgkit import update as U

ROOT_OLD = "/home/u/.claude/plugins/cache/claude-code-kaggle-grandmaster/kaggle-grandmaster/1.24.0"
ROOT_NEW = "/home/u/.claude/plugins/cache/claude-code-kaggle-grandmaster/kaggle-grandmaster/1.26.0"
ID = "kaggle-grandmaster@claude-code-kaggle-grandmaster"


def _release(v, *titles):
    body = "## What's Changed\n" + "".join(
        f"* {t} by @someone in https://github.com/{U.REPO}/pull/{100 + i}\n" for i, t in enumerate(titles))
    return {"tag_name": f"kaggle-grandmaster--v{v}", "name": f"kaggle-grandmaster {v}", "body": body + "\n**Full"}


RELEASES = [_release("1.26.0", "Kaggle pane: bad machines (fixes #113)", "ledger: decide --member (fixes #112)"),
            _release("1.25.0", "backlog set --first-experiment (fixes #111)"),
            _release("1.24.0", "accounts reach the model (fixes #106)")]


def _install(version="1.24.0", path=ROOT_OLD, project="/work/comp", scope="project", **kw):
    return {"id": ID, "version": version, "scope": scope, "enabled": True, "installPath": path,
            "projectPath": project, **kw}


@pytest.fixture
def fake(monkeypatch):
    """GitHub's releases and `claude plugin list --json`, as the test sets them."""
    state = {"releases": RELEASES, "installs": [_install(), _install("1.12.0", project="/work/old",
                                                                     path=ROOT_OLD.replace("1.24.0", "1.12.0"))]}
    monkeypatch.setattr(U, "_releases_json", lambda: state["releases"])
    monkeypatch.setattr(U, "_plugin_list", lambda: state["installs"])
    return state


def check(**kw):
    args = {"root": Path(ROOT_OLD), "cwd": Path("/work/comp"), "running": "1.24.0", "workspace": None}
    args.update(kw)
    return U.check(**args)


# ----------------------------------------------------------------------------- versions and notes
def test_versions_and_release_notes():
    assert U.parse_version("kaggle-grandmaster--v1.24.0") == (1, 24, 0) == U.parse_version("1.24.0")
    assert U.parse_version("v2.0") == (2, 0) and U.parse_version("nightly") is None
    assert U.notes(RELEASES[1]) == ["backlog set --first-experiment (fixes #111)"]


# ----------------------------------------------------------------------------- the check
def test_an_older_marketplace_install_gets_the_update_command(fake):
    info = check()
    assert (info["running"], info["latest"], info["kind"]) == ("1.24.0", "1.26.0", "marketplace")
    assert info["command"] == f"claude plugin update {ID} --scope project"
    assert [v for v, _ in info["changes"]] == ["1.26.0", "1.25.0"]  # newer than the running one only
    text = U.report(info)
    assert "1.24.0 -> 1.26.0" in text and info["command"] in text and "/reload-plugins" in text
    assert "ledger: decide --member (fixes #112)" in text and "accounts reach the model" not in text
    assert "/work/old" not in text  # another project's install is not this session's


def test_up_to_date(fake):
    fake["installs"] = [_install("1.26.0", ROOT_NEW)]
    info = check(running="1.26.0", root=Path(ROOT_NEW))
    assert info["command"] is None and "up to date" in U.report(info)


def test_after_the_update_it_waits_for_reload_plugins(fake):
    """`claude plugin update` installed 1.26.0 for this project; this session still runs 1.24.0 until the reload."""
    fake["installs"] = [_install("1.26.0", ROOT_NEW)]
    info = check()
    assert info["kind"] == "marketplace" and info["command"] is None and info["pending"]["version"] == "1.26.0"
    text = U.report(info)
    assert "1.26.0 is installed" in text and "type /reload-plugins" in text
    assert f'PYTHONPATH="{ROOT_NEW}" python -m kgkit' in text  # the path to run kgkit from after the reload


def test_the_command_says_where_to_run_it_for_a_project_install(fake):
    text = U.report(check(cwd=Path("/elsewhere")))
    assert "from /work/comp" in text


def test_a_development_checkout_is_updated_with_git(fake, tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    root = repo / "plugins" / "kaggle-grandmaster"
    (root / ".claude-plugin").mkdir(parents=True)
    info = check(root=root)
    assert info["kind"] == "dev" and info["command"] is None
    assert "git pull" in U.report(info) and "claude plugin update" not in U.report(info)


def test_github_unreachable_still_gives_the_command(fake):
    fake["releases"] = None
    info = check()
    assert info["latest"] is None and info["command"] == f"claude plugin update {ID} --scope project"
    assert "could not read the latest release" in U.report(info)


def test_no_claude_cli(fake):
    fake["installs"] = None
    text = U.report(check())
    assert "claude CLI" in text and "/plugin" in text


def test_older_vendored_copies_in_the_workspace_are_listed(fake, tmp_path):
    ws = tmp_path / "ws"
    for rel, v in (("src/kgkit", "1.20.0"), ("third_party/kgkit", "1.26.0")):
        (ws / rel).mkdir(parents=True)
        (ws / rel / "__init__.py").write_text(f'"""kgkit."""\n__version__ = "{v}"\n')
    fake["installs"] = [_install("1.26.0", ROOT_NEW)]
    info = check(workspace=ws)
    assert [(p.relative_to(ws).as_posix(), v) for p, v in info["vendored"]] == [("src/kgkit", "1.20.0")]
    assert f'PYTHONPATH="{ROOT_NEW}" python -m kgkit vendor src' in U.report(info)


def test_cli_update_check_prints_the_report(fake, monkeypatch, capsys, tmp_path):
    from kgkit import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(U, "PLUGIN_ROOT", Path(ROOT_OLD))
    monkeypatch.setattr(U, "__version__", "1.24.0")
    cli.main(["update-check"])
    out = capsys.readouterr().out
    assert "1.24.0 -> 1.26.0" in out and f"claude plugin update {ID}" in out


def test_the_request_names_no_user(monkeypatch):
    """The only outside request: GitHub's public releases of the plugin, with a generic client name."""
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(RELEASES).encode()

    def urlopen(req, timeout):
        seen.update(url=req.full_url, headers=dict(req.header_items()), timeout=timeout)
        return Resp()
    monkeypatch.setattr(U.urllib.request, "urlopen", urlopen)
    assert U._releases_json()[0]["tag_name"] == "kaggle-grandmaster--v1.26.0"
    assert seen["url"] == f"https://api.github.com/repos/{U.REPO}/releases?per_page=30"
    assert seen["headers"]["User-agent"] == "kgkit-update-check" and seen["timeout"] <= 30


def test_plugin_list_reads_the_claude_cli(monkeypatch):
    monkeypatch.setattr(U.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(U.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(
        argv, 0, json.dumps([_install()]), ""))
    assert U._plugin_list()[0]["id"] == ID
    monkeypatch.setattr(U.shutil, "which", lambda name: None)
    assert U._plugin_list() is None


# ----------------------------------------------------------------------------- the slash command
PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "kaggle-grandmaster"


def test_kg_update_is_a_slash_command_that_asks_before_it_updates():
    text = (PLUGIN / "skills" / "kg-update" / "SKILL.md").read_text(encoding="utf-8")
    front = text.split("---")[1]
    assert "disable-model-invocation: true" in front
    rules = re.findall(r"^  - '?(.+?)'?$", front, re.M)
    assert [r for r in rules if "claude plugin" in r] == []  # the update itself is approved by the user each time
    assert "Bash(kgkit update-check *)" in rules
    body = text.split("---", 2)[2]
    assert "update-check" in body and "claude plugin update" in body and "/reload-plugins" in body


def test_the_router_suggests_kg_update(plugin_root, tmp_path_factory):
    from tests.test_hooks import hook

    from kgkit import state as S

    plain = tmp_path_factory.mktemp("plain_update")
    ws = tmp_path_factory.mktemp("ws_update")
    S.CompetitionState(slug="demo", metric="auc").save(ws)

    def ask(prompt, cwd=plain):
        out = hook(plugin_root, "prompt_router.py", {"hook_event_name": "UserPromptSubmit", "prompt": prompt,
                                                     "cwd": str(cwd), "session_id": uuid.uuid4().hex})
        return (out or {}).get("additionalContext") or ""

    for prompt in ("update the kaggle-grandmaster plugin to the latest version", "upgrade kaggle grandmaster"):
        assert "`/kg-update`" in ask(prompt), prompt
    # "this plugin" means kaggle-grandmaster in a competition workspace, not outside one
    assert "`/kg-update`" in ask("is there a newer version of this plugin?", ws)
    assert "`/kg-update`" not in ask("is there a newer version of this plugin?")
    for prompt in ("update the readme", "upgrade numpy to the latest version"):
        assert "`/kg-update`" not in ask(prompt, ws), prompt
