"""The plugin as users get it: installed from the marketplace, not loaded with --plugin-dir.

Opt-in (``KGKIT_INSTALL_TEST=1``, and the Claude Code CLI on PATH), and run by the ``install`` CI job on Linux and
Windows. It adds this repo as a marketplace and installs kaggle-grandmaster at project scope into a fresh folder, with
``CLAUDE_CONFIG_DIR`` pointed at a temporary directory so the developer's own Claude config is never touched. Then it
checks what only an install can break: the copy in the plugin cache is complete and at plugin.json's version,
``bin/kgkit`` runs the cache's kgkit and finds the templates beside it, the SessionStart hook exports that copy as
``KGKIT_HOME`` (which a ``kgkit vendor`` copy reads its templates through), and the manifest, skills, agents and the
monitor mod validate and pass their tests from the cache. No Kaggle account and no Claude login are needed.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
PLUGIN = REPO / "plugins" / "kaggle-grandmaster"
SH = shutil.which("sh")

pytestmark = pytest.mark.skipif(
    not os.environ.get("KGKIT_INSTALL_TEST") or shutil.which("claude") is None or SH is None,
    reason="opt-in: KGKIT_INSTALL_TEST=1 with the Claude Code CLI and sh on PATH")


def _run(cmd, cwd, env, timeout=300):
    r = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    assert r.returncode == 0, f"{cmd}: exit {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    return r


def _clean_env(**extra) -> dict:
    """This environment without anything that would point a command at the repo's kgkit instead of the cache's."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "KGKIT_HOME", "CLAUDE_PLUGIN_ROOT")}
    env.update(extra)
    return env


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    base = tmp_path_factory.mktemp("install")
    cfg, proj = base / "claude-config", base / "project"
    cfg.mkdir()
    proj.mkdir()
    env = _clean_env(CLAUDE_CONFIG_DIR=str(cfg))
    market = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))["name"]
    version = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))["version"]
    _run(["claude", "plugin", "marketplace", "add", str(REPO), "--scope", "project", "--json"], proj, env)
    _run(["claude", "plugin", "install", "kaggle-grandmaster", "--scope", "project", "-y", "--json"], proj, env)
    cache = cfg / "plugins" / "cache" / market / "kaggle-grandmaster" / version
    return SimpleNamespace(cfg=cfg, proj=proj, env=env, cache=cache, version=version, market=market, base=base)


def test_install_puts_a_complete_copy_in_the_cache(installed):
    c = installed.cache
    assert c.is_dir(), sorted(p.as_posix() for p in installed.cfg.rglob("*"))[:40]
    assert c.resolve() != PLUGIN.resolve()
    for rel in (".claude-plugin/plugin.json", "bin/kgkit", "hooks/hooks.json", "hooks/run.sh",
                "hooks/monitor/register.tsx", "kgkit/__main__.py", "templates/CLAUDE.competition.md",
                "skills/kg-start/SKILL.md", "agents/competition-analyst.md"):
        assert (c / rel).is_file(), rel
    init = (c / "kgkit" / "__init__.py").read_text(encoding="utf-8")
    assert f'"{installed.version}"' in init or f"'{installed.version}'" in init
    settings = json.loads((installed.proj / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert settings["enabledPlugins"] == {f"kaggle-grandmaster@{installed.market}": True}
    listed = _run(["claude", "plugin", "list"], installed.proj, installed.env).stdout
    assert "kaggle-grandmaster" in listed


def test_bin_kgkit_runs_the_installed_kgkit(installed):
    kgkit = installed.cache / "bin" / "kgkit"
    if os.name != "nt":
        assert os.access(kgkit, os.X_OK), "bin/kgkit lost its executable bit in the install"
    ws = installed.base / "ws-bin"
    ws.mkdir()
    env = _clean_env(KGKIT_PYTHON=sys.executable)
    r = _run([SH, str(kgkit), "init", "demo", "--metric", "auc"], ws, env)
    assert (ws / ".kaggle-gm" / "competition.json").is_file(), r.stdout
    # the template beside the cache's kgkit, found with KGKIT_HOME unset
    assert "demo" in (ws / "CLAUDE.md").read_text(encoding="utf-8")


def test_session_start_exports_the_installed_copy_as_kgkit_home(installed):
    env_file = installed.base / "claude-env.sh"
    env = _clean_env(CLAUDE_PLUGIN_ROOT=str(installed.cache), CLAUDE_ENV_FILE=str(env_file))
    payload = json.dumps({"hook_event_name": "SessionStart", "cwd": str(installed.proj), "source": "startup"})
    r = subprocess.run([SH, str(installed.cache / "hooks" / "run.sh"), "session_start.py"], input=payload,
                       capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    exported = env_file.read_text(encoding="utf-8")
    assert f'KGKIT_HOME="{installed.cache.resolve().as_posix()}"' in exported or \
        f'KGKIT_HOME="{installed.cache.as_posix()}"' in exported, exported


def test_a_vendored_copy_finds_templates_through_kgkit_home(installed):
    ws = installed.base / "ws-vendor"
    ws.mkdir()
    _run([SH, str(installed.cache / "bin" / "kgkit"), "vendor", "src"], ws, _clean_env(KGKIT_PYTHON=sys.executable))
    assert (ws / "src" / "kgkit" / "__init__.py").is_file() and not (ws / "src" / "templates").exists()
    env = _clean_env(PYTHONPATH=str(ws / "src"), KGKIT_HOME=installed.cache.as_posix())
    _run([sys.executable, "-m", "kgkit", "init", "demo", "--metric", "auc"], ws, env)
    assert "demo" in (ws / "CLAUDE.md").read_text(encoding="utf-8")


def _plugin_test(path: Path, cwd: Path, env: dict) -> list[str]:
    """The per-test lines of `claude plugin test` ("(pass) name", "(fail) name"), timings dropped."""
    r = subprocess.run(["claude", "plugin", "test", str(path)], cwd=str(cwd), env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=600)
    return sorted(line.split(" [")[0].strip() for line in r.stdout.splitlines() if line.startswith(("(pass)", "(fail)")))


def test_the_installed_copy_validates_and_tests_like_the_source(installed):
    """The mod's own tests (tests/test_monitor_mod.py runs them on the source) give the same results from the
    cache: an install that drops or changes a file shows as a difference."""
    _run(["claude", "plugin", "validate", str(installed.cache)], installed.proj, installed.env)
    cached = _plugin_test(installed.cache, installed.proj, installed.env)
    assert cached and cached == _plugin_test(PLUGIN, installed.proj, installed.env)
