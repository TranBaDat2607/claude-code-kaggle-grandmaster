"""The kaggle-monitor mod (plugins/kaggle-grandmaster/hooks/monitor): a TypeScript hooks module Claude Code loads beside
the Python hooks. Its behaviour is tested by its own `*.test.tsx` under `claude plugin test`; these check the wiring
the repo can break without touching it."""

import importlib.util
import json
import re
import shutil
import subprocess

import pytest

KGMON_KEYS = {"v", "t", "cores", "cpu", "ram", "disk", "tmp", "gpus", "doing"}


def _mod(plugin_root):
    return plugin_root / "hooks" / "monitor"


def test_hooks_json_names_the_module_beside_the_python_hooks(plugin_root):
    hooks = json.loads((plugin_root / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    assert hooks["modules"] == ["./monitor/register.tsx"]
    assert (plugin_root / "hooks" / "monitor" / "register.tsx").is_file()
    assert "SessionStart" in hooks["hooks"] and "PreToolUse" in hooks["hooks"]  # the Python hooks stay
    manifest = json.loads((plugin_root / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert (plugin_root / manifest["types"]).is_file()


def test_state_and_prompt_sections_are_the_plugins_own(plugin_root):
    """A mod writes only $.state under its own plugin's name, and names its prompt sections <plugin>:<name>."""
    name = json.loads((plugin_root / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))["name"]
    src = "\n".join(p.read_text(encoding="utf-8") for p in _mod(plugin_root).glob("*.ts*"))
    owners = set(re.findall(r"plugin: '([^']+)'", src))
    assert owners == {name}, owners
    assert re.findall(r"id: '([^':]+):", src) == [name]
    contract = (plugin_root / "types" / "index.d.ts").read_text(encoding="utf-8")
    assert f"'{name}': {{" in contract


def test_the_pane_reads_every_field_the_runner_prints(plugin_root, monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("kg_runner_probe", plugin_root / "kgkit" / "kaggle_gpu_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "OUT", tmp_path)
    monkeypatch.setattr(runner, "RUN", tmp_path / "run")
    monkeypatch.setattr(runner, "_gpus", lambda: [[0, 50, 1000, 15360, "Tesla T4"]])
    runner.doing(0, "exp/f0 on gpu0")
    first, prev = runner.snapshot(None)
    snap, _ = runner.snapshot((prev[0] - 1, prev[1]))
    assert set(first) | set(snap) <= KGMON_KEYS, set(snap) - KGMON_KEYS
    assert {"v", "t", "cores", "disk", "tmp", "gpus", "doing"} <= set(snap)
    parser = (_mod(plugin_root) / "kaggle.ts").read_text(encoding="utf-8")
    body = parser[parser.index("export function parseKgmon"):]
    assert {k for k in KGMON_KEYS if f"j.{k}" in body} == KGMON_KEYS


@pytest.mark.skipif(shutil.which("claude") is None, reason="needs the Claude Code CLI")
def test_mod_tests_pass_under_claude_plugin_test(plugin_root):
    r = subprocess.run(["claude", "plugin", "test", str(plugin_root)], capture_output=True, text=True, timeout=600,
                       encoding="utf-8", errors="replace")
    assert r.returncode == 0 and " 0 fail" in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]
