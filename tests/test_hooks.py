"""Drive the hooks the way Claude Code does: JSON on stdin through hooks/run.sh."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kgkit import state as S

SH = shutil.which("sh")
pytestmark = pytest.mark.skipif(SH is None, reason="needs a POSIX sh")


def hook(plugin_root: Path, script: str, payload: dict) -> dict | None:
    r = subprocess.run([SH, str(plugin_root / "hooks" / "run.sh"), script], input=json.dumps(payload),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    out = r.stdout.strip()
    return json.loads(out)["hookSpecificOutput"] if out else None


@pytest.fixture
def ws(tmp_path):
    S.CompetitionState(slug="demo", metric="auc", daily_submissions=2, deadline="2099-01-01").save(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sample_submission.csv").write_text("id,target\n1,0.5\n2,0.5\n3,0.5\n")
    (tmp_path / "subs").mkdir()
    (tmp_path / "subs" / "good.csv").write_text("id,target\n1,0.1\n2,0.9\n3,0.4\n")
    (tmp_path / "subs" / "bad.csv").write_text(",id,target\n0,1,0.1\n1,2,\n")
    return tmp_path


def bash(cmd, cwd):
    return {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": str(cwd)}


def test_session_start_outside_workspace(plugin_root, tmp_path):
    out = hook(plugin_root, "session_start.py", {"hook_event_name": "SessionStart", "cwd": str(tmp_path)})
    assert out["hookEventName"] == "SessionStart"
    assert "kgkit home" in out["additionalContext"] and "kg-start" in out["additionalContext"]


def test_session_start_inside_workspace(plugin_root, ws):
    from kgkit.experiment import Ledger

    led = Ledger(ws)
    led.log("lgbm", 0.80, notes="baseline")
    led.log("cat", 0.81)
    led.attach_lb("1", 0.79)
    out = hook(plugin_root, "session_start.py", {"hook_event_name": "SessionStart", "cwd": str(ws / "subs")})
    ctx = out["additionalContext"]
    assert "demo" in ctx and "best CV 0.81000" in ctx and "days left" in ctx and "folds.csv" in ctx


def test_secret_read_denied(plugin_root, tmp_path):
    for cmd in ["cat ~/.kaggle/kaggle.json", "Get-Content $HOME\\.kaggle\\kaggle.json", "echo $KAGGLE_KEY",
                "printenv KAGGLE_KEY"]:
        out = hook(plugin_root, "pre_tool.py", bash(cmd, tmp_path))
        assert out and out["permissionDecision"] == "deny", cmd


def test_invalid_submission_denied(plugin_root, ws):
    out = hook(plugin_root, "pre_tool.py", bash('kaggle competitions submit demo -f subs/bad.csv -m "0001_x"', ws))
    assert out["permissionDecision"] == "deny"
    assert "rows" in out["permissionDecisionReason"]


def test_valid_submission_passes_through(plugin_root, ws):
    cmd = 'kaggle competitions submit demo -f subs/good.csv -m "0001_x | CV 0.8"'
    assert hook(plugin_root, "pre_tool.py", bash(cmd, ws)) is None


def test_powershell_backslash_path(plugin_root, ws):
    cmd = 'kaggle competitions submit demo -f subs\\bad.csv -m "0001_x"; Write-Host ok'
    out = hook(plugin_root, "pre_tool.py", bash(cmd, ws))
    assert out["permissionDecision"] == "deny"


def test_unrelated_and_listing_commands_ignored(plugin_root, ws):
    assert hook(plugin_root, "pre_tool.py", bash("ls -la", ws)) is None
    assert hook(plugin_root, "pre_tool.py", bash("kaggle competitions list", ws)) is None


def test_post_logs_and_budget_ask(plugin_root, ws):
    cmd = 'kaggle competitions submit demo -f subs/good.csv -m "0001_lgbm | CV 0.8"'
    for _ in range(2):
        post = {"hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_input": {"command": cmd},
                "tool_response": {"stdout": "Successfully submitted to Demo"}, "cwd": str(ws)}
        out = hook(plugin_root, "post_tool.py", post)
        assert "ledger lb 0001_lgbm" in out["additionalContext"]
    lines = (ws / ".kaggle-gm" / "submissions.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["exp_id"] == "0001_lgbm"
    out = hook(plugin_root, "pre_tool.py", bash(cmd, ws))
    assert out["permissionDecision"] == "ask"


def test_parse_submit_variants(plugin_root):
    sys.path.insert(0, str(plugin_root / "hooks"))
    import _common as C

    p = C.parse_submit('cd x && kaggle competitions submit my-comp -f "subs/a b.csv" -m "0002_y | CV 1" && echo done')
    assert p["competition"] == "my-comp" and p["file"] == "subs/a b.csv" and p["message"].startswith("0002_y")
    p = C.parse_submit("kaggle c submit -k me/nb -v 3 -f submission.csv -m m comp")
    assert p["kernel"] == "me/nb" and p["version"] == "3" and p["competition"] == "comp"
    assert C.parse_submit("kaggle competitions submissions comp") is None
    p = C.parse_submit('kaggle competitions submit comp -f subs\\x.csv -m "0003_x"; Write-Host ok')
    assert p["file"] == "subs/x.csv" and p["message"] == "0003_x"


def test_prompt_router(plugin_root, tmp_path_factory, ws):
    plain = tmp_path_factory.mktemp("plain")
    def ask(prompt, cwd):
        return hook(plugin_root, "prompt_router.py",
                    {"hook_event_name": "UserPromptSubmit", "prompt": prompt, "cwd": str(cwd)})

    out = ask("My Kaggle CV is 0.95 but the public LB is 0.81, I used KFold on daily sales with lag features", plain)
    ctx = out["additionalContext"]
    assert "kaggle-grandmaster:cv-lb-debugging" in ctx and "kaggle-grandmaster:validation-strategy" in ctx
    out = ask("How should I blend OOF predictions from 9 models for this competition?", plain)
    assert "kaggle-grandmaster:ensembling" in out["additionalContext"]
    # generic prompt outside a workspace: silent
    assert ask("Refactor this React component to use hooks", plain) is None
    # inside a workspace even generic wording routes (falls back to the playbook)
    out = ask("what should I try next?", ws)
    assert "kaggle-grandmaster:grandmaster-playbook" in out["additionalContext"]
    # slash commands are left alone
    assert ask("/kaggle-grandmaster:kg-status", ws) is None
