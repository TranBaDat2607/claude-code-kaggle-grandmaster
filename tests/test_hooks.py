"""Drive the hooks the way Claude Code does: JSON on stdin through hooks/run.sh."""

import json
import shutil
import subprocess
import sys
import uuid
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
    assert "No accepted baseline yet" in ctx
    led.decide("1", "baseline")
    from kgkit.backlog import Backlog

    Backlog(ws).add("pseudo-label the test set", gain=3, prob=0.5, cost=2)
    ctx = hook(plugin_root, "session_start.py", {"hook_event_name": "SessionStart", "cwd": str(ws)})["additionalContext"]
    assert "Accepted baseline" in ctx and "0001_lgbm" in ctx and "Backlog top: #1 pseudo-label" in ctx


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


def _post(cmd, stdout, cwd):
    return {"hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_input": {"command": cmd},
            "tool_response": {"stdout": stdout}, "cwd": str(cwd)}


def _sub_log(ws):
    return [json.loads(x) for x in (ws / ".kaggle-gm" / "submissions.jsonl").read_text().splitlines()]


def test_post_code_submit_success_and_slow_poll(plugin_root, ws):
    cmd = 'kaggle competitions submit demo -k me/infer -v 3 -f submission.csv -m "0002_cnn | CV 0.8"'
    ctx = hook(plugin_root, "post_tool.py", _post(cmd, "4 submissions remaining today.", ws))["additionalContext"]
    assert _sub_log(ws)[-1]["status"] == "submitted"
    assert "reruns the notebook" in ctx and "1-9h" in ctx and "1-2 minutes" not in ctx
    runs = ws / ".kaggle-gm" / "gpu_runs.jsonl"
    runs.write_text(json.dumps({"kernel": "me/infer", "version": 3, "pushed_at": "2026-01-01T00:00:00Z",
                                "finished_at": "2026-01-01T02:30:00Z", "status": "complete"}) + "\n")
    ctx = hook(plugin_root, "post_tool.py", _post(cmd, "3 submissions remaining today.", ws))["additionalContext"]
    assert "last run took 2.5h" in ctx


def test_post_code_submit_400_explained(plugin_root, ws):
    cmd = "kaggle competitions submit demo -k me/infer -v 1 -f submission.csv -m 0003_x"
    out = ("400 Client Error: Bad Request for url: "
           "https://api.kaggle.com/v1/competitions.CompetitionApiService/CreateCodeSubmission")
    ctx = hook(plugin_root, "post_tool.py", _post(cmd, out, ws))["additionalContext"]
    assert _sub_log(ws)[-1]["status"] == "failed"
    assert "e.response.text" in ctx and "non-versioned" in ctx


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


def _gpu_kernel(ws, quota_h, gpu=True):
    kdir = ws / "kernels" / "x-train"
    kdir.mkdir(parents=True)
    meta = {"id": "me/x-train", "code_file": "x.py", "enable_gpu": gpu}
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta))
    (ws / ".kaggle-gm" / "gpu_quota.json").write_text(json.dumps(
        {"used_h": 30 - quota_h, "remaining_h": quota_h, "total_h": 30, "refresh_at": "2099-01-03T00:00:00Z",
         "fetched_at": "2026-01-01T00:00:00Z"}))


def test_gpu_push_guard(plugin_root, ws):
    _gpu_kernel(ws, quota_h=2.0)
    out = hook(plugin_root, "pre_tool.py", bash("kaggle kernels push -p kernels/x-train -t 21600", ws))
    assert out["permissionDecision"] == "ask" and "6.00h" in out["permissionDecisionReason"]
    assert hook(plugin_root, "pre_tool.py", bash("kaggle kernels push -p kernels/x-train -t 3600", ws)) is None
    # CPU push (e.g. a preprocessing kernel) never needs GPU quota
    out = hook(plugin_root, "pre_tool.py", bash("kaggle kernels push -p kernels/x-train -t 21600 --accelerator none", ws))
    assert out is None


def test_gpu_push_guard_exhausted(plugin_root, ws):
    _gpu_kernel(ws, quota_h=0.0)
    out = hook(plugin_root, "pre_tool.py", bash("kaggle kernels push -p kernels\\x-train", ws))
    assert out["permissionDecision"] == "ask" and "exhausted" in out["permissionDecisionReason"]


def test_gpu_push_logged_and_briefed(plugin_root, ws):
    _gpu_kernel(ws, quota_h=10.0)
    post = {"hook_event_name": "PostToolUse", "tool_name": "Bash",
            "tool_input": {"command": "kaggle kernels push -p kernels/x-train -t 7200"},
            "tool_response": {"stdout": "Kernel version 4 successfully pushed.  Please check progress at https://x"},
            "cwd": str(ws)}
    out = hook(plugin_root, "post_tool.py", post)
    assert "kgkit gpu wait me/x-train" in out["additionalContext"]
    run = json.loads((ws / ".kaggle-gm" / "gpu_runs.jsonl").read_text().splitlines()[-1])
    assert run["version"] == 4 and run["hours"] == 2.0 and run["status"] == "pushed"
    ctx = hook(plugin_root, "session_start.py", {"hook_event_name": "SessionStart", "cwd": str(ws)})["additionalContext"]
    assert "Kaggle GPU quota" in ctx and "me/x-train" in ctx


def test_prompt_router_gpu(plugin_root, tmp_path_factory):
    plain = tmp_path_factory.mktemp("plain2")
    out = hook(plugin_root, "prompt_router.py", {"hook_event_name": "UserPromptSubmit", "cwd": str(plain),
               "prompt": "How do I train my model on Kaggle GPUs without wasting my weekly quota?"})
    assert "kaggle-grandmaster:kaggle-gpu" in out["additionalContext"]


def test_prompt_router_strategy_and_noise(plugin_root, tmp_path_factory):
    plain = tmp_path_factory.mktemp("plain3")

    def ask(prompt):
        return hook(plugin_root, "prompt_router.py",
                    {"hook_event_name": "UserPromptSubmit", "prompt": prompt, "cwd": str(plain)})["additionalContext"]

    assert "kaggle-grandmaster:competition-strategy" in ask(
        "Should I merge teams with a friend in this Kaggle competition? We'd swap OOF predictions first")
    assert "kaggle-grandmaster:competition-strategy" in ask("Which Kaggle competitions should I pick to get a solo gold?")
    assert "kaggle-grandmaster:grandmaster-playbook" in ask(
        "Kaggle: my new model is +0.002 AUC but the fold std is 0.006, is the gain real or noise?")
    assert "kaggle-grandmaster:tabular-mastery" in ask("Run a groupby feature search for this Kaggle playground")
    assert "kaggle-grandmaster:competition-recon" in ask("Reproduce the top public notebook of this Kaggle comp")


def test_prompt_router_harness_and_dedupe(plugin_root, tmp_path_factory, ws):
    plain = tmp_path_factory.mktemp("plain4")
    sid = uuid.uuid4().hex

    def ask(prompt, cwd=plain, session=sid):
        return hook(plugin_root, "prompt_router.py", {"hook_event_name": "UserPromptSubmit", "prompt": prompt,
                                                      "cwd": str(cwd), "session_id": session})

    # harness-injected text is never routed, even inside a workspace
    checkin = ("Goal check-in: «finish the kaggle grandmaster plugin run» is still active. Check "
               "`kaggle competitions submissions rsna` - Claude Code won't wake this session on its own.")
    assert ask(checkin) is None and ask(checkin, ws) is None
    assert ask("<task-notification>Kaggle kernel finished, OOF saved</task-notification>", ws) is None

    # each skill is suggested once per session
    blend = "How should I blend OOF predictions from 9 models for this Kaggle competition?"
    assert "kaggle-grandmaster:ensembling" in ask(blend)["additionalContext"]
    assert ask(blend) is None
    ctx = ask("Kaggle: how do I optimise the F1 threshold on my OOF predictions?")["additionalContext"]
    assert "kaggle-grandmaster:metric-optimization" in ctx and "kaggle-grandmaster:ensembling" not in ctx
    assert "kaggle-grandmaster:ensembling" in ask(blend, session=uuid.uuid4().hex)["additionalContext"]

    # plain "session" is Claude Code wording, not recsys
    fresh = uuid.uuid4().hex
    ctx = ask("Kaggle: keep training in this session and log the LB score", session=fresh)["additionalContext"]
    assert "recsys-and-ranking" not in ctx
    assert "kaggle-grandmaster:recsys-and-ranking" in ask(
        "Session-based recommendation for this Kaggle comp: how to build candidates?", session=fresh)["additionalContext"]


def test_prompt_router_task_domain(plugin_root, tmp_path):
    S.CompetitionState(slug="knee", task="cv").save(tmp_path)
    out = hook(plugin_root, "prompt_router.py", {
        "hook_event_name": "UserPromptSubmit", "cwd": str(tmp_path), "session_id": uuid.uuid4().hex,
        "prompt": "Ranking the MRI series by abnormality score: which ranking of slices should feed the CNN?"})
    ctx = out["additionalContext"]
    assert "kaggle-grandmaster:computer-vision" in ctx and "recsys-and-ranking" not in ctx


def _mid_competition_repo(root):
    k = root / "kernels" / "knee-infer"
    k.mkdir(parents=True)
    (k / "kernel-metadata.json").write_text(json.dumps(
        {"id": "me/knee-infer", "competition_sources": ["rsna-knee"], "enable_internet": False}))
    (root / "README.md").write_text("Solution for https://www.kaggle.com/competitions/rsna-knee/discussion\n")
    return root


def test_session_start_suggests_adopting_existing_repo(plugin_root, tmp_path):
    _mid_competition_repo(tmp_path)
    ctx = hook(plugin_root, "session_start.py", {"hook_event_name": "SessionStart", "cwd": str(tmp_path)})[
        "additionalContext"]
    assert "kg-adopt rsna-knee" in ctx and "kgkit adopt rsna-knee --code-competition" in ctx
    assert "1 kernel-metadata.json" in ctx and "/kg-start only for a new competition" in ctx
    assert not (tmp_path / ".kaggle-gm").exists()


def test_prompt_router_adopt(plugin_root, tmp_path):
    out = hook(plugin_root, "prompt_router.py", {
        "hook_event_name": "UserPromptSubmit", "cwd": str(tmp_path), "session_id": uuid.uuid4().hex,
        "prompt": "I'm two months into this Kaggle competition with an existing repo, how do I set up the plugin?"})
    assert "kaggle-grandmaster:kg-adopt" in out["additionalContext"]


def test_more_credential_leaks_denied(plugin_root, tmp_path):
    for cmd in ["cat ~/.kaggle/access_token", "Get-Content $HOME\\.kaggle\\credentials.json",
                "head -c 20 ~/.kaggle/access_token.txt", "kaggle auth print-access-token",
                "VERBOSE=1 kaggle competitions submissions demo", "export VERBOSE_OUTPUT=true && kaggle kernels status a/b",
                '$env:VERBOSE="1"; kaggle quota', "VERBOSE=1 python -c 'import kaggle; kaggle.api.competitions_list()'"]:
        out = hook(plugin_root, "pre_tool.py", bash(cmd, tmp_path))
        assert out and out["permissionDecision"] == "deny", cmd
    # VERBOSE scoped to another program, or a kgkit call (kgkit drops it from its Kaggle CLI calls)
    for cmd in ["VERBOSE=1 python train.py", "cat reports/credentials_audit.md", "kaggle auth login",
                "ls ~/.kaggle", "VERBOSE=1 python train.py && kaggle competitions list",
                "VERBOSE=yes python -m kgkit gpu push kernels/x"]:
        assert hook(plugin_root, "pre_tool.py", bash(cmd, tmp_path)) is None, cmd


def test_upload_with_credentials_denied(plugin_root, tmp_path):
    up = tmp_path / "kernels" / "w"
    (up / "sub").mkdir(parents=True)
    (up / "fold0.pt").write_bytes(b"x")
    (up / "sub" / ".env").write_text("KAGGLE_KEY=1")
    ok = ["kaggle datasets create -p kernels/w", "kaggle datasets version -p kernels/w -m v2",
          "kaggle datasets version -p kernels/w -m v2 --dir-mode zip --ignore-patterns '*.env'"]
    for cmd in ok:
        assert hook(plugin_root, "pre_tool.py", bash(cmd, tmp_path)) is None, cmd
    for cmd in ["kaggle datasets version -p kernels/w -m v2 -r zip",
                "kaggle d create --path=kernels/w --dir-mode=tar",
                "kaggle models instances versions create me/m/pytorch/v1 -p kernels/w -r zip"]:
        out = hook(plugin_root, "pre_tool.py", bash(cmd, tmp_path))
        assert out["permissionDecision"] == "deny" and "sub/.env" in out["permissionDecisionReason"], cmd
    (up / "kaggle.json").write_text("{}")
    out = hook(plugin_root, "pre_tool.py", bash("kaggle datasets create", up))  # -p defaults to the cwd
    assert out["permissionDecision"] == "deny" and "kaggle.json" in out["permissionDecisionReason"]


def test_resubmitting_the_same_file_asks(plugin_root, ws):
    cmd = 'kaggle competitions submit demo -f subs/good.csv -m "0001_lgbm | CV 0.8"'
    hook(plugin_root, "post_tool.py", _post(cmd, "Successfully submitted to Demo", ws))
    assert len(_sub_log(ws)[-1]["sha256"]) == 64
    out = hook(plugin_root, "pre_tool.py", bash(cmd.replace("0001_lgbm", "0002_again"), ws))
    assert out["permissionDecision"] == "ask" and "already submitted" in out["permissionDecisionReason"]
    (ws / "subs" / "other.csv").write_text("id,target\n1,0.2\n2,0.8\n3,0.4\n")
    assert hook(plugin_root, "pre_tool.py", bash(cmd.replace("good", "other"), ws)) is None


def test_kaggle_reported_budget_is_used(plugin_root, ws):
    S.CompetitionState(slug="demo", metric="auc", daily_submissions=5, deadline="2099-01-01").save(ws)
    cmd = 'kaggle competitions submit demo -f subs/good.csv -m "0001_lgbm | CV 0.8"'
    ctx = hook(plugin_root, "post_tool.py", _post(cmd, "100%|##| 30/30\n0 submissions remaining today.",
                                                  ws))["additionalContext"]
    assert "Kaggle reports 0 left today for the team" in ctx and _sub_log(ws)[-1]["remaining_today"] == 0
    (ws / "subs" / "other.csv").write_text("id,target\n1,0.2\n2,0.8\n3,0.4\n")
    out = hook(plugin_root, "pre_tool.py", bash(cmd.replace("good", "other"), ws))
    assert out["permissionDecision"] == "ask" and "0 submissions left today" in out["permissionDecisionReason"]


def test_exit_zero_failures_recognised(plugin_root, ws):
    cmd = 'kaggle competitions submit demo -f subs/good.csv -m "0001_x"'
    hook(plugin_root, "post_tool.py", _post(cmd, "Could not submit to competition", ws))
    assert _sub_log(ws)[-1]["status"] == "failed"
    ctx = hook(plugin_root, "post_tool.py", _post("kaggle datasets version -p kernels/w -m v3 && kaggle kernels push",
                                                  "Dataset version creation error: Invalid metadata", ws))
    assert "exited 0" in ctx["additionalContext"] and "Invalid metadata" in ctx["additionalContext"]
    ctx = hook(plugin_root, "post_tool.py", _post("kaggle kernels push -p kernels/x",
                                                  "Kernel push error: Maximum batch GPU session count of 2 reached.",
                                                  ws))
    assert "nothing was created or started" in ctx["additionalContext"] and "gpu status" in ctx["additionalContext"]
    assert hook(plugin_root, "post_tool.py", _post("kaggle datasets version -p w -m v", "Dataset version created", ws)) is None


def test_chained_upload_still_checks_the_submit(plugin_root, ws):
    (ws / "kernels" / "w").mkdir(parents=True)
    (ws / "kernels" / "w" / "fold0.pt").write_bytes(b"x")
    out = hook(plugin_root, "pre_tool.py",
               bash('kaggle datasets version -p kernels/w -m v3 && kaggle competitions submit demo -f subs/bad.csv -m x',
                    ws))
    assert out["permissionDecision"] == "deny" and "invalid submission" in out["permissionDecisionReason"]


def test_chained_submit_logged_after_failed_upload(plugin_root, ws):
    cmd = 'kaggle datasets version -p kernels/w -m v ; kaggle competitions submit demo -f subs/good.csv -m "0001_x"'
    ctx = hook(plugin_root, "post_tool.py", _post(cmd, "Dataset version creation error: Invalid metadata\n"
                                                       "Successfully submitted to Demo", ws))["additionalContext"]
    rec = _sub_log(ws)[-1]
    assert rec["status"] == "submitted" and rec["size"] == (ws / "subs" / "good.csv").stat().st_size
    assert "exited 0" in ctx and "submission logged" in ctx


def test_stale_remaining_count_is_not_reused(plugin_root, ws, monkeypatch):
    S.CompetitionState(slug="demo", metric="auc", daily_submissions=5, deadline="2099-01-01").save(ws)
    cmd = 'kaggle competitions submit demo -f subs/good.csv -m "0001_lgbm"'
    hook(plugin_root, "post_tool.py", _post(cmd, "0 submissions remaining today.", ws))
    hook(plugin_root, "post_tool.py", _post(cmd.replace("0001", "0002"), "", ws))  # no count printed
    monkeypatch.syspath_prepend(str(plugin_root / "hooks"))
    import _common as C

    assert C.server_remaining_today(ws) is None
    hook(plugin_root, "post_tool.py", _post(cmd, "Could not submit to competition", ws))  # failed: skipped
    assert C.server_remaining_today(ws) is None
