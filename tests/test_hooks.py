"""Drive the hooks the way Claude Code does: JSON on stdin through hooks/run.sh."""

import json
import os
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


def test_session_start_warns_about_competition_data_in_git(plugin_root, ws):
    """The brief reads the cached `kgkit audit-repo` result (UAT #69): no scan, no network call at session start."""
    (ws / ".kaggle-gm" / "audit.json").write_text(json.dumps(
        {"at": "2026-11-02T07:00:00Z", "public": True, "leaks": ["data/gold_annotated.csv"], "checks": [],
         "unignored": ["data/new.csv"]}))
    ctx = hook(plugin_root, "session_start.py", {"hook_event_name": "SessionStart", "cwd": str(ws)})["additionalContext"]
    assert "WARNING: competition data in git (the repo is PUBLIC; audit of 2026-11-02): data/gold_annotated.csv" in ctx
    assert "`git add -A` would commit (audit of 2026-11-02): data/new.csv" in ctx
    (ws / ".kaggle-gm" / "audit.json").write_text("{broken")
    ctx = hook(plugin_root, "session_start.py", {"hook_event_name": "SessionStart", "cwd": str(ws)})["additionalContext"]
    assert "demo" in ctx and "WARNING: competition data" not in ctx  # a bad cache never breaks the brief


def test_session_start_flags_a_vendored_kgkit_of_another_version(plugin_root, ws):
    import kgkit

    init = ws / "src" / "kgkit" / "__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text('"""kgkit"""\n\n__version__ = "1.6.2"\n')
    ctx = hook(plugin_root, "session_start.py", {"hook_event_name": "SessionStart", "cwd": str(ws)})["additionalContext"]
    home = ctx.split("kgkit home: ", 1)[1].split(" — ", 1)[0]
    assert f"Vendored kgkit not at the plugin's version {kgkit.__version__}: src/kgkit is 1.6.2." in ctx
    assert f'`PYTHONPATH="{home}" python -m kgkit vendor src`' in ctx
    init.write_text(f'__version__ = "{kgkit.__version__}"\n')
    ctx = hook(plugin_root, "session_start.py", {"hook_event_name": "SessionStart", "cwd": str(ws)})["additionalContext"]
    assert "Vendored kgkit" not in ctx


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


def test_prompt_router_after_the_deadline(plugin_root, tmp_path_factory, ws):
    """#70: "show my work" points to /kg-showcase, a slash command Claude cannot load; "the competition ended"
    loads kg-postmortem."""
    plain = tmp_path_factory.mktemp("plain5")
    sid = uuid.uuid4().hex

    def ask(prompt, cwd=ws, session=sid):
        return hook(plugin_root, "prompt_router.py", {"hook_event_name": "UserPromptSubmit", "prompt": prompt,
                                                      "cwd": str(cwd), "session_id": session})

    ctx = ask("how can I update my project to show my work")["additionalContext"]  # the issue's prompt, verbatim
    assert "suggest the user run `/kg-showcase`" in ctx and "Make nothing public from here" in ctx
    assert "grandmaster-playbook" not in ctx and "Skill tool" not in ctx  # a suggestion, not a skill to load
    assert ask("how can I update my project to show my work") is None  # once per session
    ctx = ask("The competition ended yesterday: write the post-mortem", session=uuid.uuid4().hex)["additionalContext"]
    assert "kaggle-grandmaster:kg-postmortem" in ctx and "kg-showcase" not in ctx
    ctx = ask("The Kaggle competition is over, help me write my write-up", cwd=plain,
              session=uuid.uuid4().hex)["additionalContext"]
    assert "kaggle-grandmaster:kg-postmortem" in ctx and "`/kg-showcase`" in ctx
    assert ask("update the README of my React project to show the results", cwd=plain) is None


def test_prompt_router_suggests_the_monitor_pane(plugin_root, tmp_path_factory):
    """Quota, running-kernel and account questions point to /kaggle-monitor, the pane the plugin's mod opens."""
    plain = tmp_path_factory.mktemp("plain6")

    def ask(prompt):
        return hook(plugin_root, "prompt_router.py", {"hook_event_name": "UserPromptSubmit", "prompt": prompt,
                                                      "cwd": str(plain), "session_id": uuid.uuid4().hex})

    for prompt in ("how many GPU hours do I have left on Kaggle?", "is my kaggle kernel still running",
                   "switch kaggle account to acc2", "what is my Kaggle notebook's GPU usage right now"):
        assert "`/kaggle-monitor`" in ask(prompt)["additionalContext"], prompt
    assert ask("monitor my React app's memory usage") is None


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
    hook(plugin_root, "post_tool.py", _post(cmd.replace("0001", "0002"), "Successfully submitted to Demo",
                                            ws))  # accepted, no count printed (-q)
    monkeypatch.syspath_prepend(str(plugin_root / "hooks"))
    import _common as C

    assert C.server_remaining_today(ws) is None
    hook(plugin_root, "post_tool.py", _post(cmd, "Could not submit to competition", ws))  # failed: skipped
    assert C.server_remaining_today(ws) is None


# ----------------------------------------------------------------------------- #63 submits made by a script
WRAPPER = '#!/bin/bash\nset -e\n# validate, submit, poll\nkaggle competitions submit demo -f "$1" -m "$2"\n'


def test_post_logs_a_submit_made_by_a_wrapper_script(plugin_root, ws):
    (ws / "submit.sh").write_text(WRAPPER)
    cmd = 'bash submit.sh subs/good.csv "0001_lgbm | CV 0.8"'
    ctx = hook(plugin_root, "post_tool.py", _post(cmd, "4 submissions remaining today.", ws))["additionalContext"]
    (rec,) = _sub_log(ws)
    assert rec["file"] == "subs/good.csv" and rec["sha256"] and rec["remaining_today"] == 4
    assert rec["exp_id"] == "0001_lgbm" and rec["status"] == "submitted"
    assert "made by submit.sh" in ctx and "ledger lb 0001_lgbm" in ctx


def test_post_script_that_stopped_before_submitting_logs_nothing(plugin_root, ws):
    (ws / "submit.sh").write_text(WRAPPER)
    out = hook(plugin_root, "post_tool.py", _post('bash submit.sh subs/bad.csv "0001_x"', "validation failed", ws))
    assert out is None and not (ws / ".kaggle-gm" / "submissions.jsonl").exists()


def test_post_script_variables_and_direct_run(plugin_root, ws):
    (ws / "go.sh").write_text('FILE=subs/good.csv\nkaggle competitions submit demo -f "$FILE" -m "$1"\n')
    (ws / "go.sh").chmod(0o755)
    hook(plugin_root, "post_tool.py", _post("./go.sh 0002_cnn", "3 submissions remaining today.", ws))
    assert _sub_log(ws)[-1]["file"] == "subs/good.csv" and _sub_log(ws)[-1]["exp_id"] == "0002_cnn"


def test_post_notes_a_reply_it_cannot_attribute(plugin_root, ws):
    ctx = hook(plugin_root, "post_tool.py", _post("make submit", "4 submissions remaining today.", ws))["additionalContext"]
    assert "not in the local log" in ctx and "kgkit status" in ctx
    assert not (ws / ".kaggle-gm" / "submissions.jsonl").exists()
    assert hook(plugin_root, "post_tool.py", _post("make test", "all passed", ws)) is None


def test_pre_asks_on_the_budget_for_a_script_submit_only(plugin_root, ws):
    (ws / "submit.sh").write_text(WRAPPER)
    cmd = 'bash submit.sh subs/bad.csv "0001_x"'
    assert hook(plugin_root, "pre_tool.py", bash(cmd, ws)) is None  # the file is not judged: the script may fix it
    for _ in range(2):
        hook(plugin_root, "post_tool.py", _post(cmd, "1 submissions remaining today.", ws))
    out = hook(plugin_root, "pre_tool.py", bash(cmd, ws))
    assert out["permissionDecision"] == "ask" and "2/2 submissions already logged" in out["permissionDecisionReason"]


# ----------------------------------------------------------------------------- compute efficiency (#91)
CE = "kaggle-grandmaster:compute-efficiency"


def test_prompt_router_compute_in_plain_words(plugin_root, tmp_path_factory, tmp_path):
    plain = tmp_path_factory.mktemp("plain6")

    def ask(prompt, cwd=plain):
        out = hook(plugin_root, "prompt_router.py", {"hook_event_name": "UserPromptSubmit", "prompt": prompt,
                                                     "cwd": str(cwd), "session_id": uuid.uuid4().hex})
        return out["additionalContext"] if out else ""

    for prompt in ["My training on Kaggle is slow and the GPU utilisation is only 30%",
                   "My Kaggle code competition notebook times out: inference takes 11 hours",
                   "Fine-tuning DeBERTa for this Kaggle competition runs out of memory on the T4",
                   "Kaggle: one epoch takes too long, how do I make it faster?",
                   "How do I use both T4 GPUs on Kaggle?",
                   "Kaggle: the submission notebook exceeded the runtime",
                   "In a Kaggle code competition with a 9-hour limit my LLM needs about 20 hours"]:
        assert CE in ask(prompt), prompt
    # building a model loads the defaults as well, not only a speed question
    assert CE in ask("Train a LightGBM baseline for this Kaggle playground")
    S.CompetitionState(slug="knee", task="cv").save(tmp_path)
    ctx = ask("write the training script for the baseline model", tmp_path)
    assert CE in ctx and "kaggle-grandmaster:computer-vision" in ctx
    # not Kaggle work and not in a workspace: silent
    assert ask("my laptop is slow") == ""
    assert ask("write a training script for my company's churn model") == ""
    assert CE not in ask("Should I join this Kaggle competition, it has 2 million rows of data?")


SLOW_TRAIN = """import timm, torch
from torch.utils.data import DataLoader
model = timm.create_model("resnet18")
dl = DataLoader(ds, batch_size=32, shuffle=True)
for x, y in dl:
    loss = model(x.cuda()).sum()
    loss.backward()
"""


def _write(plugin_root, ws, rel, text, sid, cwd=None, tool="Write"):
    path = ws / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    ti = {"file_path": str(path), "content": text} if tool == "Write" else {"file_path": str(path),
                                                                           "old_string": "a", "new_string": "b"}
    return hook(plugin_root, "post_write.py", {"hook_event_name": "PostToolUse", "tool_name": tool, "tool_input": ti,
                                               "cwd": str(cwd or ws), "session_id": sid})


def test_post_write_gives_the_compute_defaults_once(plugin_root, ws):
    sid = uuid.uuid4().hex
    out = _write(plugin_root, ws, "src/train.py", SLOW_TRAIN, sid)
    ctx = out["additionalContext"]
    ref = (plugin_root / "skills" / "compute-efficiency" / "references" / "vision.md").as_posix()
    assert out["hookEventName"] == "PostToolUse" and ref in ctx and Path(ref).is_file()
    assert "src/train.py trains a vision model" in ctx and "channels_last" in ctx
    assert "Slow patterns in src/train.py" in ctx and "line 4: DataLoader without workers" in ctx
    assert "fp32" in ctx and "cudnn.benchmark" in ctx
    # the same file again in this session: nothing new to say
    assert _write(plugin_root, ws, "src/train.py", SLOW_TRAIN, sid, tool="Edit") is None
    # a new slow pattern is reported, without repeating the defaults
    ctx = _write(plugin_root, ws, "src/train.py", SLOW_TRAIN + "df = df.apply(f, axis=1)\n", sid)["additionalContext"]
    assert "row-wise Python loop" in ctx and "Apply the compute defaults" not in ctx and "DataLoader" not in ctx
    # another session hears it all again
    assert "Apply the compute defaults" in _write(plugin_root, ws, "src/train.py", SLOW_TRAIN,
                                                  uuid.uuid4().hex)["additionalContext"]


def test_post_write_domains_and_silence(plugin_root, ws, tmp_path_factory):
    sid = uuid.uuid4().hex
    # a GBDT script: the tabular defaults; a clean script has no slow-pattern line
    gbdt = "import xgboost as xgb\nm = xgb.XGBRegressor(tree_method='hist', device='cuda')\nm.fit(X, y)\n"
    ctx = _write(plugin_root, ws, "src/train_gbdt2.py", gbdt, sid)["additionalContext"]
    assert "trains a tabular model" in ctx and "tabular.md" in ctx and "Slow patterns" not in ctx
    # domain from the workspace's task when the imports do not say (relative path, cwd in a subfolder)
    S.CompetitionState(slug="demo", metric="auc", task="cv").save(ws)
    # (relative path from a subfolder; the file is not on disk, so the written content is checked)
    torch_only = "import torch\nnet = torch.nn.Linear(4, 1)\nloss = net(x).sum()\nloss.backward()\n"
    (ws / "notebooks").mkdir()
    out = hook(plugin_root, "post_write.py", {
        "hook_event_name": "PostToolUse", "tool_name": "Write", "cwd": str(ws / "notebooks"), "session_id": sid,
        "tool_input": {"file_path": "../src/net.py", "content": torch_only}})
    ctx = out["additionalContext"]
    assert "src/net.py trains a vision model" in ctx and "vision.md" in ctx and "line 4: the training loop runs in fp32" in ctx
    # silent: not model code, not Python, outside a workspace
    assert _write(plugin_root, ws, "src/utils.py", "def f(x):\n    return x + 1\n", sid) is None
    assert _write(plugin_root, ws, "reports/notes.md", SLOW_TRAIN, sid) is None
    outside = tmp_path_factory.mktemp("not_a_workspace")
    assert _write(plugin_root, outside, "train.py", SLOW_TRAIN, sid) is None


def _kernel(ws, name, code, code_file="k.py", **meta):
    kdir = ws / "kernels" / name
    kdir.mkdir(parents=True)
    (kdir / "kernel-metadata.json").write_text(json.dumps({"id": f"me/{name}", "code_file": code_file, **meta}))
    (kdir / code_file).write_bytes(code.encode("utf-8") if isinstance(code, str) else code)
    return kdir / code_file


def _kgmon_lines(out):
    return [l for l in out.splitlines() if l.startswith("[kgmon] ")]


def test_push_adds_the_monitor_to_a_script(plugin_root, ws, tmp_path_factory):
    """Every kernel Claude pushes reports its hardware to /kaggle-monitor: the cell goes after the shebang, docstring
    and __future__ import, once, and keeps working when the script rebinds `time` later."""
    code = '#!/usr/bin/env python3\n"""Train."""\nfrom __future__ import annotations\nfrom time import time\nprint("ran", time())\n'
    f = _kernel(ws, "s", code)
    out = hook(plugin_root, "pre_tool.py", bash("KAGGLE_API_TOKEN=$HOME/.kaggle/t kaggle kernels push -p kernels/s", ws))
    assert "permissionDecision" not in out and "added the hardware monitor" in out["additionalContext"]
    lines = f.read_text(encoding="utf-8").split("\n")
    assert lines[:3] == code.split("\n")[:3] and lines[3].startswith("# >>> kaggle-grandmaster hardware monitor")
    assert lines[-3:] == ["from time import time", 'print("ran", time())', ""]
    work = tmp_path_factory.mktemp("work")
    r = subprocess.run([sys.executable, str(f)], env={**os.environ, "KAGGLE_WORKING": str(work)},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert len(_kgmon_lines(r.stdout)) == 1 and "ran " in r.stdout
    before = f.read_bytes()
    assert hook(plugin_root, "pre_tool.py", bash("kaggle kernels push -p kernels/s", ws)) is None  # once only
    assert f.read_bytes() == before


def test_push_adds_the_monitor_as_the_first_notebook_cell(plugin_root, ws):
    nb = {"metadata": {}, "nbformat": 4, "nbformat_minor": 5,
          "cells": [{"cell_type": "code", "id": "c1", "metadata": {}, "outputs": [], "execution_count": None,
                     "source": ["print(1)"]}]}
    f = _kernel(ws, "n", json.dumps(nb, separators=(",", ":")), code_file="n.ipynb", kernel_type="notebook")
    g = _kernel(ws, "m", json.dumps(nb, indent=1) + "\n", code_file="m.ipynb")
    out = hook(plugin_root, "pre_tool.py", bash("cd kernels && kaggle kernels push -p n && kaggle kernels push -p m", ws))
    assert out["additionalContext"].count("added the hardware monitor") == 2
    got = json.loads(f.read_text(encoding="utf-8"))
    assert [c.get("id") for c in got["cells"]][1:] == ["c1"] and got["cells"][0]["id"].startswith("kgmon-")
    assert "start_kgmon()" in "".join(got["cells"][0]["source"])
    assert "\n" not in f.read_text(encoding="utf-8")  # Kaggle's compact JSON stays compact
    assert g.read_text(encoding="utf-8").startswith('{\n "metadata"') and g.read_text(encoding="utf-8").endswith("}\n")


def test_push_monitor_left_out(plugin_root, ws):
    """Text that only mentions a push, code that reports already or opts out, and non-Python kernels are untouched."""
    files = [_kernel(ws, "a", "print(1)\n"),
             _kernel(ws, "b", "# kgmon: off\nprint(1)\n"),
             _kernel(ws, "c", "x <- 1\n", code_file="c.R", language="r")]
    before = [f.read_bytes() for f in files]
    for cmd in ['git commit -m "kaggle kernels push -p kernels/a"', "grep -rn 'kaggle kernels push -p kernels/a' .",
                "echo kaggle kernels push -p kernels/a", "kaggle kernels push --help",
                "kaggle kernels push -p kernels/b", "kaggle kernels push -p kernels/c"]:
        assert hook(plugin_root, "pre_tool.py", bash(cmd, ws)) is None, cmd
    assert [f.read_bytes() for f in files] == before


def test_push_monitor_windows_forms(plugin_root, ws):
    """PowerShell with a quoted kaggle.exe path, CRLF line ends and a BOM: all kept."""
    f = _kernel(ws, "w", b"\xef\xbb\xbfimport os\r\nprint(os.name)\r\n")
    payload = {"hook_event_name": "PreToolUse", "tool_name": "PowerShell", "cwd": str(ws),
               "tool_input": {"command": '& "C:\\Program Files\\Py\\Scripts\\kaggle.exe" kernels push -p kernels\\w'}}
    assert "added the hardware monitor" in hook(plugin_root, "pre_tool.py", payload)["additionalContext"]
    data = f.read_bytes()
    assert data.startswith(b"\xef\xbb\xbf# >>> kaggle-grandmaster") and data.endswith(b"import os\r\nprint(os.name)\r\n")
    assert b"\n" not in data.replace(b"\r\n", b"")


def test_push_monitor_note_rides_with_the_quota_ask(plugin_root, ws):
    _gpu_kernel(ws, quota_h=2.0)
    (ws / "kernels" / "x-train" / "x.py").write_text("print(1)\n")
    out = hook(plugin_root, "pre_tool.py", bash("kaggle kernels push -p kernels/x-train -t 21600", ws))
    assert out["permissionDecision"] == "ask" and "added the hardware monitor" in out["additionalContext"]


def test_monitor_cell_stays_quiet_in_spawned_workers(plugin_root, tmp_path):
    """multiprocessing's spawn re-imports the script as __mp_main__: the cell must not start a monitor there."""
    cell = plugin_root / "templates" / "kaggle_monitor_cell.py"
    code = f"import runpy; runpy.run_path({str(cell)!r}, run_name='__mp_main__')"
    r = subprocess.run([sys.executable, "-c", code], env={**os.environ, "KAGGLE_WORKING": str(tmp_path)},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert _kgmon_lines(r.stdout) == []


@pytest.mark.parametrize("cmd,runs", [
    ("kaggle kernels push -p k", True),
    ("conda run -n kg kaggle kernels push -p k", True),
    ("timeout 600 /opt/conda/bin/kaggle kernels push -p k", True),
    ("out=$(kaggle kernels push -p k 2>&1)", True),
    ('bash -c "cd A && kaggle kernels push -p ."', True),
    ('powershell -Command "kaggle kernels push -p ."', True),
    ("if [ -d k ]; then kaggle kernels push -p k; fi", True),
    ('git commit -m "kaggle kernels push -p k"', False),
    ("echo kaggle kernels push -p k", False),
    ("git log --grep kaggle kernels push", False),
    ("python -c \"print('kaggle kernels push')\"", False),
])
def test_push_monitor_only_for_real_calls(plugin_root, cmd, runs):
    sys.path.insert(0, str(plugin_root / "hooks"))
    import _kgmon as K
    assert K.is_call(cmd, cmd.index("kaggle")) is runs
