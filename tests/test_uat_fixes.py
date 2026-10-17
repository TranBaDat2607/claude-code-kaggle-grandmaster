"""Regression tests for the UAT reports #15-#26: backlog costs, GPU reservations and TPU support, pushes of
any kernel folder, holdout+LB workspaces, LB-only ledger records, dataset downloads, the GBDT template and
the credential guard."""

import ast
import datetime as dt
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kgkit import backlog as B
from kgkit import datasets as DS
from kgkit import gpu as G
from kgkit import kernels as K
from kgkit import state as S
from kgkit.experiment import Ledger


def _cp(stdout="", rc=0, stderr=""):
    return subprocess.CompletedProcess([], rc, stdout, stderr)


class FakeKaggle:
    """Stands in for gpu.run_kaggle: maps a command prefix to output and records every call."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, args, timeout=600):
        self.calls.append(list(args))
        for prefix, out in self.routes:
            if args[:len(prefix)] == list(prefix):
                return out(args) if callable(out) else out
        return _cp("", 1, "no route")


@pytest.fixture
def ws(tmp_path):
    S.CompetitionState(slug="demo", metric="auc", deadline=(dt.date.today() + dt.timedelta(days=20)).isoformat()
                       ).save(tmp_path)
    return tmp_path


def _quota(root, gpu_left=18.0, tpu_left=20.0, fetched=None, refresh="2099-01-03T00:00:00Z"):
    q = {"used_h": 30 - gpu_left, "remaining_h": gpu_left, "total_h": 30.0, "refresh_at": refresh,
         "fetched_at": fetched or G._iso(G._now()),
         "tpu": {"used_h": 20 - tpu_left, "remaining_h": tpu_left, "total_h": 20.0,
                 "refresh_at": "2099-01-03T00:00:00Z", "fetched_at": fetched or G._iso(G._now())}}
    G.quota_path(root).write_text(json.dumps(q))
    return q


# ----------------------------------------------------------------------------- #15 backlog
def test_backlog_add_keeps_status_and_result(ws):
    bl = B.Backlog(ws)
    i = bl.add("S6 as a 4th equal leg", gain=1, prob=0.05, cost=2, status="dropped", result="too correlated")
    assert i["status"] == "dropped" and i["result"] == "too correlated"
    assert bl.ranked() == []  # a dropped idea never enters the ranked to-do list
    with pytest.raises(ValueError):
        bl.add("x", status="maybe")


def test_backlog_background_gpu_hours_rank_against_the_quota(ws):
    _quota(ws, gpu_left=18.0, refresh=G._iso(G._now() + dt.timedelta(hours=32)))  # + 3 resets before the deadline
    bl = B.Backlog(ws)
    bl.add("re-decide final picks", gain=3, prob=0.6, cost=0.5)
    bl.add("R7 OOF teacher", gain=2, prob=0.4, cost=8)
    retrain = bl.add("S6-recipe retrain", gain=3, prob=0.35, cost=2, gpu_hours=28)
    ranked = [i["idea"] for i in bl.ranked()]
    assert ranked.index("S6-recipe retrain") < ranked.index("R7 OOF teacher")
    assert bl.gpu_budget() == pytest.approx(18 + 3 * 30)
    # 28 GPU hours fit in the budget, so they count at GPU_WEIGHT
    assert bl.score(retrain) == pytest.approx(3 * 0.35 / (2 + B.GPU_WEIGHT * 28))
    # beyond the budget the hours count in full
    assert B.score(retrain, gpu_budget=10) == pytest.approx(3 * 0.35 / (2 + B.GPU_WEIGHT * 10 + 18))
    table = bl.table()
    assert "| human h | GPU h |" in table and "GPU budget until the deadline" in table


def test_backlog_cli_add_status(ws, monkeypatch, capsys):
    from kgkit import cli

    monkeypatch.chdir(ws)
    cli.main(["backlog", "add", "idea", "--gain", "2", "--cost", "1", "--status", "dropped", "--result", "no"])
    assert "status dropped" in capsys.readouterr().out
    assert B.Backlog(ws).items()[0]["result"] == "no"


# ----------------------------------------------------------------------------- #16 reservations
def test_finished_and_superseded_runs_release_their_reservation(ws):
    _quota(ws, gpu_left=18.23)
    now = G._iso(G._now())
    for k in ("u/cache-a", "u/cache-b"):
        G.record_push(ws, {"kernel": k, "version": 1, "hours": 12.0, "pushed_at": now, "status": "pushed"})
    G.record_push(ws, {"kernel": "u/dev", "version": 1, "hours": 12.0, "pushed_at": now, "status": "pushed"})
    G.record_push(ws, {"kernel": "u/dev", "version": 2, "hours": 2.0, "pushed_at": now, "status": "pushed"})
    assert [r["status"] for r in G.runs(ws)] == ["pushed", "pushed", "superseded", "pushed"]
    assert G.available_hours(ws) == pytest.approx(18.23 - 26, abs=0.05)
    G.note_status(ws, "u/cache-a", "complete")
    G.note_status(ws, "u/cache-b", "error")
    assert G.available_hours(ws) == pytest.approx(18.23 - 2, abs=0.05)
    assert G.release(ws, "u/dev") == 1 and G.available_hours(ws) == pytest.approx(18.23)
    assert G.active_runs(ws) == []


def test_refresh_runs_and_quota_report_list_reservations(ws, monkeypatch):
    _quota(ws, gpu_left=10.0)
    G.record_push(ws, {"kernel": "u/a", "version": 3, "hours": 4.0, "pushed_at": G._iso(G._now()),
                       "status": "pushed", "via": "kaggle-cli"})
    txt = G.quota_report(ws, refresh=False)
    assert "u/a v3 [pushed, cap 4h, via kaggle CLI]" in txt and "gpu release" in txt
    fake = FakeKaggle([(["kernels", "status"], _cp('u/a has status "KernelWorkerStatus.COMPLETE"'))])
    monkeypatch.setattr(G, "run_kaggle", fake)
    assert G.refresh_runs(ws) == {"u/a": "complete"}
    assert G.runs(ws)[-1]["status"] == "complete" and G.available_hours(ws) == pytest.approx(10.0)


def test_stale_entries_are_not_in_flight(ws):
    old = G._iso(G._now() - dt.timedelta(hours=30))
    G.log_run(ws, {"kernel": "u/old", "hours": 12.0, "pushed_at": old, "status": "pushed"})
    G.log_run(ws, {"kernel": "u/tpu", "hours": 8.8, "accelerator": "TpuV5E8", "pushed_at": old, "status": "queued"})
    assert [r["kernel"] for r in G.active_runs(ws)] == ["u/tpu"]  # a TPU job may queue for a day
    _quota(ws)
    assert G.in_flight_hours(ws, G.cached_quota(ws, "TPU"), "TPU") == pytest.approx(8.8)  # queued: full cap
    assert G.in_flight_hours(ws, G.cached_quota(ws)) == 0


def test_kaggle_username_from_config_view(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))  # no kaggle.json: an access-token login
    monkeypatch.setattr(G, "_USERNAME", {})
    out = "Configuration values from x\n- username: tranbadat\n- auth_method: ACCESS_TOKEN\n- path: None\n"
    fake = FakeKaggle([(["config", "view"], _cp(out))])
    monkeypatch.setattr(G, "run_kaggle", fake)
    assert G.kaggle_username() == "tranbadat"
    assert G.kaggle_username() == "tranbadat" and len(fake.calls) == 1  # probed once
    (tmp_path / "kaggle.json").write_text(json.dumps({"username": "fromfile", "key": "k"}))
    assert G.kaggle_username() == "fromfile"


def test_diagnose_short_run_is_not_input_bound():
    rep = {"gpus": ["T4", "T4"], "elapsed_s": 324, "resource": "GPU",
           "gpu_util": {"0": {"util_mean": 10.0, "n": 10}, "1": {"util_mean": 0.0, "n": 10}}}
    msgs = " ".join(G.diagnose(rep))
    assert "too short to diagnose" in msgs and "input-bound" not in msgs and "idle" not in msgs
    rep.update(elapsed_s=7200, gpu_util={"0": {"util_mean": 30.0, "n": 240}, "1": {"util_mean": 31.0, "n": 240}})
    assert "input-bound" in " ".join(G.diagnose(rep))


# ----------------------------------------------------------------------------- #19 push of any kernel folder
def _notebook_kernel(root, gpu=True, sources=()):
    k = root / "kernels" / "r6-gold58"
    k.mkdir(parents=True)
    meta = {"id": "me/r6-gold58", "code_file": "nb.ipynb", "kernel_type": "notebook", "enable_gpu": gpu,
            "enable_tpu": False, "kernel_sources": list(sources)}
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (k / "kernel-metadata.json").write_text(json.dumps(meta))
    (k / "nb.ipynb").write_text("{}")
    return k


def _push_routes(status_by_kernel=None, busy=()):
    status_by_kernel = status_by_kernel or {}

    def status(args):
        return _cp(f'{args[2]} has status "{status_by_kernel.get(args[2], "KernelWorkerStatus.COMPLETE")}"')

    rows = [{"ref": k, "lastRunTime": G._now().strftime("%Y-%m-%dT%H:%M:%S")} for k in busy]
    return [(["kernels", "status"], status),
            (["kernels", "list"], _cp(json.dumps(rows))),
            (["quota"], _cp(json.dumps([{"resource": "GPU", "used": "10h", "remaining": "20h", "total": "30h",
                                          "refreshAt": "2099-01-03T00:00:00"},
                                         {"resource": "TPU", "used": "0h", "remaining": "20h", "total": "20h",
                                          "refreshAt": "2099-01-03T00:00:00"}]))),
            (["kernels", "push"], _cp("Kernel version 7 successfully pushed.  Please check progress at "
                                      "https://www.kaggle.com/code/me/r6-gold58"))]


def test_push_notebook_folder_needs_hours_then_logs(ws, monkeypatch):
    k = _notebook_kernel(ws)
    fake = FakeKaggle(_push_routes())
    monkeypatch.setattr(G, "run_kaggle", fake)
    with pytest.raises(SystemExit, match="--hours"):
        G.push(ws, k)
    rec = G.push(ws, k, hours=0.75)
    assert rec["kernel"] == "me/r6-gold58" and rec["version"] == 7 and rec["jobs"] == ["notebook"]
    assert rec["accelerator"] == "NvidiaTeslaT4" and rec["resource"] == "GPU"
    push = next(c for c in fake.calls if c[:2] == ["kernels", "push"])
    assert push[push.index("-t") + 1] == "2700" and "--accelerator" in push
    assert G.runs(ws)[-1]["hours"] == 0.75


def test_push_refuses_while_an_input_kernel_runs(ws, monkeypatch):
    k = _notebook_kernel(ws, sources=["me/s6t-r1-train"])
    fake = FakeKaggle(_push_routes({"me/s6t-r1-train": "KernelWorkerStatus.RUNNING"}))
    monkeypatch.setattr(G, "run_kaggle", fake)
    with pytest.raises(SystemExit, match="bound when a kernel is pushed"):
        G.push(ws, k, hours=1)
    assert not any(c[:2] == ["kernels", "push"] for c in fake.calls)


def test_push_refuses_when_gpu_session_slots_are_full(ws, monkeypatch):
    k = _notebook_kernel(ws)
    now = G._iso(G._now())
    for kern in ("me/a", "me/b"):
        G.log_run(ws, {"kernel": kern, "accelerator": "NvidiaTeslaT4", "hours": 3, "pushed_at": now, "status": "running"})
    fake = FakeKaggle(_push_routes({"me/a": "KernelWorkerStatus.RUNNING", "me/b": "KernelWorkerStatus.QUEUED"},
                                   busy=["me/a", "me/b"]))
    monkeypatch.setattr(G, "run_kaggle", fake)
    with pytest.raises(SystemExit, match="2 batch GPU session"):
        G.push(ws, k, hours=1)


def test_cancel_advice():
    assert "kaggle kernels delete me/x -y" in G.cancel_advice("me/x", "queued")
    assert "Stop" in G.cancel_advice("me/x", "running")


# ----------------------------------------------------------------------------- #18 TPU
def test_parse_quota_keeps_tpu(ws):
    q = G.parse_quota('[{"resource": "GPU", "used": "12.86h", "remaining": "17.14h", "total": "30.00h", '
                      '"refreshAt": "2026-10-24T00:00:00"}, {"resource": "TPU", "used": "2.18h", '
                      '"remaining": "17.82h", "total": "20.00h", "refreshAt": "2026-10-24T00:00:00"}]')
    assert q["remaining_h"] == pytest.approx(17.14) and q["tpu"]["remaining_h"] == pytest.approx(17.82)
    G.quota_path(ws).write_text(json.dumps(q))
    assert G.cached_quota(ws, "TPU")["total_h"] == 20
    assert "Kaggle TPU quota" in G.quota_report(ws, refresh=False)


def test_tpu_build_runner_and_collect(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    S.CompetitionState(slug="demo-comp", metric="auc").save(root)
    (root / "data").mkdir()
    pd.DataFrame({"id": range(8), "fold": np.arange(8) % 2}).to_csv(root / "data" / "folds.csv", index=False)
    (root / "src").mkdir()
    (root / "src" / "chip.py").write_text(
        "import os, sys\nprint('fold', sys.argv[1], 'chip', os.environ.get('TPU_VISIBLE_CHIPS'), "
        "os.environ.get('PJRT_DEVICE'), os.environ.get('TPU_PROCESS_ADDRESSES'))\n")
    with pytest.raises(SystemExit, match="session limit"):
        G.build_kernel(root, "x", f'"{sys.executable}" src/chip.py {{fold}}', hours=10, user="me", accelerator="TpuV5E8")
    with pytest.raises(SystemExit, match="TpuV5E8"):
        G.build_kernel(root, "x", "true", hours=1, user="me", accelerator="TpuV5E4")
    kdir = G.build_kernel(root, "x", f'"{sys.executable}" src/chip.py {{fold}}', hours=8.8, user="me",
                          accelerator="TpuV5E8")
    meta = json.loads((kdir / "kernel-metadata.json").read_text())
    assert meta["enable_tpu"] is True and meta["enable_gpu"] is False and meta["machine_shape"] == "TpuV5E8"
    assert json.loads((kdir / "kg-train.json").read_text())["resource"] == "TPU"
    base = tmp_path / "k"
    (base / "input" / "demo-comp").mkdir(parents=True)
    env = dict(os.environ, KAGGLE_INPUT=str(base / "input"), KAGGLE_WORKING=str(base / "working"),
               KG_RUN_DIR=str(base / "run"), KG_FAKE_TPU_CHIPS="2")
    env.pop("PYTHONPATH", None)
    r = subprocess.run([sys.executable, str(kdir / meta["code_file"])], env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "tpu chip 0" in r.stdout and "tpu chip 1" in r.stdout and "TPU-hours" in r.stdout
    logs = {f.name: f.read_text() for f in (base / "working" / "logs").glob("x_fold*.log")}
    chips = sorted(l.split()[3] for l in logs.values())
    assert chips == ["0", "1"] and all("TPU localhost:847" in l for l in logs.values())
    rep = json.loads((base / "working" / "kg_run.json").read_text())
    assert rep["resource"] == "TPU" and rep["gpus"] == [] and len(rep["devices"]) == 2
    G.log_run(root, {"kernel": "me/x-train", "version": 1, "accelerator": "TpuV5E8", "resource": "TPU",
                     "hours": 8.8, "pushed_at": G._iso(G._now()), "status": "running"})
    res = G.collect(root, "me/x-train", from_dir=base / "working")
    run = G.runs(root)[-1]
    assert run.get("tpu_hours") is not None and run.get("gpu_hours") is None
    assert "TPU-hours" in res["summary"] and "total GPU-hours accounted: 0.00" in G.run_table(root)


def test_plan_suggests_tpu_when_gpu_is_gone(ws):
    _quota(ws, gpu_left=0.2, tpu_left=20)
    txt = G.plan(ws, refresh=False)
    assert "TPU: 20.0h" in txt and "GPU quota is used up" in txt


def test_hook_resource_for_tpu_metadata(plugin_root):
    sys.path.insert(0, str(plugin_root / "hooks"))
    import _common as C

    assert C.push_resource({}, {"enable_tpu": True, "machine_shape": "TpuV5E8"}) == "TPU"
    assert C.push_resource({}, {"enable_gpu": True}) == "GPU"
    assert C.push_resource({"accelerator": "none"}, {"enable_gpu": True}) is None
    assert C.push_resource({}, {}) is None


# ----------------------------------------------------------------------------- #17 holdout + LB
def test_config_sets_validation_and_status_shows_holdout(ws, monkeypatch, capsys):
    from kgkit import cli

    monkeypatch.chdir(ws)
    cli.main(["config", "validation=holdout+lb", "holdout_n=58"])
    st = S.load(ws)
    assert st.holdout and st.holdout_n == 58
    with pytest.raises(SystemExit):
        cli.main(["config", "validation=whatever"])
    led = Ledger(ws)
    for i, (cv, lb) in enumerate([(0.90, 0.950), (0.92, 0.951), (0.91, 0.957), (0.93, 0.955)]):
        led.log(f"leg{i}", cv)
        led.attach_lb(str(i + 1), lb)
    capsys.readouterr()
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "validation: holdout+lb (holdout n=58)" in out
    assert "fixed holdout (n=58, SE ~0.0" in out and "CV-LB correlation over 4 subs" in out


def test_kernel_review_inference_only(tmp_path):
    d = tmp_path / "nb"
    d.mkdir()
    nb = {"cells": [{"cell_type": "code", "source": "!pip install -q x\n"  # helper sources embedded as strings
                                                     "SOURCES = {'train.py': 'model.fit(x, y)\\nloss.backward()'}\n"},
                    {"cell_type": "code", "source": "import torch\nm = torch.load('/kaggle/input/w/m.pt')\n"
                                                     "sub = pd.read_csv('sample_submission.csv')\n"
                                                     "sub.to_csv('submission.csv', index=False)\n"}]}
    (d / "nb.ipynb").write_text(json.dumps(nb))
    (d / "kernel-metadata.json").write_text(json.dumps({"dataset_sources": ["a/w"]}))
    rev = K.analyse_dir(d, "a/nb", "comp", holdout=True)
    text = (d / "REVIEW.md").read_text()
    assert rev["inference_only"] and "CV scheme**: n/a: inference-only notebook" in text
    assert "pinned to a version" in text and "on the holdout" in text and "Re-run on data/folds.csv" not in text
    assert "kaggle.com/code/a/nb" in text


def test_session_brief_in_holdout_mode(plugin_root, ws):
    import shutil

    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("needs a POSIX sh")
    st = S.load(ws)
    st.validation, st.holdout_n = "holdout+lb", 58
    st.save(ws)
    Ledger(ws).log("blend-public", None, notes="fork of public notebooks")
    Ledger(ws).attach_lb("1", 0.957)
    Ledger(ws).decide("1", "baseline")
    r = subprocess.run([sh, str(plugin_root / "hooks" / "run.sh"), "session_start.py"],
                       input=json.dumps({"hook_event_name": "SessionStart", "cwd": str(ws)}),
                       capture_output=True, text=True, timeout=60)
    ctx = json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "folds.csv" not in ctx and "fixed holdout (n=58)" in ctx
    assert "best public LB 0.95700" in ctx and "0001_blend-public LB-only" in ctx


# ----------------------------------------------------------------------------- #25 LB-only records
def test_lb_only_records_stay_out_of_the_correlation(ws, monkeypatch, capsys):
    from kgkit import cli

    led = Ledger(ws)
    stack = led.log("stack", 0.962194)
    led.attach_lb(stack["id"], 0.96185)
    for name in ("blend-a", "blend-b"):  # the UAT: blends logged with the stack's CV
        r = led.log(name, 0.9622, tags=["lb-blend"])
        led.attach_lb(r["id"], 0.96189)
    assert led.cv_lb_correlation() is None
    assert "not reported" in led.cv_lb_summary() and "2 LB-only submission(s) left out" in led.cv_lb_summary()
    monkeypatch.chdir(ws)
    cli.main(["ledger", "add", "blend-c", "lb-only", "public files"])
    rec = led.records()[-1]
    assert rec["cv"] is None and "lb-only" in rec["tags"]
    capsys.readouterr()
    cli.main(["ledger", "lb", rec["id"], "0.9619"])
    out = capsys.readouterr().out
    assert "CV LB-only -> public LB 0.9619" in out and "pearson" not in out
    assert "| LB-only |" in led.table() and led.best(1)[0]["id"] == stack["id"]
    cli.main(["status"])
    assert "best public LB: 0.96190" in capsys.readouterr().out
    for cv, lb in [(0.95, 0.951), (0.955, 0.954), (0.958, 0.957)]:
        led.attach_lb(led.log("m", cv)["id"], lb)
    corr = led.cv_lb_correlation()
    assert corr is not None and corr["n"] == 4


# ----------------------------------------------------------------------------- #23 datasets
def test_datasets_pull_falls_back_to_per_file(tmp_path, monkeypatch):
    pages = {None: ("Next Page Token = tok2\n", [{"name": "a.csv", "size": 3}, {"name": "sub/b.npy", "size": 4}]),
             "tok2": ("", [{"name": "big.parquet", "size": 5}])}

    def files(args):
        tok = args[args.index("--page-token") + 1] if "--page-token" in args else None
        head, rows = pages[tok]
        return _cp(head + json.dumps(rows))

    def download(args):
        if "-f" not in args:
            return _cp("Dataset URL: x", 1, "404 Client Error: Not Found for url: https://api.kaggle.com/v1/"
                                            "datasets.DatasetApiService/DownloadDataset")
        name, dest = args[args.index("-f") + 1], Path(args[args.index("-p") + 1])
        base = name.split("/")[-1]
        if base.endswith(".parquet"):  # the CLI zips large files
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr(base, b"12345")
            (dest / (base + ".zip")).write_bytes(buf.getvalue())
        else:
            (dest / base).write_bytes(b"x" * (3 if base == "a.csv" else 4))
        return _cp("")

    fake = FakeKaggle([(["datasets", "files"], files), (["datasets", "download"], download)])
    monkeypatch.setattr(DS, "run_kaggle", fake)
    res = DS.pull("o/lib", tmp_path / "out")
    assert res["mode"] == "per-file" and res["failed"] == [] and "404" in res["whole_error"]
    assert sorted(res["files"]) == ["a.csv", "big.parquet", "sub/b.npy"]
    assert (tmp_path / "out" / "sub" / "b.npy").is_file() and (tmp_path / "out" / "big.parquet").read_bytes() == b"12345"
    assert not (tmp_path / "out" / "big.parquet.zip").exists()
    n = len(fake.calls)
    DS.pull("o/lib", tmp_path / "out", files="*.csv")  # present with the right size: skipped
    assert not any(c[:2] == ["datasets", "download"] for c in fake.calls[n:])


# ----------------------------------------------------------------------------- #22 subprocess decoding
@pytest.mark.parametrize("rel", ["kgkit/gpu.py", "kgkit/discussions.py", "kgkit/kernels.py", "kgkit/pages.py",
                                 "kgkit/cli.py", "kgkit/datasets.py"])
def test_captured_kaggle_output_is_decoded_as_utf8(plugin_root, rel):
    """On Windows, text=True alone decodes with cp1252 and the CLI's tqdm bar (byte 0x8f) crashes it."""
    tree = ast.parse((plugin_root / rel).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "run":
            kw = {k.arg: k.value for k in node.keywords}
            first = node.args[0] if node.args else None
            is_git = isinstance(first, ast.List) and first.elts and getattr(first.elts[0], "value", "") == "git"
            if "capture_output" in kw and not is_git:
                assert "encoding" in kw, f"{rel}:{node.lineno} captures output without encoding="


# ----------------------------------------------------------------------------- #24 GBDT template
def test_gbdt_template_name_follows_model(tmp_path, plugin_root):
    from test_templates import _make_ws, _run

    w = _make_ws(tmp_path, plugin_root, "binary")
    out = _run(["src/train_gbdt.py", "--model", "hgb", "--smoke", "--device", "cuda"], w, plugin_root)
    assert "[hgb_baseline]" in out and "no effect on hgb" in out
    src = (plugin_root / "templates" / "train_gbdt.py").read_text(encoding="utf-8")
    assert 'DEVICE_PARAMS = {"cuda": {"xgb": {"device": "cuda"}, "cat": {"task_type": "GPU"}}}' in src


# ----------------------------------------------------------------------------- #26 credential guard
def _pre(plugin_root, cmd, cwd):
    import shutil

    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("needs a POSIX sh")
    r = subprocess.run([sh, str(plugin_root / "hooks" / "run.sh"), "pre_tool.py"],
                       input=json.dumps({"hook_event_name": "PreToolUse", "tool_name": "Bash",
                                         "tool_input": {"command": cmd}, "cwd": str(cwd)}),
                       capture_output=True, text=True, timeout=60)
    return json.loads(r.stdout)["hookSpecificOutput"] if r.stdout.strip() else None


def test_credential_guard_allows_username_and_prose(plugin_root, tmp_path):
    allowed = [
        "python -c \"import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])\"",
        "cat > issue.md <<'EOF'\nThe username comes from the username field of kaggle.json; we ran\n"
        "cat ~/.kaggle/kaggle.json once and the hook blocked it.\nEOF",
        'git commit -m "docs: never print ~/.kaggle/access_token or kaggle.json"',
        'gh issue create --title "hook" --body "reading kaggle.json is blocked; type the path"',
        "python -m kgkit whoami",
    ]
    for cmd in allowed:
        assert _pre(plugin_root, cmd, tmp_path) is None, cmd
    denied = [
        "python -c \"import json;print(json.load(open('/home/u/.kaggle/kaggle.json')))\"",
        "python -c \"print(open('/home/u/.kaggle/kaggle.json').read())\"",
        "python -c \"import json;d=json.load(open('kaggle.json'));print(d['username'], d['key'])\"",
        "python - <<'EOF'\nprint(open('/home/u/.kaggle/kaggle.json').read())\nEOF",
        "ls && cat ~/.kaggle/kaggle.json",
        "(Get-Content $HOME\\.kaggle\\kaggle.json | ConvertFrom-Json).username",
        "jq . ~/.kaggle/kaggle.json",
        "grep key ~/.kaggle/kaggle.json",
        "python -c \"import os; print(os.environ['KAGGLE_KEY'])\"",
    ]
    for cmd in denied:
        out = _pre(plugin_root, cmd, tmp_path)
        assert out and out["permissionDecision"] == "deny", cmd


def test_compound_command_checks_every_push(plugin_root, tmp_path):
    S.CompetitionState(slug="demo").save(tmp_path)
    k = tmp_path / "kernels" / "r6"
    k.mkdir(parents=True)
    (k / "kernel-metadata.json").write_text(json.dumps({"id": "me/r6", "enable_gpu": True,
                                                         "machine_shape": "NvidiaTeslaT4"}))
    _quota(tmp_path, gpu_left=0.5)
    cmd = ("kaggle kernels push --help 2>&1 | grep accel | head; "
           "kaggle kernels push -p kernels/r6 -t 2700 2>&1 | tail -3")
    out = _pre(plugin_root, cmd, tmp_path)
    assert out and out["permissionDecision"] == "ask" and "me/r6" in out["permissionDecisionReason"]
    import shutil

    post = {"hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_input": {"command": cmd},
            "tool_response": {"stdout": "usage: kaggle kernels push ...\nKernel version 5 successfully pushed.  "
                                        "Please check progress at https://www.kaggle.com/code/me/r6"},
            "cwd": str(tmp_path)}
    r = subprocess.run([shutil.which("sh"), str(plugin_root / "hooks" / "run.sh"), "post_tool.py"],
                       input=json.dumps(post), capture_output=True, text=True, timeout=60)
    assert "me/r6 v5 logged" in r.stdout
    run = G.runs(tmp_path)[-1]
    assert run["kernel"] == "me/r6" and run["version"] == 5 and run["hours"] == 0.75
