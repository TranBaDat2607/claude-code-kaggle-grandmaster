"""Regression tests for the UAT reports #15-#26 and #30: backlog costs, GPU reservations and TPU support, pushes
of any kernel folder, holdout+LB workspaces, LB-only ledger records, dataset downloads, the GBDT template, the
credential guard and public notebooks' scores on record."""

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
    assert "fixed holdout (n=58" in out and "CV-LB correlation over 4 subs" in out
    # #32: no threshold for differences derived from n and the score (a binomial SE of an accuracy)
    assert "not measured yet" in out and "SE ~" not in out and "are noise" not in out


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


# ----------------------------------------------------------------------------- #30 public scores of notebooks
from types import SimpleNamespace  # noqa: E402

from kgkit import kaggle_search as KS  # noqa: E402


def _doc(ref, score):
    owner, slug = ref.split("/")
    return SimpleNamespace(owner_user=SimpleNamespace(user_name=owner), owner_organization=None, slug=slug,
                           kernel_document=SimpleNamespace(best_public_score=score))


def test_search_lookup_matches_owner_and_slug():
    calls = []

    def search(query, pages, by_votes):
        calls.append((query, pages, by_votes))
        if query == "my comp":
            return [_doc("A/one", 0.958), _doc("x/two", 0.99), _doc("c/none", 0.0)]  # x/two: same slug, other owner
        if query == "Two title":
            return [_doc("b/two", 0.957)]
        raise RuntimeError("HTTP 429")

    items = [("a/one", "One"), ("b/two", "Two title"), ("c/none", "None"), ("d/gone", "Gone")]
    res = KS.lookup(search, items, "my comp", 2)
    assert res["found"] == {"a/one": 0.958, "b/two": 0.957, "c/none": None}  # 0.0: none on record
    assert res["error"] == "RuntimeError: HTTP 429"  # d/gone failed: absent, i.e. unknown
    assert calls == [("my comp", 2, True), ("Two title", 1, False), ("Gone", 1, False)]
    assert KS.lookup(search, items, "my comp", budget=0)["found"].keys() == {"a/one", "c/none"}


def _fake_kaggle_sdk(root):
    """Just enough of the kaggle and kagglesdk packages for the search child, with a chatty authenticate()."""
    files = {
        "kaggle/__init__.py": "",
        "kaggle/api/__init__.py": "",
        "kaggle/api/kaggle_api_extended.py": (
            "from types import SimpleNamespace as N\n"
            "DOCS = {'demo comp': [('a', 'one', 0.958)], 'Two': [('b', 'two', 0.0)]}\n"
            "class Client:\n"
            "    def __enter__(self): return self\n"
            "    def __exit__(self, *a): return False\n"
            "    def list_entities(self, r):\n"
            "        docs = [N(owner_user=N(user_name=o), slug=s, kernel_document=N(best_public_score=v))\n"
            "                for o, s, v in DOCS.get(r.filters.query, [])]\n"
            "        return N(documents=docs, next_page_token='')\n"
            "    @property\n"
            "    def search(self): return N(search_api_client=self)\n"
            "class KaggleApi:\n"
            "    def authenticate(self): print('Warning: outdated API version')\n"
            "    def build_kaggle_client(self): return Client()\n"),
        "kagglesdk/__init__.py": "",
        "kagglesdk/search/__init__.py": "",
        "kagglesdk/search/types/__init__.py": "",
        "kagglesdk/search/types/search_api_service.py": (
            "class ListEntitiesFilters: pass\nclass ListEntitiesRequest: pass\n"),
        "kagglesdk/search/types/search_enums.py": (
            "class DocumentType: KERNEL = 5\n"
            "class ListSearchContentOrderBy: LIST_SEARCH_CONTENT_ORDER_BY_VOTES = 6\n"),
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)


def test_public_scores_runs_the_search_child(tmp_path, monkeypatch):
    _fake_kaggle_sdk(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("VERBOSE", "1")
    monkeypatch.setattr(K, "_search_python", lambda: sys.executable)
    found, err = K.public_scores([("a/one", "One"), ("b/two", "Two"), ("c/gone", "Gone")], "demo-comp")
    assert found == {"a/one": 0.958, "b/two": None} and err is None  # the SDK's print did not break the JSON line

    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        return subprocess.CompletedProcess(cmd, 1, "", "Traceback ...\nModuleNotFoundError: no module kgkit\n")
    monkeypatch.setattr(K.subprocess, "run", fake_run)
    assert K.public_scores([("a/b", "t")]) == ({}, "the search lookup failed: ModuleNotFoundError: no module kgkit")
    assert "VERBOSE" not in seen["env"] and seen["encoding"] == "utf-8"

    def timeout(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(K.subprocess, "run", timeout)
    assert K.public_scores([("a/b", "t")]) == ({}, "the search lookup timed out")


def test_search_child_reports_missing_sdk(tmp_path):
    (tmp_path / "kaggle").mkdir()
    (tmp_path / "kaggle" / "__init__.py").write_text("raise ImportError('no kaggle here')\n")
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    r = subprocess.run([sys.executable, "-m", "kgkit.kaggle_search"], input='{"items": [["a/b", "t"]]}',
                       capture_output=True, encoding="utf-8", cwd=str(Path(K.__file__).resolve().parents[1]), env=env)
    res = json.loads(r.stdout)
    assert r.returncode == 0 and res["found"] == {} and "no Kaggle search API" in res["error"]


@pytest.mark.parametrize("title,score,flag", [
    ("RSNA Knee | CoAtNet + R6 + 2.5D MIL | LB 0.958", 0.958, None),
    ("This is best public dataset new version =0.9586", 0.958, None),        # rounded to the LB's 3 decimals
    ("≈0.957 | Latest dataset| score in one click!", 0.956, None),          # one unit of rounding
    ("RSNA Knee public pipeline slice (measured 0.943, not 0.946)", 0.943, None),
    ("RSNA Knee  CoAtNet + ConvNeXt MIL | LB 0.950", 0.95, None),            # the float drops the trailing 0
    ("R6 2.5D v1.25 blend", 0.957, None),                                    # version numbers are not claims
    ("RSNA Knee 0.956+:Micro-Boost & Clinical Tri-Reader", 0.95, "title claims 0.956; best public score on record 0.95"),
    ("RSNA Knee Fast Parent 0.957 v2", 0.939, "title claims 0.957; best public score on record 0.939"),
    ("[LB 0.94650] EV Purchase GM Ensemble", 0.94638, "title claims 0.94650; best public score on record 0.94638"),
    ("Magic LB 0.960", None, "title claims 0.960; no public score on record"),
    ("Plain title", None, None),
])
def test_title_claims_checked_against_the_record(title, score, flag):
    assert K.claim_note(title, score) == flag


def test_top_table_scores_flags_and_order(monkeypatch):
    rows = [{"ref": "a/x", "title": "LB 0.960", "totalVotes": "500", "lastRunTime": "t"},
            {"ref": "b/y", "title": "Y", "totalVotes": "3", "lastRunTime": "t"},
            {"ref": "c/z", "title": "Z", "totalVotes": "40", "lastRunTime": "t"},
            {"ref": "d/w", "title": "W", "totalVotes": "1", "lastRunTime": "t"},
            {"ref": "e/v", "title": "V", "totalVotes": "9", "lastRunTime": "t"}]
    monkeypatch.setattr(K, "public_scores", lambda items, comp: (
        {"a/x": None, "b/y": 0.957, "c/z": 0.958, "e/v": 0.957}, None))
    assert K.add_public_scores(rows, "comp") is None
    best = [r["ref"] for r in K.sort_by_score(rows)]
    assert best == ["c/z", "b/y", "e/v", "a/x", "d/w"]  # ties keep Kaggle's order; none on record, then unknown
    assert [r["ref"] for r in K.sort_by_score(rows, greater_is_better=False)][:3] == ["b/y", "e/v", "c/z"]
    lines = K.format_top(K.sort_by_score(rows)).splitlines()
    assert lines[0].startswith("score  votes  ref")
    assert lines[1].startswith("0.958     40  c/z")
    assert lines[4].startswith("    -    500  a/x") and lines[4].endswith("<- title claims 0.960; no public score on record")
    assert lines[5].startswith("    ?      1  d/w")
    assert K.format_top(rows, scores=False).splitlines()[1].startswith("  500  a/x")


def test_review_shows_the_public_score_on_record(tmp_path, monkeypatch):
    d = tmp_path / "nb"
    d.mkdir()
    (d / "nb.py").write_text("import lightgbm as lgb\n")
    (d / "kernel-metadata.json").write_text(json.dumps({"title": "Best LB 0.960"}))
    monkeypatch.setattr(K, "public_scores", lambda items, comp: pytest.fail("looked up without score_lookup"))
    K.analyse_dir(d, "a/nb", "comp")
    assert "Public score: unknown (not looked up)" in (d / "REVIEW.md").read_text()

    monkeypatch.setattr(K, "public_scores", lambda items, comp: ({"a/nb": 0.957}, None))
    rev = K.analyse_dir(d, "a/nb", "comp", score_lookup=True)
    text = (d / "REVIEW.md").read_text()
    assert rev["public"]["score"] == 0.957 and "Public score on record: 0.957" in text
    assert "**Title check**: title claims 0.960; best public score on record 0.957" in text
    assert "kaggle.com/code/a/nb" in text

    monkeypatch.setattr(K, "public_scores", lambda items, comp: ({"a/nb": None}, None))
    K.analyse_dir(d, "a/nb", "comp", score_lookup=True)
    text = (d / "REVIEW.md").read_text()
    assert "Public score on record: none" in text and "no public score on record" in text

    monkeypatch.setattr(K, "public_scores", lambda items, comp: ({}, "the search lookup timed out"))
    K.analyse_dir(d, "a/nb", "comp", score_lookup=True)
    text = (d / "REVIEW.md").read_text()
    assert "Public score: unknown (the search lookup timed out)" in text and "Title check" not in text


# ----------------------------------------------------------------------------- #31 CLI papercuts
def test_macro_auc_is_mean_col_auc_however_it_is_meant():
    from sklearn.metrics import roc_auc_score

    from kgkit import metrics as M

    m = M.get("macro_auc")
    assert m is M.get("mean_col_auc") is M.get("Macro AUC") is M.get("column-auc")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)  # multiclass codes against 4 score columns: one-vs-rest macro
    p = rng.random((400, 4)) + 0.4 * np.eye(4)[y]
    p /= p.sum(1, keepdims=True)
    ref = roc_auc_score(y, p, multi_class="ovr")
    assert m(y, p) == pytest.approx(ref) and m.influence(y, p)[0] == pytest.approx(ref)
    ml = (rng.random((300, 3)) < 0.3).astype(int)  # multilabel indicators
    s = rng.random((300, 3)) + 0.3 * ml
    assert m(ml, s) == pytest.approx(roc_auc_score(ml, s, average="macro"))
    yb, sb = ml[:, 0], s[:, 0]  # one binary column
    assert m(yb, sb) == pytest.approx(M.score("auc", yb, sb))


def test_unknown_metric_names_suggest_registered_ones():
    from kgkit import metrics as M

    assert M.suggest("macro_averaged_roc_auc")[0] == "mean_col_auc"
    assert M.suggest("mean_colauc") == ["mean_col_auc"]
    assert M.suggest("rmsee")[0] == "rmse" and M.suggest("xyz") == []
    with pytest.raises(KeyError, match="did you mean rmse"):
        M.get("rmsee")


def test_metric_given_to_adopt_and_config_is_resolved(tmp_path, monkeypatch, capsys):
    from kgkit import cli

    cli.main(["adopt", "rsna", "--dir", str(tmp_path), "--metric", "macro_auc", "--no-submissions"])
    out = capsys.readouterr().out
    st = S.load(tmp_path)
    assert st.metric == "mean_col_auc" and st.greater_is_better
    assert "macro_auc -> mean_col_auc" in out and "registry" not in out
    cli.main(["adopt", "rsna", "--dir", str(tmp_path), "--metric", "macro_averaged_auc_roc", "--no-submissions",
              "--force"])
    out = capsys.readouterr().out
    assert "not in kgkit's registry - did you mean" in out and "mean_col_auc" in out and "src/metric.py" not in out
    assert S.load(tmp_path).metric == "macro_averaged_auc_roc"

    monkeypatch.chdir(tmp_path)
    cli.main(["config", "metric=rmse"])  # the direction follows the metric
    st = S.load(tmp_path)
    assert st.metric == "rmse" and not st.greater_is_better
    cli.main(["config", "greater_is_better=false", "metric=mcauc"])  # unless it is set in the same call
    st = S.load(tmp_path)
    assert st.metric == "mean_col_auc" and not st.greater_is_better

    pd.DataFrame({"y": [0, 1, 1], "p": [0.2, 0.7, 0.9]}).to_csv(tmp_path / "d.csv", index=False)
    with pytest.raises(SystemExit, match="did you mean rmse"):  # any command: the hint, not a traceback
        cli.main(["score", "rmsee", "d.csv:y", "d.csv:p"])


def test_ledger_and_backlog_actions_show_their_own_arguments(ws, monkeypatch, capsys):
    from kgkit import cli

    monkeypatch.chdir(ws)
    monkeypatch.setenv("COLUMNS", "200")

    def help_of(*argv):
        with pytest.raises(SystemExit) as e:
            cli.main(list(argv))
        assert e.value.code == 0
        return capsys.readouterr().out

    out = help_of("ledger", "add", "--help")
    assert "name cv|auto|lb-only [notes ...]" in out and "--pred" in out and "--source" not in out
    out = help_of("ledger", "-h")
    assert "<exp_id> <public_score> [private_score]" in out and "<exp_id> <baseline|keep|discard|inconclusive>" in out
    out = help_of("backlog", "set", "--help")
    assert " id" in out and "--screen-gain" in out and "--first-experiment" not in out
    with pytest.raises(SystemExit) as e:
        cli.main(["ledger", "lb", "1"])
    assert e.value.code == 2 and "public_score" in capsys.readouterr().err

    # the command bodies still read the positionals in order; --notes joins the notes words
    cli.main(["ledger", "add", "lgbm", "0.81", "first", "try", "--notes", "seed 0"])
    cli.main(["ledger", "lb", "1", "0.8"])
    cli.main(["ledger", "decide", "1", "baseline", "the", "start"])
    r = Ledger(ws).get("1")
    assert r["notes"] == "first try seed 0" and r["lb_public"] == 0.8
    assert r["decision"] == "baseline" and r["decision_note"] == "the start"
    cli.main(["backlog", "add", "pseudo", "label", "--gain", "3"])
    cli.main(["backlog", "set", "1", "--status", "running"])
    assert B.Backlog(ws).items()[0]["idea"] == "pseudo label" and B.Backlog(ws).items()[0]["status"] == "running"


def test_status_shows_submissions_deadline_and_gpu_like_the_brief(plugin_root, ws, monkeypatch, capsys):
    from kgkit import cli

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    S.submissions_log(ws).write_text(json.dumps({"time_utc": now, "status": "submitted"}) + "\n")
    monkeypatch.chdir(ws)
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "Submissions logged today (UTC): 1/5\n" in out and "days left; phase: exploration" in out
    assert "Kaggle GPU quota: not fetched yet" in out

    with S.submissions_log(ws).open("a") as f:
        f.write(json.dumps({"time_utc": now, "status": "submitted", "remaining_today": 2}) + "\n")
    _quota(ws)
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "Submissions logged today (UTC): 2/5; Kaggle reported 2 left today for the team" in out
    assert "18.0h plannable of 30h" in out
    r = subprocess.run([sys.executable, str(plugin_root / "hooks" / "session_start.py")],
                       input=json.dumps({"hook_event_name": "SessionStart", "cwd": str(ws)}),
                       capture_output=True, text=True, timeout=60)
    ctx = json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"]
    st = S.load(ws)
    for line in (S.deadline_line(st), S.submissions_line(ws, st), G.quota_line(ws)):
        assert line and line in ctx and line in out


# ----------------------------------------------------------------------------- #37 relative paths after cd
def _hook(plugin_root, script, cmd, cwd, stdout=None):
    payload = {"hook_event_name": "PostToolUse" if stdout is not None else "PreToolUse", "tool_name": "Bash",
               "tool_input": {"command": cmd}, "cwd": str(cwd)}
    if stdout is not None:
        payload["tool_response"] = {"stdout": stdout}
    r = subprocess.run([sys.executable, str(plugin_root / "hooks" / script)], input=json.dumps(payload),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)["hookSpecificOutput"] if r.stdout.strip() else None


def _two_gpu_kernels(ws):
    base = ws / "experiments" / "phase4" / "kernels"
    for name in ("vj-gold58", "oai-a18"):
        (base / name).mkdir(parents=True)
        (base / name / "kernel-metadata.json").write_text(json.dumps(
            {"id": f"me/rsna-knee-{name}", "enable_gpu": True, "machine_shape": "NvidiaTeslaT4"}))
    return base


PUSHED_BOTH = "".join(f"Kernel version 2 successfully pushed.  Please check progress at "
                      f"https://www.kaggle.com/code/me/rsna-knee-{n}\n" for n in ("vj-gold58", "oai-a18"))


def test_each_push_is_read_from_the_folder_its_cd_left(plugin_root, ws):
    base = _two_gpu_kernels(ws)
    cmd = ("cd experiments/phase4/kernels/vj-gold58 && kaggle kernels push -p . -t 3600 && "
           "cd ../oai-a18 && kaggle kernels push -p . -t 3600")
    _quota(ws, gpu_left=0.5)
    out = _hook(plugin_root, "pre_tool.py", cmd, ws)  # before the command: cwd is where it starts
    assert out["permissionDecision"] == "ask" and "me/rsna-knee-vj-gold58 is capped" in out["permissionDecisionReason"]

    ended = base / "oai-a18"  # after the command: cwd is where it left the shell
    ctx = _hook(plugin_root, "post_tool.py", cmd, ended, PUSHED_BOTH)["additionalContext"]
    assert "me/rsna-knee-vj-gold58 v2 logged" in ctx and "me/rsna-knee-oai-a18 v2 logged" in ctx
    runs = [(r["kernel"], r["version"], r["name"], r["status"], r["hours"]) for r in G.runs(ws)]
    assert runs == [("me/rsna-knee-vj-gold58", 2, "vj-gold58", "pushed", 1.0),
                    ("me/rsna-knee-oai-a18", 2, "oai-a18", "pushed", 1.0)]
    assert _hook(plugin_root, "post_tool.py", cmd, ended, PUSHED_BOTH) is None  # each version is logged once
    assert len(G.runs(ws)) == 2


def test_push_after_a_cd_the_hook_cannot_follow_is_reported_not_guessed(plugin_root, ws):
    _two_gpu_kernels(ws)
    cmd = "for k in vj-gold58 oai-a18; do (cd experiments/phase4/kernels/$k && kaggle kernels push -p .); done"
    ctx = _hook(plugin_root, "post_tool.py", cmd, ws, PUSHED_BOTH)["additionalContext"]
    assert "me/rsna-knee-vj-gold58 v2, me/rsna-knee-oai-a18 v2 pushed, but the hook could not tell" in ctx
    assert "absolute `-p` path" in ctx and G.runs(ws) == []


@pytest.mark.parametrize("cmd, ended_in, want", [
    ("(cd A && kaggle kernels push -p .) && (cd B && kaggle kernels push -p .)", "", ["A", "B"]),
    ("pushd A && kaggle kernels push -p . && popd && kaggle kernels push -p B", "", ["A", "B"]),
    ("Set-Location kernels\\A; kaggle kernels push -p .; Set-Location ..\\B; kaggle kernels push -p .", "kernels/B",
     ["kernels/A", "kernels/B"]),
    ("cd A 2>/dev/null \\\n  && kaggle kernels push -p . 2>&1 | tail -3 && cd -", "", ["A"]),
    ('bash -c "cd A && kaggle kernels push -p ." && cd B', "B", ["A"]),
    ("git commit -m 'cd x' && kaggle kernels push -p A && cd A/sub", "A/sub", ["A"]),
    ("cd $KDIR && kaggle kernels push -p .", "A", [None]),
    ("if [ -d A ]; then cd A; fi; kaggle kernels push -p .", "A", [None]),
    ("cd A | cat; kaggle kernels push -p .", "", [None]),
])
def test_push_folder_follows_the_cd_steps_before_it(plugin_root, tmp_path, monkeypatch, cmd, ended_in, want):
    monkeypatch.delenv("CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR", raising=False)
    start = tmp_path / "ws"
    expected = [None if w is None else start / w for w in want]
    assert _push_dirs(plugin_root, cmd, start) == expected  # PreToolUse: cwd is where the command starts
    assert _push_dirs(plugin_root, cmd, start / ended_in, ran=True) == expected  # PostToolUse: where it ended


def _push_dirs(plugin_root, cmd, cwd, ran=False):
    sys.path.insert(0, str(plugin_root / "hooks"))
    import _common as C

    return [C.path_at({"tool_input": {"command": cmd}, "cwd": str(cwd)}, p["path"], p["at"], ran)
            for p in C.parse_pushes(cmd)]


def test_post_hook_undoes_only_the_cd_steps_it_can(plugin_root, tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR", raising=False)
    start = tmp_path / "ws"
    cmd = "kaggle kernels push -p . && cd .."
    assert _push_dirs(plugin_root, cmd, start) == [start]
    assert _push_dirs(plugin_root, cmd, tmp_path, ran=True) == [None]  # which child of tmp_path it left is lost
    # Claude Code can be set to return to the start directory after every command
    monkeypatch.setenv("CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR", "1")
    cmd = "cd A && kaggle kernels push -p . && cd ../B && kaggle kernels push -p ."
    assert _push_dirs(plugin_root, cmd, start, ran=True) == [start / "A", start / "B"]


def test_submit_and_upload_paths_follow_the_cd_before_them(plugin_root, ws):
    (ws / "subs").mkdir()
    (ws / "subs" / "good.csv").write_text("id,target\n1,0.1\n")
    cmd = 'cd subs && kaggle competitions submit demo -f good.csv -m "0001_x"'
    assert _hook(plugin_root, "pre_tool.py", cmd, ws) is None  # was denied: "not found: <ws>/good.csv"
    out = _hook(plugin_root, "pre_tool.py", cmd.replace("good", "missing"), ws)
    assert out["permissionDecision"] == "deny" and str(ws / "subs" / "missing.csv") in out["permissionDecisionReason"]
    _hook(plugin_root, "post_tool.py", cmd, ws / "subs", "Successfully submitted to Demo")
    logged = json.loads(S.submissions_log(ws).read_text().splitlines()[-1])
    assert logged["size"] == (ws / "subs" / "good.csv").stat().st_size and len(logged["sha256"]) == 64
    out = _hook(plugin_root, "pre_tool.py", cmd.replace("0001_x", "0002_again"), ws)
    assert out["permissionDecision"] == "ask" and "already submitted" in out["permissionDecisionReason"]

    up = ws / "kernels" / "w"
    up.mkdir(parents=True)
    (up / ".env").write_text("KAGGLE_KEY=1")
    out = _hook(plugin_root, "pre_tool.py", "cd kernels/w && kaggle datasets create -p .", ws)
    assert out["permissionDecision"] == "deny" and ".env" in out["permissionDecisionReason"]


def test_record_push_logs_a_kernel_version_once(ws):
    now = G._iso(G._now())
    first = G.record_push(ws, {"kernel": "u/a", "version": 2, "hours": 1.0, "pushed_at": now, "status": "pushed"})
    again = G.record_push(ws, {"kernel": "u/a", "version": 2, "hours": 9.0, "pushed_at": now, "status": "pushed"})
    assert again == first and G.runs(ws) == [first]
    G.record_push(ws, {"kernel": "u/a", "version": 3, "hours": 2.0, "pushed_at": now, "status": "pushed"})
    assert [(r["version"], r["status"]) for r in G.runs(ws)] == [(2, "superseded"), (3, "pushed")]


# ----------------------------------------------------------------------------- #38 pushes from background commands
def _post_bg(plugin_root, cmd, cwd, asked=True, **response):
    """PostToolUse of a Bash call that went on in the background, as Claude Code sends it: the call returns
    before the command ends, with a task id and none of its output, and the session's cwd is still where the
    command started. ``asked``: run_in_background was set (else the call was moved there, e.g. on its timeout)."""
    payload = {"hook_event_name": "PostToolUse", "tool_name": "Bash", "cwd": str(cwd),
               "tool_input": {"command": cmd, **({"run_in_background": True} if asked else {})},
               "tool_response": {"stdout": "", "stderr": "", "interrupted": False, "backgroundTaskId": "b7x2",
                                 **response}}
    r = subprocess.run([sys.executable, str(plugin_root / "hooks" / "post_tool.py")], input=json.dumps(payload),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"] if r.stdout.strip() else None


def test_background_push_is_logged_pending_with_its_cap_reserved(plugin_root, ws):
    base = _two_gpu_kernels(ws)
    _quota(ws, gpu_left=1.5)
    kdir = (base / "vj-gold58").as_posix()
    cmd = (f"for i in $(seq 1 120); do out=$(kaggle kernels push -p {kdir} -t 3600 2>&1 | tail -1); "
           'echo "$out" | grep -q "successfully pushed" && break; sleep 120; done')
    ctx = _post_bg(plugin_root, cmd, ws)
    assert "GPU push of me/rsna-knee-vj-gold58" in ctx and "logged as pending" in ctx
    assert "gpu wait me/rsna-knee-vj-gold58" in ctx and "gpu release me/rsna-knee-vj-gold58" in ctx
    run = G.runs(ws)[-1]
    assert (run["kernel"], run["version"], run["status"], run["hours"]) == ("me/rsna-knee-vj-gold58", None,
                                                                           "pending", 1.0)
    assert G.available_hours(ws) == pytest.approx(0.5)  # the whole cap: the run may not even have started
    assert ("me/rsna-knee-vj-gold58 [pending, cap 1h, via kaggle CLI]: ~1.00h reserved"
            in G.quota_report(ws, refresh=False))
    out = _hook(plugin_root, "pre_tool.py", f"kaggle kernels push -p {(base / 'oai-a18').as_posix()} -t 3600", ws)
    assert out["permissionDecision"] == "ask" and "only 0.50h of GPU quota" in out["permissionDecisionReason"]
    # a push of the same kernel that Kaggle confirms takes the reservation over
    _hook(plugin_root, "post_tool.py", f"kaggle kernels push -p {kdir} -t 3600", ws,
          "Kernel version 3 successfully pushed.  Please check progress at "
          "https://www.kaggle.com/code/me/rsna-knee-vj-gold58")
    assert [(r["version"], r["status"]) for r in G.runs(ws)] == [(None, "superseded"), (3, "pushed")]


@pytest.mark.parametrize("asked, response, ran_s", [(True, {}, 0), (False, {"timedOutAfterMs": 120000}, 120)])
def test_background_push_folder_is_read_from_where_the_command_started(plugin_root, ws, asked, response, ran_s):
    _two_gpu_kernels(ws)
    cmd = "cd experiments/phase4/kernels/oai-a18 && kaggle kernels push -p . -t 1800"
    assert "me/rsna-knee-oai-a18" in _post_bg(plugin_root, cmd, ws, asked=asked, **response)
    run = G.runs(ws)[-1]
    assert (run["kernel"], run["status"], run["hours"]) == ("me/rsna-knee-oai-a18", "pending", 0.5)
    # moved to the background on its timeout, the command had been running that long already
    ago = (G._now() - G._parse_time(run["pushed_at"])).total_seconds()
    assert ran_s <= ago < ran_s + 60


def test_background_push_the_hook_cannot_place_is_reported_not_logged(plugin_root, ws):
    _two_gpu_kernels(ws)
    ctx = _post_bg(plugin_root, "cd $KDIR && kaggle kernels push -p .", ws)
    assert "folder the hook cannot tell" in ctx and "absolute `-p` path" in ctx and G.runs(ws) == []


def _pending(ws, since, kernel="me/k", hours=2.0):
    return G.record_push(ws, {"kernel": kernel, "version": None, "name": kernel.split("/")[-1],
                              "accelerator": "NvidiaTeslaT4", "resource": "GPU", "hours": hours,
                              "pushed_at": G._iso(since), "status": G.PENDING, "via": "kaggle-cli",
                              "background": True})


def _kaggle_latest_run(latest):
    """Kaggle for me/k, whose latest version ran at latest["at"] and has status latest["status"] (moved on by
    the test): `kernels list -m` and `kernels status`."""
    def listing(args):
        return _cp(json.dumps([{"ref": "me/k", "lastRunTime": latest["at"].strftime("%Y-%m-%dT%H:%M:%S.%f")}]))

    def status(args):
        return _cp(f'me/k has status "KernelWorkerStatus.{latest["status"]}"')
    return FakeKaggle([(["kernels", "list"], listing), (["kernels", "status"], status)])


def test_pending_push_takes_no_status_until_kaggle_lists_a_run_after_it(ws, monkeypatch, capsys):
    from kgkit import cli

    _quota(ws, gpu_left=10.0)
    t0 = G._now()
    G.log_run(ws, {"kernel": "me/k", "version": 2, "hours": 1.0, "status": "complete",
                   "pushed_at": G._iso(t0 - dt.timedelta(hours=6))})
    _pending(ws, t0)
    latest = {"at": t0 - dt.timedelta(hours=6), "status": "COMPLETE"}  # Kaggle still shows version 2
    fake = _kaggle_latest_run(latest)
    monkeypatch.setattr(G, "run_kaggle", fake)
    assert G.refresh_runs(ws) == {"me/k": "pending"}
    assert not [c for c in fake.calls if c[:2] == ["kernels", "status"]]  # version 2's status is not the push's
    G.note_status(ws, "me/k", "complete")
    G.update_run(ws, "me/k", status="collected")
    assert [r["status"] for r in G.runs(ws)] == ["collected", "pending"]
    assert G.available_hours(ws) == pytest.approx(8.0)
    monkeypatch.chdir(ws)
    cli.main(["gpu", "status"])
    out = capsys.readouterr().out
    assert "me/k: pending (pushed from a background command at" in out and "its cap stays reserved" in out
    latest.update(at=t0 + dt.timedelta(minutes=4), status="RUNNING")  # the push went through
    cli.main(["gpu", "status"])
    assert "me/k: running  (its push from a background command went through)" in capsys.readouterr().out
    run = G.runs(ws)[-1]
    assert run["status"] == "running" and run["pushed_at"] == G._iso(latest["at"]) and run["started_at"]


def test_wait_waits_for_a_pending_push_then_for_its_run(ws, monkeypatch, capsys):
    t0 = G._now()
    _pending(ws, t0)
    latest = {"at": t0 - dt.timedelta(days=1), "status": "COMPLETE"}  # an earlier version
    monkeypatch.setattr(G, "run_kaggle", _kaggle_latest_run(latest))
    polls = iter([{"at": t0 + dt.timedelta(minutes=1), "status": "RUNNING"}, {"status": "COMPLETE"}])
    monkeypatch.setattr(G, "time", SimpleNamespace(time=G.time.time, sleep=lambda s: latest.update(next(polls))))
    assert G.wait(ws, "me/k") == "complete"
    out = capsys.readouterr().out
    assert "me/k: pending (waiting for its push from a background command" in out
    assert "me/k: running" in out and "me/k: complete" in out and G.runs(ws)[-1]["status"] == "complete"


def test_wait_stops_at_a_pending_push_kaggle_never_ran(ws, monkeypatch, capsys):
    from kgkit import cli

    t0 = G._now() - dt.timedelta(hours=4)  # longer ago than its 2h cap plus the hour a GPU job may queue
    _pending(ws, t0)
    monkeypatch.setattr(G, "run_kaggle", _kaggle_latest_run({"at": t0 - dt.timedelta(days=1), "status": "COMPLETE"}))
    assert G.wait(ws, "me/k") == "pending"
    assert "that push most likely failed" in capsys.readouterr().out
    monkeypatch.chdir(ws)
    cli.main(["gpu", "status"])
    assert "its reservation has lapsed" in capsys.readouterr().out
    assert G.active_runs(ws) == [] and G.release(ws, "me/k") == 1


# ----------------------------------------------------------------------------- #39 several submits in one command
def _logged(ws):
    return [(r["kernel"] or r["file"], r["status"], r["remaining_today"]) for r in S.submission_records(ws)]


def _code_submits(*kernels, sep=" ; "):
    return sep.join(f'kaggle competitions submit demo -k me/{k} -v 1 -f submission.csv -m "000{n}_{k}"'
                    for n, k in enumerate(kernels, 1))


@pytest.mark.parametrize("sep", [" ; ", " && ", "\n"])
def test_each_submit_in_a_command_is_logged_with_its_own_count(plugin_root, ws, sep):
    out = "4 submissions remaining today.\n3 submissions remaining today.\n"
    ctx = _hook(plugin_root, "post_tool.py", _code_submits("a23", "a22", sep=sep), ws, out)["additionalContext"]
    assert _logged(ws) == [("me/a23", "submitted", 4), ("me/a22", "submitted", 3)]  # was: me/a23 only
    assert "submission me/a23 v1 logged (Kaggle reports 4 left today" in ctx and "ledger lb 0001_a23" in ctx
    assert "submission me/a22 v1 logged (Kaggle reports 3 left today" in ctx and "ledger lb 0002_a22" in ctx
    assert S.submissions_today(ws) == 2 and S.server_remaining_today(ws) == 3


def test_file_submits_are_told_apart_though_kaggle_ends_its_message_without_a_newline(plugin_root, ws):
    (ws / "subs").mkdir()
    (ws / "subs" / "a.csv").write_text("id,target\n1,0.1\n")
    (ws / "subs" / "b.csv").write_text("id,target\n1,0.2\n")
    cmd = 'kaggle competitions submit demo -f subs/a.csv -m "0001_a" && kaggle c submit demo -f subs/b.csv -m 0002_b'
    title = "Playground Series - Season 5, Episode 10"  # the next count is printed right after it: "...Episode 103"
    _hook(plugin_root, "post_tool.py", cmd, ws, f"4 submissions remaining today.\nSuccessfully submitted to {title}"
                                                f"3 submissions remaining today.\nSuccessfully submitted to {title}")
    assert _logged(ws) == [("subs/a.csv", "submitted", 4), ("subs/b.csv", "submitted", 3)]
    a, b = S.submission_records(ws)
    assert len(a["sha256"]) == 64 and len(b["sha256"]) == 64 and a["sha256"] != b["sha256"]


def test_a_refused_submit_is_logged_failed_and_the_order_is_flagged(plugin_root, ws):
    out = ("400 Client Error: Bad Request for url: https://api.kaggle.com/v1/competitions.CompetitionApiService/"
           "CreateCodeSubmission\n3 submissions remaining today.\n")
    ctx = _hook(plugin_root, "post_tool.py", _code_submits("k1", "k2"), ws, out)["additionalContext"]
    assert _logged(ws) == [("me/k1", "failed", None), ("me/k2", "submitted", 3)]
    assert "refused the submission me/k1 v1 (400)" in ctx and "submission me/k2 v1 logged" in ctx
    assert "accepted 1 and refused 1" in ctx and "kaggle competitions submissions demo" in ctx
    assert S.submissions_today(ws) == 1


def test_submits_in_text_or_after_a_failed_step_are_neither_checked_nor_logged(plugin_root, ws):
    issue = ("cat > body.md <<'EOF'\nRepro: `kaggle competitions submit <comp> -k u/k1 -v 1 -f missing.csv -m \"a\"`\n"
             "EOF\ngh issue create --title t --body-file body.md")
    commit = ("git add -A && git commit -m \"$(cat <<'EOF'\nHooks: check `kaggle competitions submit demo -f "
              "missing.csv -m x`\nEOF\n)\"")
    for cmd in (issue, commit):
        assert _hook(plugin_root, "pre_tool.py", cmd, ws) is None, cmd  # was denied: missing.csv not found
        assert _hook(plugin_root, "post_tool.py", cmd, ws, "https://github.com/o/r/issues/40\n") is None, cmd
    cmd = "kaggle competitions submit -h | head -2; python make_sub.py && kaggle competitions submit demo -f s.csv -m x"
    out = "usage: kaggle competitions submit [-h]\nTraceback (most recent call last):\nFileNotFoundError: test.csv\n"
    assert _hook(plugin_root, "post_tool.py", cmd, ws, out) is None
    assert S.submission_records(ws) == []  # each of these logged an entry, counted against the budget
    script = "cat > s.sh <<'EOF'\nkaggle competitions submit demo -f missing.csv -m x\nEOF\nbash s.sh"
    assert _hook(plugin_root, "pre_tool.py", script, ws)["permissionDecision"] == "deny"  # a script it then runs


def test_submits_in_a_background_command_are_logged_with_an_unknown_outcome(plugin_root, ws):
    ctx = _post_bg(plugin_root, _code_submits("k1", "k2", sep=" && "), ws)
    assert _logged(ws) == [("me/k1", "unknown", None), ("me/k2", "unknown", None)]
    assert "outcome unknown: Kaggle's reply is not in this output" in ctx and S.submissions_today(ws) == 2


def test_a_submit_in_a_loop_is_logged_once_per_reply(plugin_root, ws):
    cmd = 'for k in k1 k2; do kaggle competitions submit demo -k me/$k -v 1 -f submission.csv -m "$k"; done'
    out = "4 submissions remaining today.\n3 submissions remaining today.\n"
    ctx = _hook(plugin_root, "post_tool.py", cmd, ws, out)["additionalContext"]
    assert _logged(ws) == [("me/$k", "submitted", 4), ("me/$k", "submitted", 3)]
    assert "accepted 1 more submit(s) than this command has" in ctx and S.server_remaining_today(ws) == 3


def test_a_quiet_notebook_submit_takes_no_count_from_the_next_one(plugin_root, ws):
    cmd = _code_submits("k1", "k2").replace("submit demo -k me/k1", "submit demo -q -k me/k1")
    _hook(plugin_root, "post_tool.py", cmd, ws, "3 submissions remaining today.\n")  # -q prints nothing when accepted
    _hook(plugin_root, "post_tool.py", cmd.split(" ; ")[0], ws, "Could not find competition - please verify")
    assert _logged(ws) == [("me/k1", "unknown", None), ("me/k2", "submitted", 3), ("me/k1", "failed", None)]


def test_pre_checks_every_submit_and_counts_them_all_against_the_budget(plugin_root, ws):
    (ws / "data").mkdir()
    (ws / "data" / "sample_submission.csv").write_text("id,target\n1,0.5\n2,0.5\n")
    (ws / "subs").mkdir()
    for name, rows in (("good", "1,0.1\n2,0.9\n"), ("other", "1,0.2\n2,0.8\n"), ("bad", "1,0.1\n")):
        (ws / "subs" / f"{name}.csv").write_text("id,target\n" + rows)
    cmd = 'kaggle competitions submit demo -f subs/good.csv -m 0001_a\nkaggle competitions submit demo -f {} -m 0002_b'
    out = _hook(plugin_root, "pre_tool.py", cmd.format("subs/bad.csv"), ws)
    assert out["permissionDecision"] == "deny"  # was: only the first submit was checked
    assert "- submit 2 (subs/bad.csv): 1 rows but sample_submission has 2" in out["permissionDecisionReason"]
    assert "submit 1" not in out["permissionDecisionReason"]

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    S.submissions_log(ws).write_text((json.dumps({"time_utc": now, "status": "submitted"}) + "\n") * 3)
    assert _hook(plugin_root, "pre_tool.py", cmd.format("subs/other.csv"), ws) is None  # 3 + 2 fits in 5
    with S.submissions_log(ws).open("a") as f:
        f.write(json.dumps({"time_utc": now, "status": "submitted", "remaining_today": 1}) + "\n")
    out = _hook(plugin_root, "pre_tool.py", cmd.format("subs/other.csv"), ws)
    assert out["permissionDecision"] == "ask"
    assert "4/5 submissions already logged today (UTC) and this command submits 2 times" in out[
        "permissionDecisionReason"]
    S.submissions_log(ws).write_text(json.dumps({"time_utc": now, "status": "submitted", "remaining_today": 1}) + "\n")
    out = _hook(plugin_root, "pre_tool.py", cmd.format("subs/other.csv"), ws)
    assert "Kaggle reported 1 submission left today for the team and this command submits 2 times" in out[
        "permissionDecisionReason"]


def test_flags_are_read_up_to_the_end_of_their_line(plugin_root):
    sys.path.insert(0, str(plugin_root / "hooks"))
    import _common as C

    subs = C.parse_submits("kaggle competitions submit c -k u/a -v 1 -m a  # it's a\n"
                           "kaggle competitions submit c -k u/b -v 2 -m b \\\n  -q")
    assert [(s["kernel"], s["version"], s["message"], s["quiet"]) for s in subs] == [("u/a", "1", "a", False),
                                                                                    ("u/b", "2", "b", True)]
    pushes = C.parse_pushes("kaggle kernels push -p A\nkaggle kernels push -p B -t 60")
    assert [(p["path"], p["timeout"]) for p in pushes] == [("A", None), ("B", "60")]  # A took B's flags before
    assert C.parse_submits("kaggle competitions submit --help") == []


# ----------------------------------------------------------------------------- #43 machine shapes, Kaggle's answers
SAVE_400 = "400 Client Error: Bad Request for url: https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel"


def _shaped_kernel(root, shape, **meta):
    k = _notebook_kernel(root)
    m = json.loads((k / "kernel-metadata.json").read_text())
    (k / "kernel-metadata.json").write_text(json.dumps({**m, "machine_shape": shape, **meta}))
    return k


@pytest.mark.parametrize("shape,ask", [("NvidiaTeslaT4Highmem", "NvidiaTeslaT4"), ("TpuV5E8Highmem", "TpuV5E8"),
                                       ("nvidiateslap100", "NvidiaTeslaP100"), ("T4", "NvidiaTeslaT4"),
                                       ("tpu", "TpuV5E8"), ("Bogus", None)])
def test_machine_shape_problem_names_the_shape_to_ask_for(shape, ask):
    problem = G.machine_shape_problem(shape)
    assert f"machine_shape {shape!r}" in problem and "bare 400" in problem and "Tpu1VmV38 (TPU v3-8)" in problem
    assert (f": ask for {ask}." in problem) if ask else ": ask for" not in problem


def test_machine_shapes_kaggle_takes_pass():
    for shape in [*G.MACHINE_SHAPES, "none", "", None]:
        assert G.machine_shape_problem(shape) is None
    assert "stored by Kaggle as TpuV5E8: ask for TpuV5E8" in G.machine_shape_problem("TpuV5E4")


def test_push_refuses_a_machine_shape_kaggle_does_not_take_before_any_kaggle_call(ws, monkeypatch, capsys):
    k = _shaped_kernel(ws, "NvidiaTeslaT4Highmem")
    fake = FakeKaggle(_push_routes())
    monkeypatch.setattr(G, "run_kaggle", fake)
    with pytest.raises(SystemExit, match=r"\(kernel-metadata\.json\): machine_shape 'NvidiaTeslaT4Highmem' .*: ask for "
                                         r"NvidiaTeslaT4\. .* --force pushes it"):
        G.push(ws, k, hours=5.6)
    assert fake.calls == []  # no session, source or quota lookups first: it fails at once
    rec = G.push(ws, k, hours=1, force=True)  # a shape Kaggle may have added since
    push = next(c for c in fake.calls if c[:2] == ["kernels", "push"])
    assert push[push.index("--accelerator") + 1] == "NvidiaTeslaT4Highmem" and rec["version"] == 7
    assert "warning: machine_shape 'NvidiaTeslaT4Highmem'" in capsys.readouterr().out
    k = _shaped_kernel(ws / "tpu", "TpuV5E4", enable_gpu=False, enable_tpu=True)
    with pytest.raises(SystemExit, match="ask for TpuV5E8$"):  # Kaggle runs it as a v5e-8: --force does not help
        G.push(ws, k, hours=1, force=True)


@pytest.mark.parametrize("shape,output,hint", [
    ("NvidiaTeslaT4Highmem", SAVE_400, "Kaggle refused the push: machine_shape 'NvidiaTeslaT4Highmem' "),
    ("NvidiaTeslaT4", f"{SAVE_400}\nKaggle's reason: Invalid docker image (INVALID_ARGUMENT)",
     "Kaggle refused the push; its reason is above."),
    ("NvidiaTeslaT4", SAVE_400, "the Kaggle CLI does not print why (machine_shape NvidiaTeslaT4 is one Kaggle accepts)"),
    ("NvidiaTeslaT4", "Kernel push error: Maximum batch GPU session count of 2 reached", "2 batch GPU sessions")])
def test_a_failed_push_says_why(ws, monkeypatch, shape, output, hint):
    k = _shaped_kernel(ws, shape)
    rc = 0 if output.startswith("Kernel push error") else 1  # the CLI exits 0 after that one
    monkeypatch.setattr(G, "run_kaggle", FakeKaggle([(["kernels", "push"], _cp("", rc, output)), *_push_routes()]))
    with pytest.raises(SystemExit) as e:
        G.push(ws, k, hours=1, force=True)
    assert str(e.value).startswith("kernel push failed (see output above); nothing was started.")
    assert hint in str(e.value) and G.runs(ws) == []


def test_push_says_which_inputs_kaggle_left_out(ws, monkeypatch, capsys):
    from kgkit import cli

    k = _notebook_kernel(ws)
    out = ("The following are not valid dataset sources and could not be added to the kernel: ['me/knee-png-5l2']\n"
           "Kernel version 7 successfully pushed.  Please check progress at https://www.kaggle.com/code/me/r6-gold58")
    monkeypatch.setattr(G, "run_kaggle", FakeKaggle([(["kernels", "push"], _cp(out)), *_push_routes()]))
    monkeypatch.chdir(ws)
    cli.main(["gpu", "push", str(k), "--hours", "1"])
    text = capsys.readouterr().out
    assert ("WARNING: Kaggle accepted the push but left out inputs it could not attach (me/r6-gold58: dataset "
            "['me/knee-png-5l2']), so the run starts without them.") in text and "gpu cancel me/r6-gold58" in text
    assert "It runs without some of its inputs: see the WARNING above." in text and "Next:" not in text
    run = G.runs(ws)[-1]  # it runs all the same: logged, its cap reserved
    assert run["version"] == 7 and run["status"] == "pushed" and run["dropped_sources"] == ["dataset ['me/knee-png-5l2']"]


def test_machine_shape_counts_without_the_enable_flags(ws, monkeypatch, plugin_root):
    assert G.kernel_meta_accelerator({"machine_shape": "TpuV5E8"}) == "TpuV5E8"  # was 'none': no TPU check or log
    assert G.kernel_meta_accelerator({"machine_shape": "None"}) == "none"
    assert G.kernel_meta_accelerator({"enable_gpu": True, "machine_shape": "None"}) == "NvidiaTeslaT4"
    assert G.kernel_meta_accelerator({"enable_tpu": True, "machine_shape": "NvidiaTeslaT4"}) == "TpuV5E8"
    sys.path.insert(0, str(plugin_root / "hooks"))
    import _common as C

    assert C.push_resource({}, {"machine_shape": "TpuV5E8"}) == "TPU"
    # an unknown shape and no flags: checked, not taken for a CPU kernel (the CLI would send that shape all the same)
    k = _shaped_kernel(ws, "Bogus", enable_gpu=False)
    monkeypatch.setattr(G, "run_kaggle", FakeKaggle(_push_routes()))
    with pytest.raises(SystemExit, match="machine_shape 'Bogus'"):
        G.push(ws, k, hours=1)


def test_build_names_the_accelerator_to_ask_for(ws):
    with pytest.raises(SystemExit, match="ask for NvidiaTeslaT4"):
        G.build_kernel(ws, "x", "true", hours=1, user="me", accelerator="NvidiaTeslaT4Highmem")
    with pytest.raises(SystemExit, match="`gpu build` targets NvidiaTeslaT4, NvidiaTeslaP100, TpuV5E8, TpuV6E8 or none"):
        G.build_kernel(ws, "x", "true", hours=1, user="me", accelerator="Tpu1VmV38")  # Kaggle takes it; the runner not


def _fake_kaggle_cli(root):
    """A kaggle package whose CLI fails the way Kaggle CLI 2.2.4 does on a refused request: kagglesdk calls
    requests' raise_for_status on Kaggle's error response (its first argument here), and the CLI's main prints it."""
    (root / "kaggle").mkdir()
    (root / "kaggle" / "__init__.py").write_text("")
    (root / "kaggle" / "cli.py").write_text(
        "import sys\nimport requests\n\n"
        "def main():\n"
        "    print('kgkit on the path:', [p for p in sys.path if p.rstrip('/\\\\').endswith('kgkit')])\n"
        "    r = requests.Response()\n"
        "    r.status_code, r.reason, r._content = 400, 'Bad Request', sys.argv[1].encode()\n"
        "    r.url = 'https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel'\n"
        "    try:\n"
        "        r.raise_for_status()\n"
        "    except requests.HTTPError as e:\n"
        "        print(e, file=sys.stderr)\n"
        "        sys.exit(1)\n")


@pytest.mark.parametrize("body,reason", [
    ('{"error": {"code": 400, "message": "Invalid machine shape", "status": "INVALID_ARGUMENT"}}',
     "Invalid machine shape (INVALID_ARGUMENT)"),
    ('{"code": 9, "message": "FAILED_PRECONDITION: Your Notebook uses non-versioned datasets"}',
     "FAILED_PRECONDITION: Your Notebook uses non-versioned datasets"),
    ("<html>Bad Request</html>", None)])
def test_kaggle_cli_prints_kaggles_reason(tmp_path, plugin_root, body, reason):
    pytest.importorskip("requests")
    _fake_kaggle_cli(tmp_path)
    r = subprocess.run([sys.executable, str(plugin_root / "kgkit" / "kaggle_cli.py"), body], capture_output=True,
                       text=True, timeout=60, env={**os.environ, "PYTHONPATH": str(tmp_path)})
    assert r.returncode == 1 and SAVE_400 in r.stderr
    assert (f"\nKaggle's reason: {reason}" in r.stderr) if reason else "Kaggle's reason" not in r.stderr
    assert "kgkit on the path: []" in r.stdout  # kgkit's modules (cli, datasets, ...) cannot shadow the CLI's imports


def test_kaggle_cli_imports_only_the_standard_library_at_import_time(plugin_root):
    """It runs under the Kaggle CLI's interpreter, and the hooks import it through kgkit.gpu."""
    import ast

    tree = ast.parse((plugin_root / "kgkit" / "kaggle_cli.py").read_text(encoding="utf-8"))
    mods = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.level == 0}
    assert mods <= set(sys.stdlib_module_names) | {"__future__"}, mods


def test_run_kaggle_falls_back_to_the_plain_cli(tmp_path, monkeypatch):
    (tmp_path / "kaggle").mkdir()
    (tmp_path / "kaggle" / "__init__.py").write_text("raise ImportError('no kaggle here')\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(G, "kaggle_python", lambda: sys.executable)
    monkeypatch.setattr(G, "kaggle_cmd", lambda: [sys.executable, "-c", "import sys; print('plain', sys.argv[1:])"])
    monkeypatch.setattr(G, "_PLAIN", set())
    assert G.run_kaggle(["quota", "--format", "json"]).stdout.strip() == "plain ['quota', '--format', 'json']"
    assert G._PLAIN == {sys.executable}  # not tried again in this process
    monkeypatch.setattr(G, "_PLAIN", set())
    monkeypatch.setenv("KGKIT_KAGGLE", "my-kaggle")  # the user's own command: used as given
    monkeypatch.setattr(G, "kaggle_python", lambda: pytest.fail("the launcher was tried"))
    assert G.run_kaggle(["quota"]).stdout.strip() == "plain ['quota']"


def test_datasets_pull_keeps_kaggles_reason():
    url = "404 Client Error: Not Found for url: https://api.kaggle.com/v1/datasets.DatasetApiService/DownloadDataset"
    assert DS._error_line(f"Dataset URL: x\n{url}\nKaggle's reason: Not found (NOT_FOUND)\n") == (
        f"{url} (Kaggle's reason: Not found (NOT_FOUND))")
    assert DS._error_line(f"Dataset URL: x\n{url}\n") == url


def test_kaggle_python_reads_the_cli_script(tmp_path, monkeypatch):
    py, cli = sys.executable, tmp_path / "kaggle"
    which = {"kaggle": None, "python3": py}
    monkeypatch.setattr(G.shutil, "which", lambda name: which.get(name))
    assert G.kaggle_python() == py  # not on PATH: the CLI runs as `python -m kaggle`
    which["kaggle"] = str(cli)
    for script, want in [(f"#!{py}\nfrom kaggle.cli import main\n", py),
                         ("#!/usr/bin/env python3\n", py),
                         (f"#!/bin/sh\n'''exec' \"{py}\" \"$0\" \"$@\"\n' '''\n", py),  # pip, for long paths
                         ("#!/bin/sh\nexec python3 -m kaggle \"$@\"\n", None),  # a shell must never get the .py file
                         ("MZ\x90\x00 a Windows .exe launcher", None)]:
        cli.write_text(script, encoding="utf-8")  # Windows' cp1252 cannot write the .exe header's \x90
        assert G.kaggle_python() == want, script


def test_post_hook_explains_a_refused_push(plugin_root, ws):
    _shaped_kernel(ws, "NvidiaTeslaT4Highmem")
    cmd = "kaggle kernels push -p kernels/r6-gold58 -t 3600"
    ctx = _hook(plugin_root, "post_tool.py", cmd, ws, stdout=SAVE_400)["additionalContext"]
    assert "Kaggle refused the kernel push (a 4xx from SaveKernel), so nothing was started." in ctx
    assert "me/r6-gold58: machine_shape 'NvidiaTeslaT4Highmem'" in ctx and ": ask for NvidiaTeslaT4." in ctx
    assert G.runs(ws) == []
    # --accelerator is what the CLI sends: an accepted one rules the folder's shape out
    ctx = _hook(plugin_root, "post_tool.py", cmd + " --accelerator NvidiaTeslaT4", ws, stdout=SAVE_400)["additionalContext"]
    assert "Highmem" not in ctx and "`python -m kgkit gpu push <dir> --hours H`, which shows it" in ctx


def test_post_hook_names_inputs_kaggle_left_out(plugin_root, ws):
    _notebook_kernel(ws)
    out = ("The following are not valid kernel sources and could not be added to the kernel: ['me/s6t-r1-trian']\n"
           "Kernel version 3 successfully pushed.  Please check progress at https://www.kaggle.com/code/me/r6-gold58\n")
    ctx = _hook(plugin_root, "post_tool.py", "kaggle kernels push -p kernels/r6-gold58 -t 3600", ws,
                stdout=out)["additionalContext"]
    assert "left out inputs it could not attach (me/r6-gold58: kernel ['me/s6t-r1-trian'])" in ctx
    assert "me/r6-gold58 v3 logged" in ctx  # it runs: its cap is reserved


def test_gpu_push_end_to_end_prints_the_cause_after_the_output(ws, tmp_path, plugin_root):
    """The issue's run: a refused push used to end with 'see output above' printed above the output, and no cause."""
    k = _shaped_kernel(ws, "NvidiaTeslaT4Highmem")
    calls = tmp_path / "calls.txt"
    fake = tmp_path / "fake_kaggle.py"
    fake.write_text("import sys\n"
                    f"open({str(calls)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
                    "if sys.argv[1:3] == ['kernels', 'push']:\n"
                    f"    print({SAVE_400!r}, file=sys.stderr)\n"
                    "sys.exit(1)\n")
    import shlex

    cli = [sys.executable, str(fake)]
    if os.name == "nt" and any(" " in a for a in cli):
        pytest.skip("KGKIT_KAGGLE is split without quote handling on Windows")
    env = {**os.environ, "PYTHONPATH": str(plugin_root),
           "KGKIT_KAGGLE": " ".join(cli) if os.name == "nt" else shlex.join(cli)}
    run = [sys.executable, "-m", "kgkit", "gpu", "push", str(k), "--hours", "5.6", "--no-session-check"]
    r = subprocess.run(run, cwd=ws, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert r.returncode == 1 and "ask for NvidiaTeslaT4" in r.stdout and not calls.exists()  # refused at once
    r = subprocess.run([*run, "--force"], cwd=ws, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=120)
    out = r.stdout
    assert r.returncode == 1 and "kernels push" in calls.read_text()
    assert out.index("warning: machine_shape") < out.index(SAVE_400) < out.index("kernel push failed (see output above)")
    assert "Kaggle refused the push: machine_shape 'NvidiaTeslaT4Highmem'" in out


# ----------------------------------------------------------------------------- #44 inputs of a remote run
SHOW_INPUTS = ("import os, sys\n"
               "sys.stdout.reconfigure(encoding='utf-8')\n"  # the runner's own log must be ASCII, not this one's
               "for d, _, files in sorted(os.walk('data', followlinks=True)):\n"
               "    for f in sorted(files):\n"
               "        p = os.path.join(d, f)\n"
               "        print('SEE', p.replace(os.sep, '/'), open(p).read().strip().replace(chr(10), '|'))\n")


def _inputs_ws(tmp_path):
    root = tmp_path / "ws"
    (root / "src").mkdir(parents=True)
    S.CompetitionState(slug="demo-comp", metric="auc").save(root)
    (root / "data").mkdir()
    (root / "data" / "folds.csv").write_text("id,fold\n0,0\n1,1\n")
    (root / "src" / "show.py").write_text(SHOW_INPUTS)
    return root


def _run_inputs(root, base, mounted, **build):
    """Build a CPU kernel whose command prints every file under data/, mount ``mounted`` ({path under
    /kaggle/input: text}) and run it: (stdout, {file under data/: text}, kg_run.json)."""
    kdir = G.build_kernel(root, "show", f'"{sys.executable}" src/show.py', hours=1, user="me", accelerator="none",
                          **build)
    return _run_built(kdir, base, mounted)


def _run_built(kdir, base, mounted):
    for rel, text in mounted.items():
        (base / "input" / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / "input" / rel).write_text(text)
    meta = json.loads((kdir / "kernel-metadata.json").read_text())
    env = dict(os.environ, KAGGLE_INPUT=str(base / "input"), KAGGLE_WORKING=str(base / "working"),
               KG_RUN_DIR=str(base / "run"), KG_FAKE_GPUS="0")
    env.pop("PYTHONPATH", None)
    r = subprocess.run([sys.executable, str(kdir / meta["code_file"])], env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    seen = dict(l[len("[train] SEE "):].split(" ", 1) for l in r.stdout.splitlines() if l.startswith("[train] SEE "))
    return r.stdout, seen, json.loads((base / "working" / "kg_run.json").read_text())


OOF_LIBRARIES = {
    "competitions/demo-comp/train.csv": "competition train",
    "datasets/megayak/s6e10-oof/oof_predictions.parquet": "megayak oof",
    "datasets/megayak/s6e10-oof/test_predictions.parquet": "megayak test",
    "datasets/mitudru/s6e10-oof/oof_predictions.parquet": "mitudru oof",  # another owner, the same slug
    "datasets/mitudru/s6e10-oof/test_predictions.parquet": "mitudru test",
    "datasets/thisray/s6e10-stack/oof_predictions.parquet": "thisray oof",
    "datasets/thisray/s6e10-stack/folds.csv": "thisray's split",
    "s6e10-stack/oof_members.csv": "kernel output of kagankoral",  # the layout of older Kaggle images
}


def test_flat_inputs_warn_about_every_name_two_inputs_share(tmp_path, capsys):
    root = _inputs_ws(tmp_path)
    mounted = {**OOF_LIBRARIES, "datasets/megayak/s6e10-oof/notes-\u0111.txt": "megayak notes",
               "datasets/mitudru/s6e10-oof/notes-\u0111.txt": "mitudru notes"}  # not in cp1252 (Windows consoles)
    out, seen, rep = _run_inputs(root, tmp_path / "k", mounted, resume_from=["me/old-train"],
                                 datasets=["megayak/s6e10-oof", "mitudru/s6e10-oof", "thisray/s6e10-stack",
                                           "me/not-attached"], kernels=["kagankoral/s6e10-stack"])
    note = capsys.readouterr().out  # said at build time
    assert "5 inputs are linked side by side into data/ (megayak/s6e10-oof, mitudru/s6e10-oof, thisray/s6e10-stack, " \
           "me/not-attached and 1 more): a name two of them share is linked from the first only" in note
    # the first input keeps a name; every other input that has it is named, with what holds it
    assert seen["data/oof_predictions.parquet"] == "megayak oof" and seen["data/train.csv"] == "competition train"
    for line in ["data/oof_predictions.parquet from mitudru/s6e10-oof skipped: already linked from megayak/s6e10-oof",
                 "data/test_predictions.parquet from mitudru/s6e10-oof skipped: already linked from megayak/s6e10-oof",
                 "data/oof_predictions.parquet from thisray/s6e10-stack skipped: already linked from megayak/s6e10-oof",
                 "data/folds.csv from thisray/s6e10-stack skipped: already in the code bundle",
                 "data/notes-?.txt from mitudru/s6e10-oof skipped: already linked from megayak/s6e10-oof"]:
        assert "WARNING: " + line in out
    assert seen["data/folds.csv"] == "id,fold|0,0|1,1"  # the frozen split, not the library's
    # a slug two owners share is read from each owner's folder, not twice from the first one found
    rows = {r["source"]: r for r in rep["inputs"]}
    assert rows["mitudru/s6e10-oof"]["path"].replace("\\", "/").endswith("datasets/mitudru/s6e10-oof")
    assert rows["mitudru/s6e10-oof"]["linked"] == 0 and len(rows["mitudru/s6e10-oof"]["skipped"]) == 3
    assert seen["data/oof_members.csv"] == "kernel output of kagankoral"
    assert rows["kagankoral/s6e10-stack"]["kind"] == "kernel"
    # inputs that are not there are said, the resume source too (its folds would be trained again)
    assert "WARNING: input me/not-attached not found under" in out and "Kaggle could not attach" in out
    assert "WARNING: resume source me/old-train not found under" in out
    assert "5 input entries were not linked" in out.splitlines()[-2]  # repeated at the end of the log
    assert out.isascii() and seen["data/notes-?.txt"] == "megayak notes"
    res = G.collect(root, "me/show-train", from_dir=tmp_path / "k" / "working")
    assert "inputs the run did not have: me/not-attached (not found under " in res["summary"]
    assert "me/old-train (resume source not found under " in res["summary"]
    assert ("5 input entries were not linked, as the name was already taken: data/notes-\u0111.txt "
            "from mitudru/s6e10-oof (already linked from megayak/s6e10-oof); data/oof_predictions.parquet "
            "from mitudru/s6e10-oof (already linked from megayak/s6e10-oof);") in res["summary"]
    assert "`gpu build --input-layout nested`" in res["summary"]


def test_nested_inputs_keep_every_source_in_its_own_folder(tmp_path):
    root = _inputs_ws(tmp_path)
    mounted = {**OOF_LIBRARIES, "datasets/dariushafshar/s6e10-golem-oof-library/oof.npy": "golem",
               "datasets/me/prep/train_512.csv": "preprocessed"}
    out, seen, rep = _run_inputs(root, tmp_path / "k", mounted, input_layout="nested",
                                 datasets=["megayak/s6e10-oof", "mitudru/s6e10-oof=data/ext/mitudru-s6e10-oof",
                                           "thisray/s6e10-stack",
                                           "dariushafshar/s6e10-golem-oof-library=data/ext/golem", "me/prep=data"],
                                 kernels=["kagankoral/s6e10-stack"])
    meta = json.loads((root / "kernels" / "show-train" / "kernel-metadata.json").read_text())
    assert meta["dataset_sources"] == ["megayak/s6e10-oof", "mitudru/s6e10-oof", "thisray/s6e10-stack",
                                       "dariushafshar/s6e10-golem-oof-library", "me/prep"]  # Kaggle gets refs only
    assert meta["kernel_sources"] == ["kagankoral/s6e10-stack"]
    assert "WARNING" not in out and rep["input_layout"] == "nested"
    assert seen == {
        "data/folds.csv": "id,fold|0,0|1,1", "data/train.csv": "competition train",
        "data/train_512.csv": "preprocessed",  # =data: this one beside the competition data
        "data/ext/s6e10-oof/oof_predictions.parquet": "megayak oof",
        "data/ext/s6e10-oof/test_predictions.parquet": "megayak test",
        "data/ext/mitudru-s6e10-oof/oof_predictions.parquet": "mitudru oof",
        "data/ext/mitudru-s6e10-oof/test_predictions.parquet": "mitudru test",
        "data/ext/s6e10-stack/oof_predictions.parquet": "thisray oof",
        "data/ext/s6e10-stack/folds.csv": "thisray's split",
        "data/ext/kout-s6e10-stack/oof_members.csv": "kernel output of kagankoral",
        "data/ext/golem/oof.npy": "golem"}
    assert G.diagnose(rep) == []


def test_an_input_is_not_linked_into_a_folder_another_input_holds(tmp_path):
    root = _inputs_ws(tmp_path)
    mounted = {"competitions/demo-comp/train.csv": "competition train", "datasets/me/prep/ext/notes.txt": "notes",
               "datasets/megayak/s6e10-oof/oof_predictions.parquet": "megayak oof"}
    out, seen, rep = _run_inputs(root, tmp_path / "k", mounted, input_layout="nested",
                                 datasets=["me/prep=data", "megayak/s6e10-oof"])
    assert "WARNING: input megayak/s6e10-oof not linked: data/ext is already linked from me/prep" in out
    assert seen["data/ext/notes.txt"] == "notes" and not (tmp_path / "k" / "input" / "datasets" / "me" / "prep" / "ext"
                                                           / "s6e10-oof").exists()  # nothing written into an input
    assert "megayak/s6e10-oof (data/ext is already linked from me/prep)" in G.diagnose(rep)[0]


def _runner_module(plugin_root, inp):
    import importlib.util

    spec = importlib.util.spec_from_file_location("kg_runner_under_test",
                                                  plugin_root / "kgkit" / "kaggle_gpu_runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.INPUT = inp
    return mod


def test_each_input_is_found_in_the_folder_of_its_owner_and_kind(tmp_path, plugin_root):
    inp = tmp_path / "input"
    for d in ["competitions/demo/train/lib", "datasets/a/demo", "datasets/a/lib", "datasets/b/lib", "datasets/abc/abc",
              "notebooks/a/lib", "old-style"]:
        (inp / d).mkdir(parents=True)
    R = _runner_module(plugin_root, inp)
    refs = {"demo": "competition", "a/demo": "dataset", "a/lib": "dataset", "b/lib": "dataset", "abc/abc": "dataset",
            "a/lib ": "kernel", "x/old-style": "dataset", "x/missing": "dataset"}
    found = R.scan_inputs([r.strip() for r in refs])
    got = {f"{r.strip()} {k}": R.find_source(found, r.strip(), k) for r, k in refs.items()}
    rel = {k: v.relative_to(inp).as_posix() if v else None for k, v in got.items()}
    assert rel == {"demo competition": "competitions/demo", "a/demo dataset": "datasets/a/demo",
                   "a/lib dataset": "datasets/a/lib", "b/lib dataset": "datasets/b/lib",
                   "abc/abc dataset": "datasets/abc/abc",  # an owner folder named like the slug is walked into
                   "a/lib kernel": "notebooks/a/lib", "x/old-style dataset": "old-style", "x/missing dataset": None}
    assert all("train" not in str(p) for p, _ in found["lib"])  # the walk stays out of an input's own folders


def test_build_places_each_input_and_refuses_folders_outside_data(ws, monkeypatch, capsys):
    from kgkit import cli

    srcs = G.input_sources(["a/lib", "b/lib=data\\ext\\b-lib", "c/x=data"], ["a/lib", "d/run"], "nested")
    assert [(s["ref"], s["kind"], s["dest"]) for s in srcs] == [
        ("a/lib", "dataset", "data/ext/lib"), ("b/lib", "dataset", "data/ext/b-lib"), ("c/x", "dataset", "data"),
        ("a/lib", "kernel", "data/ext/kout-lib"), ("d/run", "kernel", "data/ext/kout-run")]
    assert [s["dest"] for s in G.input_sources(["a/lib", "a/lib"], [], "flat")] == ["data"]
    with pytest.raises(SystemExit, match=r"a/lib and b/lib would share the folder data/ext/lib/: give b/lib its own, "
                                         r"e\.g\. --dataset b/lib=data/ext/lib-b"):
        G.input_sources(["a/lib", "b/lib"], [], "nested")
    with pytest.raises(SystemExit, match="would share the folder data/ext/lib/"):  # also with a folder given to one
        G.input_sources(["a/lib=data/ext/lib", "b/lib"], [], "nested")
    G.input_sources(["a/lib=data/ext", "b/lib=data/ext"], [], "nested")  # given on purpose: one folder for both
    for bad in ["a/lib=../lib", "a/lib=/abs/lib", "a/lib=src/lib", "a/lib=artifacts/lib", "a/lib=", "a/lib=data/../x"]:
        with pytest.raises(SystemExit, match="the folder must be data or one under it"):
            G.input_sources([bad], [], "flat")
    with pytest.raises(SystemExit, match="expected <owner>/<slug>"):
        G.input_sources(["lib"], [], "flat")
    with pytest.raises(SystemExit, match="--input-layout tree"):
        G.input_sources([], [], "tree")
    monkeypatch.chdir(ws)
    cli.main(["gpu", "build", "--name", "stk", "--cmd", "python src/stack.py", "--hours", "1", "--user", "me",
              "--include", "--accelerator", "none", "--input-layout", "nested", "--dataset", "a/lib", "b/x=data",
              "--kernel", "a/lib"])
    text = capsys.readouterr().out
    assert "inputs: data/ <- demo, b/x; data/ext/lib/ <- a/lib; data/ext/kout-lib/ <- a/lib" in text
    assert "side by side" not in text
    tm = json.loads((ws / "kernels" / "stk-train" / "kg-train.json").read_text())
    assert tm["input_layout"] == "nested" and tm["inputs"][0] == {"ref": "demo", "kind": "competition", "dest": "data"}


# ----------------------------------------------------------------------------- #45 vendored kgkit
def _kgkit(cwd, home, *args, **env_extra):
    """``python -m kgkit`` as a session's Bash command runs it with ``PYTHONPATH=<home>``: the plugin, or a folder
    `kgkit vendor` copied kgkit into."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "KGKIT_HOME")}
    return subprocess.run([sys.executable, "-m", "kgkit", *args], cwd=cwd, env={**env, "PYTHONPATH": str(home),
                          **env_extra}, capture_output=True, text=True, timeout=300)


def test_a_vendored_kgkit_builds_a_kernel_with_its_own_runner(tmp_path, plugin_root):
    root = _inputs_ws(tmp_path)
    r = _kgkit(root, plugin_root, "vendor", "src")
    assert r.returncode == 0 and "copied kgkit " in r.stdout, r.stdout + r.stderr
    with (root / "src" / "kgkit" / "kaggle_gpu_runner.py").open("a") as f:
        f.write("# the vendored copy's runner\n")
    r = _kgkit(root, root / "src", "gpu", "build", "--name", "show", "--cmd", f'"{sys.executable}" src/show.py',
               "--hours", "1", "--user", "me", "--accelerator", "none")
    assert r.returncode == 0, r.stdout + r.stderr
    kdir = root / "kernels" / "show-train"
    assert "# the vendored copy's runner" in (kdir / "show_train.py").read_text(encoding="utf-8")
    out, seen, rep = _run_built(kdir, tmp_path / "k", {"competitions/demo-comp/train.csv": "competition train"})
    assert seen == {"data/folds.csv": "id,fold|0,0|1,1", "data/train.csv": "competition train"}
    assert rep["inputs"][0]["source"] == "demo-comp" and rep["inputs"][0]["linked"] == 1


def test_vendor_never_removes_the_kgkit_it_runs_from(tmp_path, plugin_root):
    _kgkit(tmp_path, plugin_root, "vendor", "src")
    r = _kgkit(tmp_path, tmp_path / "src", "vendor", "src")
    assert r.returncode == 1 and "is the kgkit running this command" in r.stderr
    assert "python -m kgkit vendor src" in r.stderr and (tmp_path / "src" / "kgkit" / "gpu.py").is_file()
    r = _kgkit(tmp_path, tmp_path / "src", "vendor", "lib")  # a copy of the copy: nothing is removed
    assert r.returncode == 0 and (tmp_path / "lib" / "kgkit" / "kaggle_gpu_runner.py").is_file()


def test_init_from_a_vendored_kgkit_takes_the_template_from_the_plugin_or_says_why_not(tmp_path, plugin_root):
    _kgkit(tmp_path, plugin_root, "vendor", "lib")
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    r = _kgkit(a, tmp_path / "lib", "init", "demo", "--metric", "auc")
    assert r.returncode == 0 and "CLAUDE.md not written" in r.stdout and not (a / "CLAUDE.md").exists()
    r = _kgkit(b, tmp_path / "lib", "init", "demo", "--metric", "auc", KGKIT_HOME=str(plugin_root))
    assert r.returncode == 0 and "not written" not in r.stdout
    assert "# Kaggle competition: demo" in (b / "CLAUDE.md").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- #46 fold-split fingerprint
from kgkit import evalset as EV  # noqa: E402


def _old_hash(data: bytes, suffix: bytes = b"") -> str:
    """The fingerprint kgkit 1.7.1 took: sha1 of the file's bytes (then '|' and the columns of an eval spec)."""
    import hashlib

    return hashlib.sha1(data + suffix).hexdigest()[:12]


def _split_files(folds) -> dict[str, bytes]:
    """One split as different tools and platforms write it."""
    df = pd.DataFrame({"id": np.arange(len(folds)), "fold": folds})
    rows = list(zip(df["id"], df["fold"]))
    return {
        "pandas on Linux": df.to_csv(index=False, lineterminator="\n").encode(),
        "pandas on Windows": df.to_csv(index=False, lineterminator="\r\n").encode(),
        "no newline at the end": df.to_csv(index=False, lineterminator="\n").rstrip("\n").encode(),
        "quoted": ('"id","fold"\r\n' + "".join(f'"{i}","{f}"\r\n' for i, f in rows)).encode(),
        "BOM, columns reordered, one more": ("\ufefffold,id,note\n" + "".join(f"{f},{i},x\n" for i, f in rows)).encode(),
        "folds written as floats": ("id,fold\n" + "".join(f"{i},{f}.0\n" for i, f in rows)).encode(),
        "no id column": ("fold\n" + "".join(f"{f}\n" for _, f in rows)).encode(),
        "blank lines": ("id,fold\n\n" + "".join(f"{i},{f}\n" for i, f in rows) + "\n\n").encode(),
    }


def test_folds_hash_is_of_the_split_not_of_the_file_bytes(tmp_path):
    folds = np.arange(70_000) % 4  # more rows than one batch of the reader
    (tmp_path / "data").mkdir()
    f = tmp_path / "data" / "folds.csv"
    want = Ledger(tmp_path)._folds_hash(folds)  # what Ledger.log(folds=...) records for the array
    for what, data in _split_files(folds).items():
        f.write_bytes(data)
        assert EV.folds_file_hash(tmp_path) == want, what
        assert EV.fold_ids(data) == [0, 1, 2, 3], what
    moved = folds.copy()
    moved[0] = 1
    f.write_bytes(_split_files(moved)["pandas on Linux"])
    assert EV.folds_file_hash(tmp_path) != want  # one row in another fold: another split
    f.write_bytes(_split_files(folds[::-1])["no id column"])
    assert EV.folds_file_hash(tmp_path) != want  # without ids, the row order is the assignment
    f.write_bytes(b"id,split\r\n0,a\r\n")  # not a split kgkit can read: still one fingerprint for both line endings
    other = EV.folds_file_hash(tmp_path)
    f.write_bytes(b"id,split\n0,a\n")
    assert EV.folds_file_hash(tmp_path) == other and other != want
    with pytest.raises(ValueError, match="no 'fold' column"):
        EV.fold_ids(b"id,split\n0,a\n")
    with pytest.raises(ValueError, match="fold 'x' is not an integer"):
        EV.fold_ids(b"id,fold\n0,1\n1,x\n")


def test_records_from_older_kgkit_stay_on_their_split_after_a_move_to_linux(ws, monkeypatch, capsys):
    """The issue: records logged on Windows carry sha1 of the CRLF bytes of data/folds.csv; on Linux the same split
    was regenerated with LF."""
    from kgkit import cli

    monkeypatch.chdir(ws)
    rng = np.random.default_rng(0)
    folds = np.arange(60) % 5
    y = (rng.random(60) < 0.5).astype(int)
    (ws / "data").mkdir()
    pd.DataFrame({"id": np.arange(60), "target": y}).to_csv(ws / "data" / "train.csv", index=False)
    linux, windows = _split_files(folds)["pandas on Linux"], _split_files(folds)["pandas on Windows"]
    (ws / "data" / "folds.csv").write_bytes(linux)
    led = Ledger(ws)
    win = led.log("win", 0.7, fold_scores=[0.7] * 5, oof=rng.random(60), decision="keep")
    gone = led.log("gone", 0.7, fold_scores=[0.7] * 5, oof=rng.random(60))
    lp = ws / ".kaggle-gm" / "ledger.jsonl"
    old = {win["id"]: _old_hash(windows), gone["id"]: _old_hash(b"id,fold\r\n0,1\r\n")}  # a split no longer on disk
    lp.write_text("".join(json.dumps({**r, "folds_hash": old[r["id"]]}) + "\n" for r in map(json.loads, lp.read_text(
        encoding="utf-8").splitlines())), encoding="utf-8")
    EV.log_comparison(ws, {"new": win["id"], "base": "0000", "eval": _old_hash(windows), "question": "paired",
                           "gain": 0.01, "se": 0.004, "decision": "keep", "method": "bootstrap"})

    new = Ledger(ws).log("linux", 0.71, fold_scores=[0.71] * 5, oof=rng.random(60))
    led = Ledger(ws)
    assert new["folds_hash"] == EV.folds_file_hash(ws) == led.get(win["id"])["folds_hash"]
    assert led.get(gone["id"])["folds_hash"] == old[gone["id"]]  # another split stays another split
    assert led.decisions_on_evalset() == 1
    assert EV.recent_noise(ws, led.eval_key())["se"] == 0.004
    cli.main(["ledger", "compare", new["id"], win["id"], "--truth", "data/train.csv:target", "--boot", "50"])
    assert "different fold splits" not in capsys.readouterr().out
    cli.main(["ledger", "compare", new["id"], gone["id"], "--truth", "data/train.csv:target", "--boot", "50"])
    assert "different fold splits" in capsys.readouterr().out
    cli.main(["ledger", "decide", new["id"], "discard", "no gain"])  # a rewrite of the ledger stores the upgrade
    on_disk = {r["id"]: r["folds_hash"] for r in map(json.loads, lp.read_text(encoding="utf-8").splitlines())}
    assert on_disk[win["id"]] == new["folds_hash"] and on_disk[gone["id"]] == old[gone["id"]]


def test_eval_hash_is_of_the_labels_not_of_the_file_bytes(tmp_path):
    st = S.CompetitionState(slug="x", validation="holdout+lb", eval_truth="h.csv:y,z")
    rows = [(f"s{i}", i % 2, i % 3) for i in range(30)]
    lf = "id,y,z\n" + "".join(f"{i},{y},{z}\n" for i, y, z in rows)

    def key(text: str) -> str:
        (tmp_path / "h.csv").write_bytes(text.encode())
        return EV.current_key(tmp_path, st)

    want = key(lf)
    assert key(lf.replace("\n", "\r\n")) == want
    assert key('"id","y","z"\n' + "".join(f'"{i}","{y}","{z}"\n' for i, y, z in rows)) == want
    assert key("z,note,y,id\n" + "".join(f"{z},x,{y},{i}\n" for i, y, z in rows)) == want
    assert key("id,y,z\n" + "".join(f"{i},{y}.0,{z}\n" for i, y, z in rows)) == want
    assert key(lf.replace("s0,0,0", "s0,1,0")) != want  # one label changed
    assert key("id,y,z\n" + "".join(f"{i},{z},{y}\n" for i, y, z in rows)) != want  # y and z swapped
    key(lf)
    old = _old_hash(lf.replace("\n", "\r\n").encode(), b"|y,z")  # 1.7.1, holdout file written on Windows
    assert EV.upgrades(tmp_path, [old, "zz"], st) == {old: want}
    (tmp_path / "h.parquet").write_bytes(b"PAR1 binary")  # no line endings: still the bytes
    assert EV.spec_hash(tmp_path, "h.parquet:y") == _old_hash(b"PAR1 binary", b"|y")


def test_gpu_build_reads_the_folds_of_any_csv_writer(tmp_path):
    root = _inputs_ws(tmp_path)
    folds = np.arange(12) % 3
    (root / "data" / "folds.csv").write_bytes(_split_files(folds)["quoted"])
    kdir = G.build_kernel(root, "q", "python src/train.py --folds {fold}", hours=1, user="me", accelerator="none")
    assert json.loads((kdir / "kg-train.json").read_text())["folds"] == [0, 1, 2]
    (root / "data" / "folds.csv").write_bytes(b"id,split\n0,a\n")
    with pytest.raises(SystemExit, match="data/folds.csv: no 'fold' column"):
        G.build_kernel(root, "q", "python src/train.py --folds {fold}", hours=1, user="me", accelerator="none")
