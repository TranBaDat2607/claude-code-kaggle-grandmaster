"""Sweeps (#134): many variants of one pipeline run as one unit, with successive halving. Each rung runs the survivors
at a higher fidelity (a screen, a bigger screen, ..., the full run), compares each with the rung's control and keeps
the best 1/eta; the last rung is the full run, compared with the baseline itself, with best of N over the whole
sweep. The runner only runs commands, so it works for any pipeline that honours KG_SCREEN / KG_FIDELITY."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from kgkit import cli
from kgkit import sweep as SW
from kgkit.experiment import Ledger

PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "kaggle-grandmaster"

PIPE = r'''
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from kgkit import screen as SC
from kgkit.experiment import Ledger

ap = argparse.ArgumentParser()
ap.add_argument("--name"); ap.add_argument("--parent"); ap.add_argument("--noise", type=float, default=1.0)
ap.add_argument("--epochs", type=int); ap.add_argument("--fail", action="store_true")
a = ap.parse_args()
if a.fail:
    raise SystemExit(3)
train = pd.read_csv("data/train.csv")
folds = pd.read_csv("data/folds.csv")["fold"].to_numpy()
s = SC.active()
use = s.mask(train["id"]) if s is not None else np.ones(len(train), bool)
run = (s.folds if s is not None and s.folds else None) or sorted(set(folds.tolist()))
rng = np.random.default_rng(0)
signal = (1.5 * train["x1"] - train["x2"]).to_numpy()
noise = rng.normal(0, 1, len(train))
oof = np.full(len(train), np.nan)
scores = []
for k in run:
    rows = use & (folds == k)
    oof[rows] = signal[rows] + a.noise * noise[rows]
    scores.append(roc_auc_score(train["target"][rows], oof[rows]))
ok = ~np.isnan(oof)
with open("epochs_seen.txt", "a") as f:
    f.write(f"{a.name} {a.epochs}\n")
Ledger().log(a.name, roc_auc_score(train["target"][ok], oof[ok]), fold_scores=scores, oof=oof, parent=a.parent,
             params={"noise": a.noise})
'''


@pytest.fixture
def ws(tmp_path, monkeypatch, binary_df):
    monkeypatch.chdir(tmp_path)
    for k in ("KG_SCREEN", "KG_FIDELITY", "KG_CONTROL_OF", "KG_SWEEP", "KG_SWEEP_VARIANT", "KG_RUNG"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PYTHONPATH", str(PLUGIN))
    cli.main(["init", "demo", "--metric", "auc", "--target", "target", "--id-col", "id"])
    binary_df.to_csv(tmp_path / "data" / "train.csv", index=False)
    cli.main(["folds", "data/train.csv", "--target", "target", "--id-col", "id", "--out", "data/folds.csv"])
    (tmp_path / "pipe.py").write_text(PIPE, encoding="utf-8")
    cmd = f'"{sys.executable}" pipe.py --name base'
    assert os.system(cmd) == 0
    Ledger(tmp_path).decide("0001", "baseline")
    cli.main(["screen", "new", "f0", "--folds", "0"])
    cli.main(["screen", "new", "f01", "--folds", "0", "1"])
    return tmp_path


CMD = f'"{sys.executable}" pipe.py --parent {{base}} --name {{sweep}}_{{variant}}_r{{rung}} {{rung_args}} {{args}}'


# ---------------------------------------------------------------------------- definitions
def test_grid_and_variants_make_the_variant_list():
    vs = SW.variants(["a=--noise 0.5"], ["--noise=0.6,0.7", "--epochs=2,4"])
    assert vs[0] == {"name": "a", "args": "--noise 0.5"}
    names = [v["name"] for v in vs[1:]]
    assert names == ["noise0.6_epochs2", "noise0.6_epochs4", "noise0.7_epochs2", "noise0.7_epochs4"]
    assert vs[1]["args"] == "--noise 0.6 --epochs 2"
    with pytest.raises(ValueError, match="twice"):
        SW.variants(["a=--x 1", "a=--x 2"], [])


def test_rungs_parse_screens_and_settings_and_end_with_the_full_run():
    rs = SW.rungs(["screen=f0,epochs=2", "screen=f01"])
    assert rs == [{"screen": "f0", "settings": {"epochs": 2}}, {"screen": "f01", "settings": {}},
                  {"screen": None, "settings": {}}]
    assert SW.rung_args({"epochs": 2, "img_size": 256}) == "--epochs 2 --img-size 256"
    assert SW.rungs([]) == [{"screen": None, "settings": {}}]


def test_survivors_are_the_best_third_by_gain():
    runs = {f"v{i}": {"gain": g, "z": 1.0, "status": "done"} for i, g in enumerate([0.1, 0.5, -0.2, 0.3, 0.0, 0.4])}
    runs["broken"] = {"status": "failed"}
    assert SW.survivors(runs, eta=3) == ["v1", "v5"]
    assert SW.survivors({"only": {"gain": -1, "z": 0, "status": "done"}}, eta=3) == ["only"]


# ---------------------------------------------------------------------------- running a sweep
def _sweep(ws, *extra):
    cli.main(["sweep", "new", "nz", "--base", "0001", "--cmd", CMD,
              "--grid", "--noise=0.2,0.3,0.4,0.6,0.7,0.8,0.9,1.1,1.2",
              "--rung", "screen=f0,epochs=2", "--rung", "screen=f01", "--eta", "3", *extra])


def test_a_sweep_halves_through_its_rungs_and_confirms_the_finalist(ws, capsys):
    _sweep(ws)
    cli.main(["sweep", "run", "nz", "--parallel", "3"])
    out = capsys.readouterr().out
    st = json.loads((ws / ".kaggle-gm" / "sweeps" / "nz.json").read_text())
    assert st["survivors"]["0"] == [v["name"] for v in st["variants"]]
    assert st["survivors"]["1"] == ["noise0.2", "noise0.3", "noise0.4"]
    assert st["survivors"]["2"] == ["noise0.2"]
    assert st["state"] == "done"
    led = Ledger(ws)
    recs = led.records()
    controls = [r for r in recs if r.get("control_of")]
    assert len(controls) == 2 and all(r["control_of"] == "0001_base" for r in controls)
    screens = [r for r in recs if r.get("fidelity") and not r.get("control_of")]
    assert len(screens) == 9 + 3
    assert all(r["sweep"]["name"] == "nz" for r in screens)
    final = st["runs"]["noise0.2"]["2"]["record"]
    assert not led.get(final).get("fidelity") and led.get(final)["parent"] == "0001_base"
    assert led.tries(final, "0001_base") == 9  # the whole sweep, each variant once
    assert "best of 9" in out and f"kgkit exp finish {final} keep" in out
    seen = (ws / "epochs_seen.txt").read_text().split("\n")
    assert any(line.endswith(" 2") and "_r0" in line for line in seen)  # rung 0 ran with --epochs 2
    assert all(line.endswith(" None") for line in seen if "_r1" in line or "_r2" in line)
    fid = led.get(st["runs"]["noise0.9"]["0"]["record"])["fidelity"]
    assert fid["screen"] == "f0" and fid["epochs"] == 2


def test_sweep_status_decide_and_resume(ws, capsys):
    _sweep(ws)
    cli.main(["sweep", "run", "nz", "--one", "--parallel", "3"])
    st = json.loads((ws / ".kaggle-gm" / "sweeps" / "nz.json").read_text())
    assert st["state"] == "rung 1" and "1" not in st["runs"]["noise0.2"]
    n = len(Ledger(ws).records())
    cli.main(["sweep", "run", "nz", "--parallel", "3"])  # continues at rung 1, nothing of rung 0 is re-run
    assert len(Ledger(ws).records()) == n + 1 + 3 + 1  # control, 3 screens; then the finalist's full run
    capsys.readouterr()
    cli.main(["sweep", "status", "nz"])
    out = capsys.readouterr().out
    assert "noise1.2" in out and "pruned at rung 0" in out and "noise0.2" in out and "done" in out
    cli.main(["sweep", "decide", "nz"])
    out = capsys.readouterr().out
    led = Ledger(ws)
    st = json.loads((ws / ".kaggle-gm" / "sweeps" / "nz.json").read_text())
    pruned = led.get(st["runs"]["noise1.2"]["0"]["record"])
    assert pruned["decision"] == "discard" and "pruned at rung 0" in pruned["decision_note"]
    promoted = led.get(st["runs"]["noise0.3"]["0"]["record"])
    assert promoted["decision"] == "inconclusive" and "promoted" in promoted["decision_note"]
    final = led.get(st["runs"]["noise0.2"]["2"]["record"])
    assert final.get("decision") is None and f"kgkit exp finish {final['id']}" in out
    assert all(not r.get("decision") for r in led.records() if r.get("control_of"))
    cli.main(["sweep", "run", "nz"])
    assert "already done" in capsys.readouterr().out


def test_a_failed_run_is_reported_and_left_out(ws, capsys):
    cli.main(["sweep", "new", "f", "--base", "0001", "--cmd", CMD, "--variant", "good=--noise 0.3",
              "--variant", "bad=--fail", "--variant", "meh=--noise 0.9", "--rung", "screen=f0", "--eta", "2"])
    cli.main(["sweep", "run", "f", "--one"])
    out = capsys.readouterr().out
    st = json.loads((ws / ".kaggle-gm" / "sweeps" / "f.json").read_text())
    assert st["runs"]["bad"]["0"]["status"] == "failed" and "exit 3" in out
    assert st["survivors"]["1"] == ["good"]
    assert (ws / ".kaggle-gm" / "sweeps" / "f" / "bad_r0.log").is_file()


def test_dry_run_prints_the_commands_with_their_environment(ws, capsys):
    _sweep(ws)
    capsys.readouterr()
    cli.main(["sweep", "run", "nz", "--dry-run"])
    out = capsys.readouterr().out
    assert "KG_SCREEN=f0" in out and "KG_CONTROL_OF=0001_base" in out and '--noise 0.2' in out
    assert "--epochs 2" in out and len(Ledger(ws).records()) == 1


def test_an_existing_control_is_reused(ws, capsys):
    _sweep(ws)
    cli.main(["sweep", "run", "nz", "--one", "--parallel", "3"])
    cli.main(["sweep", "new", "again", "--base", "0001", "--cmd", CMD, "--variant", "x=--noise 0.5",
              "--rung", "screen=f0,epochs=2"])
    before = [r["id"] for r in Ledger(ws).records() if r.get("control_of")]
    cli.main(["sweep", "run", "again", "--one"])
    after = [r["id"] for r in Ledger(ws).records() if r.get("control_of")]
    assert before == after  # same baseline, same fidelity: the control on file is reused


def test_sweep_new_refuses_a_command_without_the_variant_args(ws):
    with pytest.raises(SystemExit, match="args"):
        cli.main(["sweep", "new", "x", "--base", "0001", "--cmd", "python pipe.py --name {variant}",
                  "--variant", "a=--noise 1"])
    with pytest.raises(SystemExit, match="screen"):
        cli.main(["sweep", "new", "x", "--base", "0001", "--cmd", CMD, "--variant", "a=--noise 1",
                  "--rung", "screen=nope"])


def test_finalists_keep_the_best_and_a_failed_variant_is_no_try(ws, capsys):
    cli.main(["sweep", "new", "two", "--base", "0001", "--cmd", CMD, "--variant", "good=--noise 0.3",
              "--variant", "bad=--fail", "--variant", "meh=--noise 0.6", "--rung", "screen=f0", "--eta", "1"])
    cli.main(["sweep", "run", "two"])
    out = capsys.readouterr().out
    assert "picked from 2 variants" in out and "best of 2" in out  # the failed one gave noise no chance
    assert "2 finalists: keep the best, good" in out
    cli.main(["sweep", "decide", "two"])
    out = capsys.readouterr().out
    st = json.loads((ws / ".kaggle-gm" / "sweeps" / "two.json").read_text())
    meh = st["runs"]["meh"]["1"]["record"]
    assert f'kgkit exp finish {meh} discard "lost to good in sweep two"' in out


def test_sweep_run_is_never_pre_approved():
    from kgkit import permissions as PM

    rules = PM.rules()
    assert "sweep run" in PM.NEVER and "Bash(kgkit sweep status *)" in rules
    assert not any(r.startswith("Bash(kgkit sweep run") or r == "Bash(kgkit sweep *)" for r in rules)
