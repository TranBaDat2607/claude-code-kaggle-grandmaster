"""Screens (#134): cheap runs of a pipeline on a frozen subset (rows, whole groups, the latest window, some folds) or
at reduced settings, tagged with their fidelity. A screen is compared only with a control (the baseline's pipeline at
the same fidelity), never with a full run, and its score is never taken for the pipeline's: not the best CV, not a
baseline, not a final pick."""

import json

import numpy as np
import pandas as pd
import pytest

from kgkit import cli
from kgkit import evalset as E
from kgkit import screen as SC
from kgkit.experiment import Ledger, report_comparison


@pytest.fixture
def ws(tmp_path, monkeypatch, binary_df):
    monkeypatch.chdir(tmp_path)
    for k in ("KG_SCREEN", "KG_FIDELITY", "KG_CONTROL_OF"):
        monkeypatch.delenv(k, raising=False)
    cli.main(["init", "demo", "--metric", "auc", "--target", "target", "--id-col", "id"])
    df = binary_df.copy()
    df["patient"] = df["id"] // 10
    df["day"] = pd.date_range("2024-01-01", periods=len(df), freq="h").strftime("%Y-%m-%d %H:%M")
    df.to_csv(tmp_path / "data" / "train.csv", index=False)
    cli.main(["folds", "data/train.csv", "--target", "target", "--id-col", "id", "--out", "data/folds.csv"])
    return tmp_path


# ---------------------------------------------------------------------------- making a screen
def test_select_takes_a_stratified_share_and_is_deterministic():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.2).astype(int)
    m = SC.select(np.arange(1000), 0.3, seed=1, strata=y)
    assert m.sum() == pytest.approx(300, abs=2) and y[m].mean() == pytest.approx(0.2, abs=0.01)
    assert (SC.select(np.arange(1000), 0.3, seed=1, strata=y) == m).all()
    assert not (SC.select(np.arange(1000), 0.3, seed=2, strata=y) == m).all()
    assert SC.select(np.arange(10), 1.0).all()


def test_select_keeps_groups_whole_and_takes_the_latest_window():
    group = np.repeat(np.arange(100), 10)
    m = SC.select(np.arange(1000), 0.3, seed=0, group=group)
    assert m.sum() == pytest.approx(300, abs=10)
    for g in np.unique(group):
        assert len(set(m[group == g])) == 1  # all of a group, or none of it
    t = np.r_[np.arange(500, 1000), np.arange(500)]  # newest rows first in the file
    m = SC.select(np.arange(1000), 0.2, time=t)
    assert set(t[m]) == set(range(800, 1000))
    tg = SC.select(np.arange(1000), 0.2, time=np.arange(1000), group=group)
    assert set(group[tg]) == set(range(80, 100))  # the latest groups, whole


def test_screen_new_writes_a_frozen_subset_from_the_workspace(ws, capsys):
    cli.main(["screen", "new", "s30", "--frac", "0.3", "--folds", "0", "--group", "patient"])
    out = capsys.readouterr().out
    assert "s30" in out and "KG_SCREEN=s30" in out
    spec = json.loads((ws / ".kaggle-gm" / "screens" / "s30.json").read_text())
    assert spec["frac"] == 0.3 and spec["folds"] == [0] and spec["strategy"] == "group:patient"
    s = SC.load(ws, "s30")
    train = pd.read_csv(ws / "data" / "train.csv")
    m = s.mask(train["id"])
    assert m.sum() == s.n and abs(m.mean() - 0.3) < 0.02
    assert train.groupby("patient")["id"].apply(lambda ids: len(set(s.mask(ids)))).max() == 1
    cli.main(["screen", "list"])
    assert "s30" in capsys.readouterr().out
    cli.main(["screen", "new", "late", "--frac", "0.2", "--time-col", "day"])
    late = SC.load(ws, "late")
    assert train.loc[late.mask(train["id"]), "day"].min() >= train["day"].sort_values().iloc[int(0.8 * len(train)) - 1]


def test_screen_new_on_ids_of_another_table(ws, capsys):
    pd.DataFrame({"tomo": [f"t{i}" for i in range(40)], "site": np.arange(40) % 4}).to_csv(ws / "tomos.csv",
                                                                                         index=False)
    cli.main(["screen", "new", "tq", "--frac", "0.5", "--ids", "tomos.csv:tomo", "--group", "site"])
    s = SC.load(ws, "tq")
    assert s.n == 20 and s.strategy == "group:site"
    assert s.mask([f"t{i}" for i in range(40)]).sum() == 20


# ---------------------------------------------------------------------------- the pipeline's side
def _run(ws, name, noise, seed=0, parent=None, folds_only=None, control_of=None, monkeypatch=None):
    """A pipeline that honours the active screen: trains on its rows, predicts only its folds' rows."""
    train = pd.read_csv(ws / "data" / "train.csv")
    folds = pd.read_csv(ws / "data" / "folds.csv")["fold"].to_numpy()
    s = SC.active(ws)
    use = s.mask(train["id"]) if s is not None else np.ones(len(train), bool)
    run_folds = (s.folds if s is not None and s.folds else None) or sorted(set(folds))
    rng = np.random.default_rng(seed)
    signal = 1.5 * train["x1"] - train["x2"]
    oof = np.full(len(train), np.nan)
    scores = []
    from sklearn.metrics import roc_auc_score

    for k in run_folds:
        rows = use & (folds == k)
        oof[rows] = signal[rows] + rng.normal(0, noise, rows.sum())
        scores.append(roc_auc_score(train["target"][rows], oof[rows]))
    ok = ~np.isnan(oof)
    cv = roc_auc_score(train["target"][ok], oof[ok])
    return Ledger(ws).log(name, cv, fold_scores=scores, oof=oof, parent=parent)


def test_a_run_under_a_screen_records_its_fidelity_and_a_control_names_its_baseline(ws, monkeypatch):
    cli.main(["screen", "new", "s30", "--frac", "0.3", "--folds", "0"])
    base = _run(ws, "base", 1.0)
    monkeypatch.setenv("KG_SCREEN", "s30")
    monkeypatch.setenv("KG_FIDELITY", '{"epochs": 3}')
    s = _run(ws, "screen", 0.5, parent=base["id"])
    fid = s["fidelity"]
    assert fid["screen"] == "s30" and fid["frac"] == 0.3 and fid["folds"] == [0] and fid["epochs"] == 3
    assert fid["screen_hash"] == SC.load(ws, "s30").hash
    monkeypatch.setenv("KG_CONTROL_OF", "1")
    ctl = _run(ws, "control", 1.0)
    assert ctl["control_of"] == base["id"] and E.fidelity_key(ctl) == E.fidelity_key(s)
    assert E.fidelity_key(base) == "" and E.partial(s) and not E.partial(base)
    monkeypatch.setenv("KG_FIDELITY", "{not json")
    with pytest.raises(ValueError, match="KG_FIDELITY"):
        _run(ws, "bad", 1.0)
    monkeypatch.setenv("KG_SCREEN", "nope")
    with pytest.raises(ValueError, match="nope"):
        SC.active(ws)


def test_a_screen_is_never_the_best_cv_a_baseline_or_an_own_score(ws, monkeypatch):
    base = _run(ws, "base", 1.0)
    Ledger(ws).decide(base["id"], "baseline")
    cli.main(["screen", "new", "s30", "--frac", "0.3", "--folds", "0"])
    monkeypatch.setenv("KG_SCREEN", "s30")
    s = _run(ws, "lucky", 0.01, parent=base["id"])  # a near-perfect score on a small subset
    led = Ledger(ws)
    assert not E.own_score(s) and s["id"] not in [r["id"] for r in led.best(10)]
    assert s["id"] not in [r["id"] for r in led.borrowed_scores()]
    with pytest.raises(ValueError, match="screen"):
        led.decide(s["id"], "keep")
    with pytest.raises(ValueError, match="screen"):
        led.decide(s["id"], "baseline")
    led.decide(s["id"], "discard", "pruned")
    assert led.baseline()["id"] == base["id"]


# ---------------------------------------------------------------------------- comparisons
def test_a_screen_is_compared_with_its_control_never_with_the_full_baseline(ws, monkeypatch, capsys):
    base = _run(ws, "base", 1.0)
    Ledger(ws).decide(base["id"], "baseline")
    cli.main(["screen", "new", "f0", "--folds", "0"])  # all of fold 0
    monkeypatch.setenv("KG_SCREEN", "f0")
    s = _run(ws, "screen", 0.05, seed=1, parent=base["id"])
    capsys.readouterr()
    report_comparison(s)  # what a template prints when it logs: there is no control yet
    out = capsys.readouterr().out
    assert "not compared" in out and "control" in out and f"kgkit screen control {base['id']} f0" in out
    with pytest.raises(SystemExit, match="fidelity"):
        cli.main(["ledger", "compare", s["id"], base["id"]])
    monkeypatch.setenv("KG_CONTROL_OF", base["id"])
    ctl = _run(ws, "control", 1.0, seed=2)
    monkeypatch.delenv("KG_CONTROL_OF")
    capsys.readouterr()
    report_comparison(s)
    out = capsys.readouterr().out
    assert f"{s['id']} vs {ctl['id']}" in out and "KEEP" in out
    assert "never kept" in out and "Promote" in out and "record it:" not in out
    report_comparison(ctl)  # a control is not compared with anything itself
    assert "control of" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="screen"):
        cli.main(["exp", "finish", s["id"], "keep"])
    cli.main(["exp", "finish", s["id"], "discard", "pruned"])
    assert Ledger(ws).get(s["id"])["decision"] == "discard"


def test_screen_control_prints_the_baseline_command_on_the_screen(ws, capsys):
    led = Ledger(ws)
    base = led.log("base", 0.8, notes="")
    led.update(base["id"], cmd="python src/train_gbdt.py --model lgbm --name base")
    cli.main(["screen", "new", "s30", "--frac", "0.3"])
    capsys.readouterr()
    cli.main(["screen", "control", base["id"], "s30"])
    out = capsys.readouterr().out
    assert f"KG_SCREEN=s30 KG_CONTROL_OF={base['id']} python src/train_gbdt.py --model lgbm --name base" in out
    nocmd = led.log("typed", 0.8)
    led.update(nocmd["id"], cmd=None)  # logged by hand (under pytest, the command would be pytest's)
    cli.main(["screen", "control", nocmd["id"], "s30"])
    assert "no command on record" in capsys.readouterr().out


def test_tries_count_screens_but_not_controls(ws, monkeypatch):
    base = _run(ws, "base", 1.0)
    Ledger(ws).decide(base["id"], "baseline")
    cli.main(["screen", "new", "s30", "--frac", "0.3", "--folds", "0"])
    monkeypatch.setenv("KG_SCREEN", "s30")
    monkeypatch.setenv("KG_CONTROL_OF", base["id"])
    _run(ws, "control", 1.0, parent=base["id"])
    monkeypatch.delenv("KG_CONTROL_OF")
    ids = [_run(ws, f"v{i}", 0.9, seed=i, parent=base["id"])["id"] for i in range(3)]
    assert Ledger(ws).tries(ids[0], base["id"]) == 3


def test_controls_are_not_listed_as_undecided(ws, monkeypatch):
    from kgkit import layout

    base = _run(ws, "base", 1.0)
    monkeypatch.setenv("KG_CONTROL_OF", base["id"])
    for i in range(layout.UNDECIDED_MIN + 2):
        _run(ws, f"control{i}", 1.0, seed=i)
    assert [r["id"] for r in layout.undecided(Ledger(ws).records())] == [base["id"]]
