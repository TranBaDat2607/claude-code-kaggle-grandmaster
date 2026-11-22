"""Seed noise, measured once per workspace (#134): one pipeline run again with another seed (KG_SEED) and logged as
a replica of its record (KG_REPLICA_OF). The ledger groups the two whatever their params or git state, so every
paired comparison adds the training noise to the resampling SE; `kgkit status` says while it is missing, and
`ledger seed-noise` prints the re-run that measures it."""

import numpy as np
import pytest

from kgkit import cli
from kgkit import compare as C
from kgkit import layout
from kgkit import screen as SC
from kgkit.experiment import Ledger, report_comparison

ENV = ("KG_SCREEN", "KG_FIDELITY", "KG_CONTROL_OF", "KG_REPLICA_OF", "KG_SEED", "KG_SWEEP", "KG_SWEEP_VARIANT",
       "KG_RUNG")


@pytest.fixture
def ws(tmp_path, monkeypatch, binary_df):
    monkeypatch.chdir(tmp_path)
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    cli.main(["init", "demo", "--metric", "auc", "--target", "target", "--id-col", "id"])
    binary_df.to_csv(tmp_path / "data" / "train.csv", index=False)
    cli.main(["folds", "data/train.csv", "--target", "target", "--id-col", "id", "--out", "data/folds.csv"])
    return tmp_path


def _run(ws, name, noise, seed, parent=None, seeds=None):
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    train = pd.read_csv(ws / "data" / "train.csv")
    folds = pd.read_csv(ws / "data" / "folds.csv")["fold"].to_numpy()
    rng = np.random.default_rng(seed)
    oof = (1.5 * train["x1"] - train["x2"]).to_numpy() + rng.normal(0, noise, len(train))
    scores = [roc_auc_score(train["target"][folds == k], oof[folds == k]) for k in sorted(set(folds))]
    return Ledger(ws).log(name, roc_auc_score(train["target"], oof), fold_scores=scores, oof=oof, parent=parent,
                          seeds=seeds or [seed])


def test_a_replica_is_grouped_with_its_record_and_measures_the_seed_noise(ws, monkeypatch):
    base = _run(ws, "base", 1.0, seed=0)
    assert C.seed_variance(Ledger(ws).records()) is None
    monkeypatch.setenv("KG_REPLICA_OF", base["id"])
    monkeypatch.setenv("KG_SEED", "7")
    rep = _run(ws, "base", 1.0, seed=7)
    assert rep["replica_of"] == base["id"] and rep["replica_seed"] == 7 and not rep.get("replica_identical")
    sv = C.seed_variance(Ledger(ws).records())
    assert sv["df"] == 1 and sv["sd"] == pytest.approx(abs(rep["cv"] - base["cv"]) / np.sqrt(2))


def test_an_identical_replica_measures_nothing(ws, monkeypatch):
    base = _run(ws, "base", 1.0, seed=0)
    monkeypatch.setenv("KG_REPLICA_OF", base["id"])
    rep = _run(ws, "base", 1.0, seed=0)  # the pipeline ignored KG_SEED: the same score again
    assert rep["replica_identical"] is True
    assert C.seed_variance(Ledger(ws).records()) is None
    out = _report(rep)
    assert "identical" in out and "KG_SEED" in out


def _report(rec):
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_comparison(rec)
    return buf.getvalue()


def test_comparisons_add_the_seed_noise_and_a_replica_is_no_try_and_needs_no_decision(ws, monkeypatch, capsys):
    base = _run(ws, "base", 1.0, seed=0)
    Ledger(ws).decide(base["id"], "baseline")
    v = _run(ws, "v", 0.9, seed=1, parent=base["id"])
    capsys.readouterr()
    cmp = ["ledger", "compare", v["id"], base["id"], "--truth", "data/train.csv:target"]
    cli.main(cmp)
    assert "seed noise (sd" not in capsys.readouterr().out
    monkeypatch.setenv("KG_REPLICA_OF", base["id"])
    monkeypatch.setenv("KG_SEED", "7")
    rep = _run(ws, "base", 1.0, seed=7, parent=base["id"])
    monkeypatch.delenv("KG_REPLICA_OF")
    monkeypatch.delenv("KG_SEED")
    out = _report(rep)
    assert f"seed replica of {base['id']}" in out and "seed noise" in out
    capsys.readouterr()
    cli.main(cmp)
    out = capsys.readouterr().out
    assert "seed noise (sd" in out and "seed noise not included" not in out
    led = Ledger(ws)
    assert led.tries(v["id"], base["id"]) == 1  # the replica is a measurement, not a try
    assert rep["id"] not in [r["id"] for r in layout.undecided(led.records())]


def test_ledger_seed_noise_prints_the_re_run_then_the_measure(ws, monkeypatch, capsys):
    base = _run(ws, "base", 1.0, seed=42, seeds=[42, 43])
    led = Ledger(ws)
    led.update(base["id"], cmd="python src/train_gbdt.py --model lgbm --name base")
    led.decide(base["id"], "baseline")
    capsys.readouterr()
    cli.main(["status", "--offline"])
    assert "seed noise: not measured" in capsys.readouterr().out
    cli.main(["ledger", "seed-noise"])
    out = capsys.readouterr().out
    assert "not measured" in out
    assert f"KG_SEED=44 KG_REPLICA_OF={base['id']} python src/train_gbdt.py --model lgbm --name base" in out
    monkeypatch.setenv("KG_REPLICA_OF", base["id"])
    monkeypatch.setenv("KG_SEED", "44")
    _run(ws, "base", 1.0, seed=44, seeds=[44, 45])
    monkeypatch.delenv("KG_REPLICA_OF")
    monkeypatch.delenv("KG_SEED")
    capsys.readouterr()
    cli.main(["ledger", "seed-noise"])
    out = capsys.readouterr().out
    assert "SD" in out and "not measured" not in out
    cli.main(["status", "--offline"])
    assert "seed noise: not measured" not in capsys.readouterr().out


def test_run_tag_keeps_a_seed_re_run_apart(ws, monkeypatch):
    assert SC.run_tag(ws) == ""
    monkeypatch.setenv("KG_SEED", "7")
    assert SC.run_tag(ws) == "__seed7"
