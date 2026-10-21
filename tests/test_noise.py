"""Noise of comparisons on an evaluation set (#32): holdout predictions, multi-column truth, paired estimates on
any metric, cached results for status and the session hook, and the #33 rule for blends with their own holdout
score."""

import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from kgkit import evalset as EV
from kgkit import state as S
from kgkit.experiment import Ledger

N, L = 58, 12
COLS = [f"l{j}" for j in range(L)]
SPEC = "holdout.csv:" + ",".join(COLS)


def _multilabel(rng, n=N):
    """A 12-label target with 5-50% prevalence and two strongly correlated models (b = a + small tweak)."""
    thr = -np.quantile(np.random.default_rng(1).standard_normal(100_000), np.linspace(0.05, 0.5, L))
    x = rng.standard_normal((n, L)) + 0.5 * rng.standard_normal((n, 1))
    y = (x + 0.45 * rng.standard_normal((n, L)) > thr).astype(int)
    a = x + 0.55 * rng.standard_normal((n, L))
    return y, a, a + 0.12 * rng.standard_normal((n, L)), x + 1.2 * rng.standard_normal((n, L))


@pytest.fixture
def hws(tmp_path, monkeypatch):
    """holdout+lb workspace with a 58-study, 12-label holdout and three prediction files (rows shuffled)."""
    y, a, b, c = _multilabel(np.random.default_rng(3))
    ids = [f"s{i:03d}" for i in range(N)]
    pd.DataFrame(y, columns=COLS).assign(study_id=ids)[["study_id", *COLS]].to_csv(tmp_path / "holdout.csv", index=False)
    for k, p in {"a": a, "b": b, "c": c}.items():
        df = pd.DataFrame(1 / (1 + np.exp(-p)), columns=COLS).assign(study_id=ids)
        df.sample(frac=1, random_state=1).to_csv(tmp_path / f"pred_{k}.csv", index=False)
    S.CompetitionState(slug="rsna", metric="mean_col_auc", id_col="study_id", validation="holdout+lb", holdout_n=N,
                       eval_truth=SPEC).save(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _cli(*args):
    from kgkit import cli

    cli.main(list(args))


# ----------------------------------------------------------------------------- phase 1: predictions on the eval set
def test_multi_column_truth_and_id_aligned_preds(hws):
    from kgkit import cli
    from kgkit import metrics as M

    y = cli._truth(SPEC)
    assert y.shape == (N, L)
    p = cli._preds("pred_a.csv:" + ",".join(COLS), SPEC, "study_id")  # shuffled file, put back in truth order
    raw = pd.read_csv(hws / "pred_a.csv").set_index("study_id").loc[pd.read_csv(hws / "holdout.csv")["study_id"]]
    assert np.allclose(p, raw[COLS].to_numpy())
    assert M.score("mean_col_auc", y, p) > 0.85
    pd.DataFrame({"study_id": ["x"], "l0": ["a"]}).to_csv(hws / "bad.csv", index=False)
    with pytest.raises(SystemExit):
        cli._truth("bad.csv:study_id,l0")


def test_ledger_add_pred_recomputes_and_stores_holdout_predictions(hws, capsys):
    _cli("ledger", "add", "a16", "0.5", "base", "--pred", "pred_a.csv:" + ",".join(COLS))
    out = capsys.readouterr().out
    rec = Ledger(hws).records()[-1]
    assert "recomputed from the predictions" in out and rec["cv"] > 0.85  # the typed 0.5 is not trusted
    assert rec["holdout_path"] == "artifacts/0001_a16/holdout.npy" and rec["eval_hash"] == EV.spec_hash(hws, SPEC)
    assert Ledger(hws).load_eval_pred("1").shape == (N, L)
    with pytest.raises(SystemExit):
        _cli("ledger", "add", "x", "auto")  # auto needs --pred


def test_blend_with_its_own_holdout_predictions_counts(hws):
    # #33: a blend that includes public files still has its own holdout score once its predictions are stored
    led = Ledger(hws)
    solo = led.log("solo", 0.905)
    _cli("ledger", "add", "blend", "auto", "--tag", "lb-blend", "--pred", "pred_a.csv:" + ",".join(COLS))
    blend = led.records()[-1]
    assert EV.own_score(blend) and not EV.own_score({"cv": 0.94, "tags": ["lb-blend"]})
    assert led.best(1)[0]["id"] == blend["id"] != solo["id"]
    for r, lb in zip(led.records() + [led.log("m", 0.91), led.log("n", 0.92)], (0.95, 0.96, 0.951, 0.955)):
        led.attach_lb(r["id"], lb)
    assert led.cv_lb_correlation()["n"] == 4


def test_compare_in_holdout_mode_uses_the_stored_predictions(hws, capsys):
    cols = ",".join(COLS)
    _cli("ledger", "add", "a", "auto", "--pred", f"pred_a.csv:{cols}")
    _cli("ledger", "decide", "1", "baseline")
    _cli("ledger", "add", "c", "auto", "--pred", f"pred_c.csv:{cols}")
    capsys.readouterr()
    _cli("ledger", "compare", "2", "--boot", "200")
    out = capsys.readouterr().out
    assert "0002_c vs 0001_a: DISCARD" in out and "58 rows" in out
    with pytest.raises(SystemExit):
        _cli("ledger", "compare", "2", "--folds", "holdout.csv:l0")  # no folds on a fixed holdout


def test_records_without_predictions_explain_what_is_missing(hws):
    led = Ledger(hws)
    led.log("typed", 0.93)
    with pytest.raises(ValueError, match="no holdout predictions"):
        led.load_eval_pred("1")


def test_gpu_import_remaps_holdout_path(tmp_path):
    from kgkit import gpu as G

    out = tmp_path / "out"
    (out / "artifacts" / "0001_r").mkdir(parents=True)
    np.save(out / "artifacts" / "0001_r" / "holdout.npy", np.zeros(3))
    S.CompetitionState(slug="x").save(tmp_path)
    Ledger(tmp_path).log("local", 0.5)
    rec = {"id": "0001_r", "name": "r", "cv": 0.9, "holdout_path": "artifacts/0001_r/holdout.npy"}
    new = G.import_ledger(tmp_path, out, {"ledger": [rec], "build_id": "b1"}, "u/k", 1)[0]
    assert new["holdout_path"] == "artifacts/0002_r/holdout.npy" and (tmp_path / new["holdout_path"]).is_file()


def test_evalset_keys_and_decision_counts(tmp_path):
    st = S.CompetitionState(slug="x", validation="holdout+lb")
    assert EV.current_key(tmp_path, st) is None  # no labels configured: no fingerprint
    pd.DataFrame({"y": [0, 1]}).to_csv(tmp_path / "h.csv", index=False)
    st.eval_truth = "h.csv:y"
    k = EV.current_key(tmp_path, st)
    assert k and k != EV.spec_hash(tmp_path, "h.csv:y,y")
    recs = [{"decision": "keep", "eval_hash": k}, {"decision": "discard"}, {"decision": "keep", "eval_hash": "zz"},
            {"decision": "baseline", "eval_hash": k}]
    assert EV.decisions_on(recs, k, holdout=True) == 2
