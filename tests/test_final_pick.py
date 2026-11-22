"""Choosing the k finals by simulating the private leaderboard (#142): the set with the highest expected best score,
from the candidates' predictions on the evaluation rows, in any metric direction, for any k."""

import json

import numpy as np
import pandas as pd
import pytest

from kgkit import cli
from kgkit import finalpick as FP
from kgkit import finals as FN
from kgkit import metrics as M
from kgkit import state as S
from kgkit.experiment import Ledger


def _binary(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    shared = rng.normal(size=n)
    strong = y + 1.0 * shared + 0.2 * rng.normal(size=n)       # the main line
    twin = strong + 1e-3 * rng.normal(size=n)                   # a near-identical blend of it
    other = y + 1.15 * rng.normal(size=n)                       # weaker, errors of its own
    return y, {"strong": strong, "twin": twin, "other": other}


# ---------------------------------------------------------------------------- the ranking
def test_an_uncorrelated_second_pick_beats_a_near_identical_one():
    y, p = _binary()
    sim = FP.simulate(y, p, "auc", size=400)
    ranked = FP.rank_sets(sim, 2, greater=True)
    value = dict(zip(sim["ids"], sim["value"]))
    assert value["other"] < min(value["strong"], value["twin"])  # the top 2 by score are the twins
    assert sorted(ranked["top"]) == ["strong", "twin"]
    best = ranked["sets"][0]
    assert "other" in best["set"] and best["gain"] > 2 * best["gain_mc"]
    assert best["corr"] < 0.5 and ranked["sets"][-1]["corr"] > 0.99
    assert FP.settled(ranked)


def test_a_candidate_best_everywhere_is_always_in_the_set():
    y, p = _binary()
    p["star"] = y * 3.0 + 0.2 * np.random.default_rng(5).normal(size=len(y))
    for k in (1, 2, 3):
        ranked = FP.rank_sets(FP.simulate(y, p, "auc", size=400), k, greater=True)
        assert "star" in ranked["sets"][0]["set"] and len(ranked["sets"][0]["set"]) == k


def test_lower_is_better_picks_the_same_way():
    rng = np.random.default_rng(1)
    n = 4000
    y = rng.normal(size=n)
    shared = rng.normal(size=n)
    p = {"strong": y + 0.5 * shared + 0.1 * rng.normal(size=n)}
    p["twin"] = p["strong"] + 1e-3 * rng.normal(size=n)
    p["other"] = y + 0.53 * rng.normal(size=n)
    sim = FP.simulate(y, p, "rmse", size=300)
    ranked = FP.rank_sets(sim, 2, greater=M.get("rmse").greater_is_better)
    assert sorted(ranked["top"]) == ["strong", "twin"]
    assert "other" in ranked["sets"][0]["set"]
    assert ranked["sets"][0]["e_best"] < ranked["top_e_best"]  # a lower expected error than the twins


def test_one_final_is_the_best_expected_single_and_three_finals_are_sets_of_three():
    y, p = _binary()
    sim = FP.simulate(y, p, "auc")
    one = FP.rank_sets(sim, 1, greater=True)
    assert one["sets"][0]["set"] == [sim["ids"][int(np.argmax(sim["value"]))]]
    three = FP.rank_sets(sim, 3, greater=True)
    assert len(three["sets"]) == 1 and len(three["sets"][0]["set"]) == 3
    assert FP.rank_sets(sim, 5, greater=True)["k"] == 3  # more finals than candidates: all of them


def test_the_bootstrap_scores_every_candidate_on_the_same_resamples():
    y, p = _binary(n=1500)
    sim = FP.simulate(y, p, "auc", method="bootstrap", n_draws=300, size=500)
    assert sim["method"] == "bootstrap" and sim["draws"].shape == (300, 3)
    i, j = sim["ids"].index("strong"), sim["ids"].index("twin")
    assert sim["corr"][i, j] > 0.99  # paired: the twins move together resample by resample


def test_groups_are_resampled_whole():
    y, p = _binary(n=2000)
    groups = np.repeat(np.arange(200), 10)
    sim = FP.simulate(y, p, "auc", groups=groups, method="bootstrap", n_draws=100)
    assert sim["unit"] == "groups" and sim["n_units"] == 200
    clt = FP.simulate(y, p, "auc", groups=groups, method="clt")
    assert clt["unit"] == "groups" and clt["method"] == "clt"
    # clustered rows share their noise: wider than the same rows taken one by one
    assert (clt["se"] > FP.simulate(y, p, "auc", method="clt")["se"]).all()


def test_time_blocks_are_contiguous_in_time():
    t = np.random.default_rng(0).permutation(1000)
    b = FP.time_blocks(t, 50)
    assert len(set(b.tolist())) == 50
    order = np.argsort(t)
    assert (np.diff(b[order]) >= 0).all()  # in time order, the block never goes back


def test_a_small_evaluation_set_says_the_sets_are_hard_to_tell_apart():
    y, p = _binary(n=80)
    sim = FP.simulate(y, p, "auc", n_draws=200)
    assert any("small evaluation set" in c for c in sim["caveats"])


# ---------------------------------------------------------------------------- the candidates
def test_the_pool_says_why_each_record_is_left_out():
    st = S.CompetitionState(slug="demo", metric="auc")
    recs = [
        {"id": "0001_ok", "cv": 0.8, "oof_path": "a.npy", "decision": "baseline", "folds_hash": "h"},
        {"id": "0002_fit", "cv": 0.9, "oof_path": "b.npy", "tags": ["public-lb-fitted"], "folds_hash": "h"},
        {"id": "0003_tuned", "cv": 0.9, "oof_path": "c.npy", "tags": ["public-tuned"], "folds_hash": "h"},
        {"id": "0004_lbonly", "cv": 0.9, "decision": "keep", "folds_hash": "h"},
        {"id": "0005_old", "cv": 0.9, "oof_path": "d.npy", "folds_hash": "other"},
        {"id": "0006_seed", "cv": 0.9, "oof_path": "e.npy", "replica_of": "0001_ok", "folds_hash": "h"},
        {"id": "0007_gone", "cv": 0.9, "oof_path": "f.npy", "decision": "discard", "folds_hash": "h"},
    ]
    cands, out = FP.pool(st, recs, None, "h", [r["id"] for r in recs])
    assert [r["id"] for r in cands] == ["0001_ok"]
    why = dict(out)
    assert "fitted" in why["0002_fit"] and "public-tuned" in why["0003_tuned"]
    assert "no OOF" in why["0004_lbonly"] and "another fold split" in why["0005_old"]
    assert "seed re-run" in why["0006_seed"] and why["0007_gone"] == "discarded"


def test_the_pool_keeps_only_records_through_the_vault():
    st = S.CompetitionState(slug="demo", metric="auc")
    recs = [{"id": "0001_a", "cv": 0.8, "oof_path": "a.npy", "vault": "pass"},
            {"id": "0002_b", "cv": 0.8, "oof_path": "b.npy"}]
    cands, out = FP.pool(st, recs, {"overrides": {}}, None, ["0001_a", "0002_b"])
    assert [r["id"] for r in cands] == ["0001_a"] and "vault" in dict(out)["0002_b"]


def test_a_candidate_scored_on_a_subset_of_the_rows_is_dropped():
    full = np.arange(100.0)
    part = full.copy()
    part[:50] = np.nan
    mask, dropped = FP.shared_rows({"a": full, "b": full + 1, "c": part})
    assert dropped == ["c"] and mask.all()


@pytest.mark.parametrize("n_test,public,hidden,size,others", [
    (10_000, 2_500, False, 7_500, 0),
    (10_000, None, False, 7_500, 2),
    (None, None, False, 3_000, 2),
    (10_000, 2_500, True, 3_000, 2),
])
def test_private_size(n_test, public, hidden, size, others):
    got, alts, note = FP.private_sizes(n_test, public, 3_000, hidden)
    assert got == size and len(alts) == others and note


# ---------------------------------------------------------------------------- the command
def _workspace(tmp_path, monkeypatch, task=""):
    monkeypatch.chdir(tmp_path)
    cli.main(["init", "demo", "--metric", "auc", "--target", "target", "--id-col", "id"])
    y, p = _binary(n=3000)
    pd.DataFrame({"id": np.arange(len(y)), "target": y}).to_csv(tmp_path / "data" / "train.csv", index=False)
    if task:
        st = S.load(tmp_path)
        st.task = task
        st.save(tmp_path)
    led = Ledger(tmp_path)
    test = np.zeros(4000)
    led.log("strong", M.get("auc")(y, p["strong"]), oof=p["strong"], test_pred=test, decision="baseline")
    led.log("twin", M.get("auc")(y, p["twin"]), oof=p["twin"], test_pred=test, decision="keep", parent="0001")
    led.log("other", M.get("auc")(y, p["other"]), oof=p["other"], test_pred=test, decision="baseline", track="nn")
    return tmp_path


def test_final_pick_recommends_and_saves_the_set(tmp_path, monkeypatch, capsys):
    ws = _workspace(tmp_path, monkeypatch)
    cli.main(["config", "public_lb_n=3800"])
    capsys.readouterr()
    cli.main(["final", "pick", "0001", "0002", "0003", "--draws", "5000"])
    out = capsys.readouterr().out
    assert "private LB = 4,000 test rows - 3,800 public" in out
    assert "Recommended:" in out and "not submitted yet" in out
    pick = FN.load_pick(ws, "demo")
    assert "0003_other" in pick["ids"] and sorted(pick["top"]) == ["0001_strong", "0002_twin"]
    assert json.loads((ws / ".kaggle-gm" / FN.PICK_FILE).read_text())["k"] == 2


def test_final_pick_default_pool_is_the_honest_candidates_and_line_baselines(tmp_path, monkeypatch, capsys):
    _workspace(tmp_path, monkeypatch)
    capsys.readouterr()
    cli.main(["final", "pick", "--draws", "2000"])
    out = capsys.readouterr().out
    assert "0002_twin" in out and "0003_other" in out and "0001_strong " not in out  # 0001 is superseded


@pytest.mark.parametrize("task", ["rl", "optimization"])
def test_final_pick_skips_where_rows_mean_nothing(tmp_path, monkeypatch, capsys, task):
    _workspace(tmp_path, monkeypatch, task)
    capsys.readouterr()
    cli.main(["final", "pick"])
    assert "does not apply" in capsys.readouterr().out
    assert FN.load_pick(tmp_path, "demo") is None


def test_an_unknown_record_is_named(tmp_path, monkeypatch):
    _workspace(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="'nope' not found"):
        cli.main(["final", "pick", "nope"])


# ---------------------------------------------------------------------------- after the deadline
def test_the_outcome_shows_where_the_recommended_set_would_have_ranked():
    pick = {"ids": ["0003_other", "0001_strong"], "when": "2026-11-20 10:00 UTC"}
    views = [{"id": "0001_strong", "private": 0.80}, {"id": "0002_twin", "private": 0.79},
             {"id": "0003_other", "private": 0.82}]
    data = {"leaderboard": {"shown": {"scores": [0.85, 0.83, 0.81, 0.80, 0.78], "ranks": [4]}}}
    line = FN.pick_lines(pick, views, data, final=0.80, greater=True)[0]
    assert "best private 0.82000" in line and "rank 3" in line and "0.02000 above the finals" in line
    assert FN.pick_lines(None, views, data, 0.8, True) == []
    lone = FN.pick_lines({"ids": ["0009_x"]}, views, data, 0.8, True)[0]
    assert "none of it has a private score" in lone


def test_a_time_series_is_resampled_in_time_blocks(tmp_path, monkeypatch, capsys):
    _workspace(tmp_path, monkeypatch, "timeseries")
    capsys.readouterr()
    cli.main(["final", "pick", "0002", "0003", "--draws", "200"])
    out = capsys.readouterr().out
    assert "contiguous time blocks (by row order" in out and "bootstrap" in out
