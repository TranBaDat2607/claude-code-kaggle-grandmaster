"""Simulation competitions (#134, #136): an agent's strength is its score over matches. The control and every variant
play one fixed schedule (opponent, seed, seat), so their results pair match by match and most of a game's randomness
cancels; the comparison resamples whole opponents. A screen plays fewer games of every opponent."""

import numpy as np
import pandas as pd
import pytest

from kgkit import cli
from kgkit import screen as SC
from kgkit import units as U
from kgkit.experiment import Ledger


def test_schedule_is_fixed_mirrors_each_seed_and_a_shorter_one_is_its_prefix():
    s = U.schedule(["bot_a", "bot_b"], games=6, seats=2, seed=0)
    assert len(s) == 12 and len({f["id"] for f in s}) == 12
    a = [f for f in s if f["opponent"] == "bot_a"]
    assert [f["seat"] for f in a] == [0, 1, 0, 1, 0, 1] and [f["seed"] for f in a] == [0, 0, 1, 1, 2, 2]
    assert U.schedule(["bot_a", "bot_b"], games=6, seats=2, seed=0) == s
    short = U.schedule(["bot_a", "bot_b"], games=2, seats=2, seed=0)
    assert {f["id"] for f in short} <= {f["id"] for f in s}
    assert a[0]["id"] == U.fixture_id("bot_a", 0, 0)


def test_matches_turn_results_into_a_unit_table_grouped_by_opponent():
    rows = [{"opponent": "a", "seed": 1, "seat": 0, "result": "win"},
            {"opponent": "a", "seed": 1, "seat": 1, "result": "draw"},
            {"opponent": "b", "seed": 1, "seat": 0, "result": "loss"},
            {"opponent": "b", "seed": 1, "seat": 1, "result": 1}]
    u = U.matches(rows)
    assert u.agg == "mean" and u.greater_is_better is True
    assert u.score() == pytest.approx((1 + 0.5 + 0 + 1) / 4)
    assert list(u.group) == ["a", "a", "b", "b"] and u.ids[0] == U.fixture_id("a", 1, 0)
    assert U.matches(pd.DataFrame(rows)).ids.tolist() == u.ids.tolist()
    assert U.matches([{**rows[0], "result": 37.5}]).score() == 37.5  # a reward, not a win/draw/loss
    with pytest.raises(ValueError, match="result"):
        U.matches([{**rows[0], "result": "maybe"}])
    with pytest.raises(ValueError, match="twice|unique|duplicate"):
        U.matches([rows[0], rows[0]])


def _play(fixtures, p_win, strength, draw):
    """Common random numbers: a fixture's draw is the same for every agent (same seed, same seat, same game)."""
    return [{**f, "result": "win" if draw[f["id"]] < p_win - strength[f["opponent"]] else "loss"} for f in fixtures]


def test_the_same_fixtures_pair_the_comparison():
    ops = [f"bot{i}" for i in range(8)]
    fx = U.schedule(ops, games=40, seed=3)
    rng = np.random.default_rng(0)
    draw = {f["id"]: rng.random() for f in fx}
    strength = dict(zip(ops, np.linspace(-0.2, 0.2, len(ops))))
    a, b = U.matches(_play(fx, 0.55, strength, draw)), U.matches(_play(fx, 0.50, strength, draw))
    res = U.compare(a, b, seed=0)
    assert res["gain"] == pytest.approx(0.05, abs=0.03) and res["unit"] == "groups"
    unpaired = np.sqrt(a.cols["score"].var() / len(fx) + b.cols["score"].var() / len(fx))
    assert res["se"] < 0.5 * unpaired  # most of the game's randomness cancels


def test_a_screen_plays_fewer_games_of_every_opponent_and_compares_with_its_control(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for k in ("KG_SCREEN", "KG_FIDELITY", "KG_CONTROL_OF"):
        monkeypatch.delenv(k, raising=False)
    cli.main(["init", "arena", "--metric", "accuracy", "--target", "result"])
    ops = [f"bot{i}" for i in range(6)]
    fx = pd.DataFrame(U.schedule(ops, games=50))
    fx.to_csv("fixtures.csv", index=False)
    cli.main(["screen", "new", "g10", "--ids", "fixtures.csv:id", "--frac", "0.2", "--stratify", "opponent"])
    s = SC.load(tmp_path, "g10")
    assert s.strategy == "stratified:opponent"
    assert fx[s.mask(fx["id"])].groupby("opponent").size().tolist() == [10] * len(ops)

    rng = np.random.default_rng(1)
    draw = {i: rng.random() for i in fx["id"]}
    strength = dict(zip(ops, np.linspace(-0.2, 0.2, len(ops))))
    led = Ledger(tmp_path)
    base = led.log_units("v1", U.matches(_play(fx.to_dict("records"), 0.5, strength, draw)))
    led.decide(base["id"], "baseline")
    monkeypatch.setenv("KG_SCREEN", "g10")
    played = fx[SC.active().mask(fx["id"])].to_dict("records")  # the arena plays only the screen's fixtures
    monkeypatch.setenv("KG_CONTROL_OF", base["id"])
    ctl = led.log_units("v1_ctl", U.matches(_play(played, 0.5, strength, draw)))
    monkeypatch.delenv("KG_CONTROL_OF")
    v2 = led.log_units("v2", U.matches(_play(played, 0.8, strength, draw)), parent=base["id"])
    capsys.readouterr()
    cli.main(["ledger", "compare", v2["id"]])
    out = capsys.readouterr().out
    assert f"{v2['id']} vs {ctl['id']}" in out and "KEEP" in out and "never kept" in out
