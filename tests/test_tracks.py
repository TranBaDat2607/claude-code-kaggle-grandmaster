"""Model lines as a tree (#133): a record is compared with its nearest kept ancestor, else the accepted baseline of
its track; a track is a name on a root (`--track cnn`), inherited by the records that branch from it; blends live in
the `ensemble` track, so a new blend is tested against the best accepted blend (the blend-gain test); `ledger tracks`
shows each line's baseline and recent gain, and says when the ensemble misses a newer baseline."""

import datetime as dt

import numpy as np
import pytest

from kgkit import cli
from kgkit import evalset as EV
from kgkit import state as S
from kgkit.experiment import Ledger


@pytest.fixture
def led(tmp_path):
    S.CompetitionState(slug="demo", metric="auc", deadline=(dt.date.today() + dt.timedelta(days=20)).isoformat()
                       ).save(tmp_path)
    return Ledger(tmp_path)


@pytest.fixture
def ws(tmp_path, monkeypatch, binary_df):
    monkeypatch.chdir(tmp_path)
    cli.main(["init", "demo", "--metric", "auc", "--target", "target", "--id-col", "id"])
    binary_df.to_csv(tmp_path / "data" / "train.csv", index=False)
    cli.main(["folds", "data/train.csv", "--target", "target", "--id-col", "id", "--out", "data/folds.csv"])
    return tmp_path


SPEC = ["--truth", "data/train.csv:target", "--folds", "data/folds.csv:fold"]


def _import(ws, binary_df, name, noise, seed, *extra):
    """A model whose OOF is the signal plus noise: less noise, a better model; another seed, a different one."""
    rng = np.random.default_rng(seed)
    signal = 1.5 * binary_df["x1"] - binary_df["x2"]
    np.save(ws / f"{name}.npy", signal + rng.normal(0, noise, len(signal)))
    np.save(ws / f"{name}_test.npy", rng.random(50))
    cli.main(["ledger", "import", name, f"{name}.npy", "--test", f"{name}_test.npy", *SPEC, *extra])
    return Ledger(ws).records()[-1]


# ----------------------------------------------------------------------------- the rule (evalset, stdlib)
def test_a_records_track_is_its_own_else_a_blends_else_its_parents_else_main(led):
    root = led.log("cnn", 0.80, track="cnn")
    child = led.log("cnn-512", 0.81, parent=root["id"])
    grandchild = led.log("cnn-512-tta", 0.82, parent=child["id"])
    blend = led.log("blend", 0.86, model="blend", parent=child["id"])
    plain = led.log("gbdt", 0.85)
    by_id = {r["id"]: r for r in led.records()}
    assert EV.track_of(by_id[grandchild["id"]], by_id) == "cnn"
    assert EV.track_of(by_id[blend["id"]], by_id) == "ensemble"
    assert EV.track_of(by_id[plain["id"]], by_id) == "main"
    # a parent cycle (a hand-edited ledger) ends the walk instead of looping
    led.update(root["id"], parent=grandchild["id"], track=None)
    by_id = {r["id"]: r for r in led.records()}
    assert EV.track_of(by_id[child["id"]], by_id) == "main"


def test_a_single_model_never_inherits_the_ensemble_line(led):
    """Before #133 a kept blend was the one baseline, so single models were decided against it (parent = the blend):
    they stay in their own line, and keep the main baseline theirs."""
    g = led.log("gbdt", 0.85)
    led.decide(g["id"], "baseline")
    b = led.log("blend", 0.87, model="blend")
    led.decide(b["id"], "keep", parent=g["id"])
    m = led.log("gbdt-fe", 0.86)
    led.decide(m["id"], "keep", parent=b["id"])
    by_id = {r["id"]: r for r in led.records()}
    assert EV.track_of(by_id[m["id"]], by_id) == "main"
    assert led.baseline()["id"] == m["id"] and led.baseline(track="ensemble")["id"] == b["id"]


def test_each_track_has_its_own_accepted_baseline(led):
    """Keeping a CNN no longer moves the baseline a new GBDT idea is compared with (the bug #133 is about)."""
    g = led.log("gbdt", 0.85)
    led.decide(g["id"], "baseline")
    c = led.log("cnn", 0.81, track="cnn")
    led.decide(c["id"], "baseline")
    assert led.baseline()["id"] == g["id"]
    assert led.baseline(track="cnn")["id"] == c["id"]
    assert {t: r["id"] for t, r in led.heads().items()} == {"main": g["id"], "cnn": c["id"]}
    assert EV.accepted_baseline(led.records())["id"] == g["id"]
    # a new GBDT idea without a parent is compared with the main line, a CNN one with the CNN line
    assert led.base_for(led.log("gbdt-fe", 0.86))["id"] == g["id"]
    assert led.base_for(led.log("cnn-aug", 0.82, track="cnn"))["id"] == c["id"]


def test_without_a_main_line_the_baseline_is_the_last_accepted_one(led):
    c = led.log("cnn", 0.81, track="cnn")
    led.decide(c["id"], "baseline")
    assert led.baseline()["id"] == c["id"]
    b = led.log("blend", 0.86, model="blend")
    led.decide(b["id"], "baseline")
    assert led.baseline()["id"] == c["id"]  # a blend is the primary baseline only when it is the only one
    assert led.baseline(track="ensemble")["id"] == b["id"]


def test_a_record_is_compared_with_its_nearest_kept_ancestor(led):
    a = led.log("a", 0.85)
    led.decide(a["id"], "baseline")
    b = led.log("b", 0.84, parent=a["id"])
    led.decide(b["id"], "discard", parent=a["id"])
    c = led.log("c", 0.86, parent=b["id"])  # built on a discarded idea: judged against what was accepted
    assert led.base_for(c)["id"] == a["id"]
    # a sibling branch: kept against a, while the main head is elsewhere
    d = led.log("d", 0.87, parent=a["id"])
    led.decide(d["id"], "keep", parent=a["id"])
    e = led.log("e", 0.86, parent=a["id"])
    assert led.base_for(e)["id"] == a["id"]
    # an ensemble member kept with --member is a kept node: what branches from it is compared with it
    m = led.log("nn", 0.80, parent=d["id"])
    led.decide(m["id"], "keep", parent=d["id"], member=True)
    n = led.log("nn-wider", 0.81, parent=m["id"])
    assert led.base_for(n)["id"] == m["id"]
    # no kept ancestor at all: the parent it was built from
    x = led.log("x", 0.5, track="tfm")
    y = led.log("y", 0.6, parent=x["id"])
    assert led.base_for(y)["id"] == x["id"]


def test_the_first_record_of_a_new_track_has_nothing_to_compare_with(led):
    g = led.log("gbdt", 0.85)
    led.decide(g["id"], "baseline")
    assert led.base_for(led.log("cnn", 0.8, track="cnn")) is None


def test_track_names_are_short_words(led):
    with pytest.raises(ValueError, match="track"):
        led.log("x", 0.5, track="two words")
    r = led.log("x", 0.5)
    with pytest.raises(ValueError, match="track"):
        led.decide(r["id"], "baseline", track="a/b")
    assert led.decide(r["id"], "baseline", track="vit-l")["track"] == "vit-l"


# ----------------------------------------------------------------------------- the commands
def test_exp_finish_starts_a_track_and_compares_its_runs_with_its_head(ws, binary_df, capsys):
    _import(ws, binary_df, "gbdt", 0.6, 1)
    cli.main(["exp", "finish", "0001", "baseline", "first", "gbdt"])
    _import(ws, binary_df, "cnn", 1.2, 2)
    capsys.readouterr()
    cli.main(["exp", "finish", "0002", "baseline", "first", "cnn", "--track", "cnn"])
    out = capsys.readouterr().out
    assert "accepted baseline of track cnn is now 0002_cnn" in out and "warning" not in out
    assert Ledger(ws).baseline()["id"] == "0001_gbdt" and Ledger(ws).get("0002")["track"] == "cnn"
    # a better CNN: compared with the CNN line, not the (better) GBDT
    _import(ws, binary_df, "cnn2", 0.8, 3, "--track", "cnn")
    capsys.readouterr()
    cli.main(["exp", "finish", "0003", "keep"])
    out = capsys.readouterr().out
    assert "0003_cnn2 vs 0002_cnn: KEEP" in out and "accepted baseline of track cnn is now 0003_cnn2" in out
    assert Ledger(ws).baseline()["id"] == "0001_gbdt"
    # the first run of a track with no baseline yet says how to start it
    _import(ws, binary_df, "tfm", 1.0, 4, "--track", "tfm")
    with pytest.raises(SystemExit, match="track tfm"):
        cli.main(["exp", "finish", "0004", "keep"])


def test_ledger_compare_defaults_to_the_records_own_baseline(ws, binary_df, capsys):
    _import(ws, binary_df, "gbdt", 0.6, 1)
    cli.main(["ledger", "decide", "0001", "baseline"])
    _import(ws, binary_df, "cnn", 1.2, 2, "--track", "cnn")
    cli.main(["ledger", "decide", "0002", "baseline", "--track", "cnn"])
    _import(ws, binary_df, "cnn2", 0.8, 3, "--track", "cnn")
    capsys.readouterr()
    cli.main(["ledger", "compare", "0003"])
    assert "0003_cnn2 vs 0002_cnn" in capsys.readouterr().out


def test_status_baseline_and_brief_list_one_baseline_per_track(ws, binary_df, capsys, plugin_root):
    _import(ws, binary_df, "gbdt", 0.6, 1)
    cli.main(["ledger", "decide", "0001", "baseline"])
    capsys.readouterr()
    cli.main(["ledger", "baseline"])
    single = capsys.readouterr().out
    assert single.startswith("0001_gbdt") and "track" not in single  # one line of models: as before
    _import(ws, binary_df, "cnn", 1.2, 2)
    cli.main(["ledger", "decide", "0002", "baseline", "--track", "cnn"])
    capsys.readouterr()
    cli.main(["ledger", "baseline"])
    out = capsys.readouterr().out
    assert out.startswith("0001_gbdt") and "cnn: 0002_cnn" in out
    cli.main(["ledger", "baseline", "--track", "cnn"])
    assert capsys.readouterr().out.startswith("0002_cnn")
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "accepted baseline: " in out and "0001_gbdt" in out and "track cnn: " in out and "0002_cnn" in out
    from tests.test_hooks import hook

    ctx = hook(plugin_root, "session_start.py", {"hook_event_name": "SessionStart", "cwd": str(ws)})["additionalContext"]
    assert "Accepted baseline (compare new ideas against this): 0001_gbdt" in ctx
    assert "track cnn 0002_cnn" in ctx


def test_a_blend_is_tested_against_the_best_accepted_blend(ws, binary_df, capsys):
    """The blend-gain test: a model is worth keeping for the ensemble when the blend with it beats the blend
    without it, paired on the same folds; nested weights, so adding a member cannot win by fitting the OOFs."""
    _import(ws, binary_df, "gbdt", 0.8, 1)
    cli.main(["ledger", "decide", "0001", "baseline"])
    _import(ws, binary_df, "nn", 0.8, 2, "--track", "nn")
    cli.main(["ledger", "decide", "0002", "baseline"])
    capsys.readouterr()
    cli.main(["blend", "--exp", "0001", "0002", *SPEC, "--name", "b1"])
    out = capsys.readouterr().out
    b1 = Ledger(ws).get("0003")
    assert b1["model"] == "blend" and b1["params"]["members"] == ["0001_gbdt", "0002_nn"]
    assert b1["params"]["oof"] == "nested" and isinstance(b1["params"]["in_sample_cv"], float)
    assert "no accepted blend yet" in out and "kgkit exp finish 0003_b1 baseline" in out
    cli.main(["exp", "finish", "0003", "baseline", "first", "blend"])
    out = capsys.readouterr().out
    assert "accepted baseline of track ensemble is now 0003_b1" in out
    assert Ledger(ws).baseline()["id"] == "0001_gbdt"  # a kept blend is the ensemble's baseline, not the GBDT's
    _import(ws, binary_df, "cnn", 0.8, 3, "--track", "cnn")
    cli.main(["ledger", "decide", "0004", "baseline"])
    capsys.readouterr()
    cli.main(["blend", "--exp", "0001", "0002", "0004", *SPEC, "--name", "b2"])
    out = capsys.readouterr().out
    b2 = Ledger(ws).get("0005")
    assert b2["parent"] == "0003_b1" and EV.track_of(b2, {r["id"]: r for r in Ledger(ws).records()}) == "ensemble"
    assert "0005_b2 vs 0003_b1:" in out and "kgkit exp finish 0005_b2" in out
    # without --folds the weights are fit on the OOFs they are scored on: said, not hidden
    capsys.readouterr()
    cli.main(["blend", "--exp", "0001", "0002", "--truth", "data/train.csv:target", "--name", "b3"])
    out = capsys.readouterr().out
    assert "optimistic" in out and "--folds" in out


def test_ledger_tracks_shows_each_lines_gain_and_a_stale_ensemble(ws, binary_df, capsys):
    _import(ws, binary_df, "gbdt", 1.0, 1)
    cli.main(["ledger", "decide", "0001", "baseline"])
    _import(ws, binary_df, "gbdt2", 0.7, 1)
    cli.main(["exp", "finish", "0002", "keep"])
    _import(ws, binary_df, "cnn", 1.0, 2, "--track", "cnn")
    cli.main(["ledger", "decide", "0003", "baseline"])
    capsys.readouterr()
    cli.main(["ledger", "tracks"])
    out = capsys.readouterr().out
    assert "main" in out and "0002_gbdt2" in out and "1 kept of 1 compared" in out
    assert "cnn" in out and "0003_cnn" in out
    assert "no accepted blend" in out and "kgkit blend --exp 0002_gbdt2 0003_cnn" in out
    cli.main(["blend", "--exp", "0002", "0003", *SPEC, "--name", "b1"])
    cli.main(["exp", "finish", "0004", "baseline"])
    _import(ws, binary_df, "cnn2", 0.7, 2, "--track", "cnn")
    cli.main(["exp", "finish", "0005", "keep"])
    capsys.readouterr()
    cli.main(["ledger", "tracks"])
    out = capsys.readouterr().out
    assert "ensemble" in out and "0004_b1" in out
    assert "0005_cnn2" in out and "not in the ensemble" in out and "kgkit blend --exp 0002_gbdt2 0005_cnn2" in out
    # the same line in the session brief, so a grind re-blends when a track improves
    from kgkit import finals as FN

    honest = dict(FN._honest(S.load(ws), {r["id"]: r for r in Ledger(ws).records()}, {}))
    assert any("ensemble" in why for why in honest["0004_b1"])


def test_member_points_to_the_blend_gain_test(ws, binary_df, capsys):
    _import(ws, binary_df, "gbdt", 0.6, 1)
    cli.main(["ledger", "decide", "0001", "baseline"])
    _import(ws, binary_df, "nn", 1.2, 2)
    capsys.readouterr()
    cli.main(["ledger", "decide", "0002", "keep", "--member"])
    out = capsys.readouterr().out
    assert "0002_nn: keep (member)" in out and "kgkit blend" in out


# ----------------------------------------------------------------------------- the templates
def test_a_template_run_joins_its_track_and_is_compared_inside_it(tmp_path, plugin_root):
    from tests.test_templates import _make_ws, _run

    ws = _make_ws(tmp_path, plugin_root, "binary")
    run = ["src/train_gbdt.py", "--model", "hgb"]
    _run([*run, "--name", "base", "--params", '{"max_iter": 30}'], ws, plugin_root)
    _run(["-m", "kgkit", "exp", "finish", "0001", "baseline", "gbdt"], ws, plugin_root)
    out = _run([*run, "--name", "lin", "--model", "linear", "--track", "linear"], ws, plugin_root)
    assert Ledger(ws).get("0002")["track"] == "linear"
    assert "no accepted baseline in track linear yet" in out and "kgkit exp finish 0002_lin baseline" in out
    _run(["-m", "kgkit", "exp", "finish", "0002", "baseline", "first", "linear"], ws, plugin_root)
    out = _run([*run, "--name", "lin2", "--model", "linear", "--track", "linear"], ws, plugin_root)
    assert "0003_lin2 vs 0002_lin:" in out


@pytest.mark.parametrize("script", ["train_gbdt.py", "train_image.py", "train_transformer.py"])
def test_every_training_template_takes_a_track(plugin_root, script):
    src = (plugin_root / "templates" / script).read_text(encoding="utf-8")
    assert '"--track"' in src and "track=" in src
