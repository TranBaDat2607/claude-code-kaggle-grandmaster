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
    truth = pd.DataFrame(y, columns=COLS).assign(study_id=ids)[["study_id", *COLS]]
    truth.to_csv(tmp_path / "holdout.csv", index=False)
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


# ----------------------------------------------------------------------------- phase 2: one estimator, cached noise
def test_paired_noise_is_far_below_the_noise_of_one_score():
    from kgkit.compare import estimate

    y, a, b, _ = _multilabel(np.random.default_rng(0))
    paired = estimate(y, a, b, "mean_col_auc", n_boot=300)
    single = estimate(y, a, None, "mean_col_auc", n_boot=300)
    assert paired["question"] == "paired" and single["question"] == "absolute" and "gain" not in single
    assert paired["se"] < single["se"] / 2  # the 58-study case of #32: ~0.003 vs ~0.013
    p = single["value"]
    assert single["se"] < (p * (1 - p) / N) ** 0.5 / 2  # and the binomial formula overstates even one score
    assert any("small evaluation set" in c for c in paired["caveats"])


def test_every_resample_keeps_the_label_set_of_the_full_set():
    from kgkit.compare import estimate

    rng = np.random.default_rng(1)
    y = (rng.random((N, 3)) < [0.5, 0.4, 0.05]).astype(int)
    y[:2, 2] = 1  # a rare label: 2-4 positives
    seen = []

    def strict(yy, pp):
        seen.append(int((yy.min(0) != yy.max(0)).sum()))
        return float(np.mean(pp[yy == 1]) - np.mean(pp[yy == 0]))

    res = estimate(y, rng.random((N, 3)), None, strict, greater_is_better=True, n_boot=200)
    full = int((y.min(0) != y.max(0)).sum())
    assert res["n_redrawn"] > 0 and res["n_boot"] == 200 and set(seen) == {full}


def test_projection_to_another_sample_size():
    from kgkit.compare import describe, estimate

    rng = np.random.default_rng(2)
    n = 2000
    y = rng.integers(0, 2, n)
    a = y + rng.normal(0, 1.0, n)
    b = a + rng.normal(0, 0.3, n)
    here = estimate(y, a, b, "auc", n_boot=300)
    lb = estimate(y, a, b, "auc", n_boot=300, size=4 * n)
    assert 0.35 < lb["se"] / here["se"] < 0.7  # ~ sqrt(1/4)
    assert "at n=8000" in describe(lb) and any("projected to 8000 rows" in c for c in lb["caveats"])


def test_groups_are_resampled_whole():
    from kgkit.compare import estimate

    rng = np.random.default_rng(4)
    g = np.repeat(np.arange(300), 4)  # 4 images per patient, patient-level effects
    eff = rng.normal(0, 1, 300)[g]
    y = (eff + rng.normal(0, 0.5, len(g)) > 0).astype(int)
    a = eff + rng.normal(0, 1, len(g))
    rows = estimate(y, a, None, "auc", n_boot=300)
    grouped = estimate(y, a, None, "auc", n_boot=300, groups=g)
    assert grouped["unit"] == "groups" and grouped["n_units"] == 300 and grouped["se"] > rows["se"]


def test_status_and_hook_read_the_measured_paired_noise(hws, capsys, plugin_root):
    import shutil

    cols = ",".join(COLS)
    _cli("ledger", "add", "a", "auto", "--pred", f"pred_a.csv:{cols}")
    _cli("ledger", "decide", "1", "baseline")
    _cli("status")
    assert "not measured yet" in capsys.readouterr().out
    _cli("ledger", "add", "b", "auto", "--pred", f"pred_b.csv:{cols}", "--boot", "200", "--public-n", "600")
    out = capsys.readouterr().out
    assert "0002_b vs 0001_a" in out and "paired SE of the difference" in out and "public LB (n=600)" in out
    entry = EV.comparisons(hws)[-1]
    assert entry["new"] == "0002_b" and entry["eval"] == EV.spec_hash(hws, SPEC) and entry["se_public"] < entry["se"]
    _cli("status")
    out = capsys.readouterr().out
    assert f"paired SE of differences ~{entry['se']:.4f}" in out and "not measured yet" not in out
    _cli("ledger", "noise", "1", "--boot", "200", "--public-n", "600")
    out = capsys.readouterr().out
    assert "SE of the score" in out and "CV-LB gap within" in out and "(paired), not with this" in out
    sh = shutil.which("sh")
    if sh is None:
        return
    r = subprocess.run([sh, str(plugin_root / "hooks" / "run.sh"), "session_start.py"],
                       input=json.dumps({"hook_event_name": "SessionStart", "cwd": str(hws)}),
                       capture_output=True, text=True, timeout=60)
    ctx = json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "measured paired SE of differences" in ctx


def test_kfold_compare_without_oof_still_uses_fold_scores(tmp_path, monkeypatch, capsys):
    pd.DataFrame({"y": [0, 1] * 10}).to_csv(tmp_path / "train.csv", index=False)
    S.CompetitionState(slug="x", metric="auc", eval_truth="train.csv:y").save(tmp_path)
    led = Ledger(tmp_path)
    led.log("base", 0.80, fold_scores=[0.80, 0.84, 0.78, 0.83, 0.81], decision="baseline")
    led.log("new", 0.803, fold_scores=[0.803, 0.843, 0.783, 0.833, 0.813])
    monkeypatch.chdir(tmp_path)
    _cli("ledger", "compare", "2")
    assert "KEEP" in capsys.readouterr().out
    assert EV.comparisons(tmp_path)[-1]["method"] == "fold-paired t"


# ----------------------------------------------------------------------------- phase 3: fast paths, seeds, ROPE
@pytest.mark.parametrize("metric", ["logloss", "rmse", "auc", "mean_col_auc", "accuracy", "map@5"])
def test_influence_clt_agrees_with_the_bootstrap(metric):
    from kgkit.compare import estimate

    rng = np.random.default_rng(5)
    n = 3000
    if metric == "mean_col_auc":
        y, a, b, _ = _multilabel(rng, n)
    elif metric == "map@5":
        y = [list(rng.choice(30, 3, replace=False)) for _ in range(n)]
        a = [list(t[:2]) + list(rng.permutation(30)[:10]) for t in y]
        b = [list(t[:1]) + list(rng.permutation(30)[:11]) for t in y]
    else:
        y = rng.integers(0, 2, n) if metric != "rmse" else rng.normal(size=n)
        s = y + rng.normal(0, 1.0, n)
        a, b = s, s + rng.normal(0, 0.4, n)
        if metric == "logloss":
            a, b = 1 / (1 + np.exp(-a)), 1 / (1 + np.exp(-b))
        if metric == "accuracy":
            a, b = (a > 0.5).astype(int), (b > 0.5).astype(int)
    clt = estimate(y, a, b, metric, method="clt")
    boot = estimate(y, a, b, metric, method="bootstrap", n_boot=300)
    assert clt["method"] == "clt" and clt["n_boot"] == 0
    assert np.isclose(clt["gain"], boot["gain"])
    assert 0.75 < clt["se"] / boot["se"] < 1.33, (clt["se"], boot["se"])
    assert estimate(y, a, b, metric)["method"] == "clt"  # auto: >= 1000 units and an influence function


def test_clt_is_cluster_robust_and_projects():
    from kgkit.compare import estimate

    rng = np.random.default_rng(6)
    g = np.repeat(np.arange(500), 4)
    eff = rng.normal(0, 1, 500)[g]
    y = (eff + rng.normal(0, 0.5, len(g)) > 0).astype(int)
    a = eff + rng.normal(0, 1, len(g))
    rows = estimate(y, a, None, "auc", method="clt")
    grouped = estimate(y, a, None, "auc", method="clt", groups=g)
    boot = estimate(y, a, None, "auc", method="bootstrap", n_boot=300, groups=g)
    assert grouped["se"] > 1.15 * rows["se"] and 0.8 < grouped["se"] / boot["se"] < 1.25
    lb = estimate(y, a, None, "auc", method="clt", groups=g, size=8 * len(g))
    assert np.isclose(lb["se"], grouped["se"] / np.sqrt(8))


def test_bayesian_bootstrap_and_fallbacks():
    from kgkit.compare import estimate

    y, a, b, _ = _multilabel(np.random.default_rng(7))
    bayes = estimate(y, a, b, "mean_col_auc", method="bayes", n_boot=300)
    boot = estimate(y, a, b, "mean_col_auc", method="bootstrap", n_boot=300)
    assert bayes["method"] == "bayesian bootstrap" and bayes["n_redrawn"] == 0
    assert 0.6 < bayes["se"] / boot["se"] < 1.6
    rng = np.random.default_rng(8)
    yr = rng.normal(size=200)
    sp = estimate(yr, yr + rng.normal(size=200), None, "spearman", method="bayes", n_boot=100)
    assert sp["method"] == "bootstrap" and any("takes no sample weights" in c for c in sp["caveats"])
    custom = estimate(yr, yr + rng.normal(size=200), None, lambda t, p: float(np.mean(t - p)), greater_is_better=True,
                      method="clt", n_boot=100)
    assert custom["method"] == "bootstrap" and any("no influence function" in c for c in custom["caveats"])
    assert estimate(yr, yr, None, "rmse", n_boot=50)["method"] == "bootstrap"  # auto below 1000 units


def test_rope_marks_practically_equivalent_changes():
    from kgkit.compare import estimate, verdict

    rng = np.random.default_rng(9)
    n = 5000
    y = rng.integers(0, 2, n)
    a = y + rng.normal(0, 1.0, n)
    b = a + rng.normal(0, 0.05, n)  # a negligible change
    res = estimate(y, b, a, "auc", method="clt", rope=0.005)
    assert res["p_rope"] > 0.95 and np.isclose(res["p_rope"] + res["p_above_rope"] + res["p_below_rope"], 1)
    v = verdict(None, res)
    assert v["decision"] == "INCONCLUSIVE" and any("practically equivalent" in r for r in v["reasons"])
    big = estimate(y, a + rng.normal(0, 0.01, n), a + rng.normal(0, 2.0, n), "auc", n_boot=200, rope=0.005)
    assert big["p_above_rope"] > 0.95 and verdict(None, big)["decision"] == "KEEP"


def test_seed_noise_is_pooled_from_re_runs_and_added():
    from kgkit.compare import estimate, n_seeds, replica_key, seed_variance, with_seed_noise

    recs = [{"cv": c, "variant": v, "tags": []} for c, v in
            [(0.900, "a"), (0.904, "a"), (0.902, "a"), (0.910, "b"), (0.914, "b"), (0.95, None)]]
    sv = seed_variance(recs)
    assert sv["df"] == 3 and sv["n_variants"] == 2 and 0.001 < sv["sd"] < 0.004
    assert replica_key({"params": {"lr": 0.1, "seed": 1}, "git_commit": "abc", "git_dirty": False}) == \
        replica_key({"params": {"lr": 0.1, "seed": 2}, "git_commit": "abc", "git_dirty": False})
    assert replica_key({"params": {"lr": 0.1}, "git_commit": "abc", "git_dirty": True}) is None
    assert n_seeds({"seeds": [1, 2, 3]}) == 3 and n_seeds({}) == 1
    y, a, b, _ = _multilabel(np.random.default_rng(10))
    res = estimate(y, a, b, "mean_col_auc", n_boot=200)
    total = with_seed_noise(res, sv)
    assert total["se"] > res["se"] and total["se_rows"] == res["se"] and "seed noise" in total["method"]


def test_cli_compare_with_seed_replicas_rope_and_clt(hws, capsys):
    cols = ",".join(COLS)
    _cli("ledger", "add", "a", "auto", "--pred", f"pred_a.csv:{cols}", "--variant", "A")
    _cli("ledger", "decide", "1", "baseline")
    _cli("ledger", "add", "b", "auto", "--pred", f"pred_b.csv:{cols}", "--variant", "A", "--boot", "100")
    out = capsys.readouterr().out
    assert "seed noise (sd" in out  # a and b share a variant: their spread is taken as seed noise
    _cli("ledger", "compare", "2", "1", "--method", "clt", "--rope", "0.01")
    out = capsys.readouterr().out
    assert "influence-function CLT" in out and "ROPE +/-0.01" in out
    _cli("config", "rope=0.002", "public_lb_n=600")
    assert S.load(hws).rope == 0.002 and S.load(hws).public_lb_n == 600
