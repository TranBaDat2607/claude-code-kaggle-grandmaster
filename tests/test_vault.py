"""The sealed vault split (#141): held-out rows whose labels leave the workspace, predicted like the test set by every
pipeline, and checked pass/fail only from a fixed budget before a final candidate."""

import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from kgkit import cli
from kgkit import cv as CV
from kgkit import finals as FN
from kgkit import state as S
from kgkit import vault as V
from kgkit.experiment import Ledger


def _data(n=4000, seed=0, groups=40000):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * x1 - x2)))).astype(int)
    return pd.DataFrame({"id": np.arange(n), "x1": x1, "x2": x2, "grp": rng.integers(0, groups, size=n),
                         "day": rng.integers(0, 200, size=n), "target": y})


def _init(tmp_path, monkeypatch, df):
    monkeypatch.chdir(tmp_path)
    cli.main(["init", "demo", "--metric", "auc", "--target", "target", "--id-col", "id"])
    df.to_csv(tmp_path / "data" / "train.csv", index=False)
    return tmp_path


FOLDS = ["folds", "data/train.csv", "--target", "target", "--id-col", "id", "--out", "data/folds.csv"]


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    df = _data()
    ws = _init(tmp_path, monkeypatch, df)
    cli.main([*FOLDS, "--group", "grp", "--vault", "0.1"])
    return ws, df


# ---------------------------------------------------------------------------- the split
def test_the_vault_fold_is_in_neither_side_of_any_fold():
    folds = np.array([0, 1, 0, 1, -1, CV.VAULT, CV.VAULT])
    for _, tr, va in CV.split_indices(folds):
        assert not set(np.flatnonzero(folds == CV.VAULT)) & (set(tr) | set(va))
        assert 4 in tr and 4 not in va  # fold -1 is still trained on, never scored
    assert CV.VAULT == V.VAULT


def test_folds_with_a_vault_moves_the_rows_and_their_labels_out(sealed):
    ws, df = sealed
    m = V.meta(ws)
    assert m["state"] == "sealed" and m["unit"] == "groups" and m["budget"] == 20
    train, vault = pd.read_csv(ws / "data/train.csv"), pd.read_csv(ws / V.FEATURES)
    folds = pd.read_csv(ws / "data/folds.csv")
    assert len(train) + len(vault) == len(df) and 0.07 < len(vault) / len(df) < 0.13
    assert "target" not in vault.columns and list(train.columns) == list(df.columns)
    assert set(vault["id"]).isdisjoint(train["id"])
    assert set(folds.loc[folds["fold"] == V.VAULT, "id"]) == set(vault["id"])
    assert set(folds.loc[folds["fold"] >= 0, "id"]) == set(train["id"])
    assert set(vault["grp"]).isdisjoint(train["grp"])  # whole groups
    assert train["id"].is_monotonic_increasing  # the visible rows keep their order
    rows = pd.read_csv(V.store(m) / V.ROWS)  # the labels live outside the workspace only
    assert not str(V.store(m)).startswith(str(ws)) and set(rows["id"]) == set(vault["id"])
    assert m["vault_hash"] == V.vault_hash(vault["id"])
    spec = json.loads((ws / ".kaggle-gm/folds_spec.json").read_text())
    assert spec["vault"]["frac"] == 0.1 and spec["group"] == "grp"


def test_the_vault_follows_time_when_asked(tmp_path, monkeypatch):
    df = _data()
    ws = _init(tmp_path, monkeypatch, df)
    cli.main([*FOLDS, "--vault", "0.1", "--time-col", "day"])
    vault, train = pd.read_csv(ws / V.FEATURES), pd.read_csv(ws / "data/train.csv")
    assert vault["day"].min() > train["day"].max()


def test_little_data_measures_before_sealing(tmp_path, monkeypatch, capsys):
    ws = _init(tmp_path, monkeypatch, _data(n=1500))
    cli.main([*FOLDS, "--vault", "0.1"])
    out = capsys.readouterr().out
    assert "SMALL DATA" in out and "kgkit vault size" in out and "Ask the user" in out
    assert V.meta(ws) is None and not (ws / V.FEATURES).exists()


def test_a_vault_needs_an_id_column(tmp_path, monkeypatch, capsys):
    ws = _init(tmp_path, monkeypatch, _data())
    cli.main(["folds", "data/train.csv", "--target", "target", "--out", "data/folds.csv", "--vault", "0.1"])
    assert "needs an id column" in capsys.readouterr().out and V.meta(ws) is None


def test_plain_folds_never_seal(tmp_path, monkeypatch):
    ws = _init(tmp_path, monkeypatch, _data())
    cli.main(FOLDS)
    assert V.meta(ws) is None and len(pd.read_csv(ws / "data/train.csv")) == 4000


def test_sealing_over_logged_records_needs_a_reset_and_retires_them(tmp_path, monkeypatch):
    ws = _init(tmp_path, monkeypatch, _data())
    cli.main(FOLDS)
    led = Ledger(ws)
    led.log("old", 0.8, decision="baseline")
    with pytest.raises(SystemExit, match="reset-comparisons"):
        cli.main(["vault", "seal"])
    cli.main(["vault", "seal", "--reset-comparisons"])
    r = led.get("0001")
    assert r["pre_vault"] and led.best() == [] and led.baseline(fallback=False) is None
    assert V.meta(ws)["pre_vault_records"] == 1


def test_folds_again_keeps_the_vault(sealed):
    ws, _ = sealed
    vault_ids = set(pd.read_csv(ws / V.FEATURES)["id"])
    cli.main([*FOLDS, "--group", "grp", "--seed", "7"])
    folds = pd.read_csv(ws / "data/folds.csv")
    assert set(folds.loc[folds["fold"] == V.VAULT, "id"]) == vault_ids and V.sealed(ws)


def test_recheck_redraw_keeps_the_vault(sealed):
    from kgkit import recheck

    ws, _ = sealed
    spec = json.loads((ws / ".kaggle-gm/folds_spec.json").read_text())
    out = pd.read_csv(recheck.redraw(ws, spec, seed=7))
    vault_ids = set(pd.read_csv(ws / V.FEATURES)["id"])
    assert set(out.loc[out["fold"] == V.VAULT, "id"]) == vault_ids
    assert (out["fold"] >= 0).sum() == len(pd.read_csv(ws / "data/train.csv"))


# ---------------------------------------------------------------------------- checking
def _vault_truth(ws):
    m = V.meta(ws)
    return pd.read_csv(V.store(m) / V.ROWS)["target"].to_numpy()


def _log(led, name, pred, parent=None, decision=None):
    return led.log(name, 0.8, vault_pred=pred, parent=parent, decision=decision)


def test_a_real_gain_passes_and_noise_fails_and_neither_shows_a_score(sealed, capsys):
    ws, _ = sealed
    y = _vault_truth(ws)
    rng = np.random.default_rng(1)
    weak = 0.3 * y + rng.normal(0, 1, len(y))
    led = Ledger(ws)
    _log(led, "root", weak, decision="baseline")
    _log(led, "strong", 2.0 * y + rng.normal(0, 1, len(y)), parent="0001")
    _log(led, "same", weak + rng.normal(0, 1e-3, len(y)), parent="0001")
    capsys.readouterr()
    cli.main(["vault", "check", "0002"])
    cli.main(["vault", "check", "0003"])
    out = capsys.readouterr().out
    assert "0002_strong vs 0001_root: PASS" in out and "0003_same vs 0001_root: FAIL" in out
    assert re.findall(r"\d+\.\d+", out) == []  # no score, z or gain: only ids and the checks left
    assert led.get("0002")["vault"] == "pass" and led.get("0003")["vault"] == "fail"
    assert V.left(V.meta(ws)) == 18


def test_asking_again_is_free_and_the_budget_runs_out(sealed, capsys):
    ws, _ = sealed
    m = V.meta(ws)
    m["budget"] = 2
    V._save_meta(ws, m)
    y = _vault_truth(ws)
    led = Ledger(ws)
    _log(led, "root", y * 0.0 + 0.5, decision="baseline")
    for k in range(3):
        _log(led, f"v{k}", y + k, parent="0001")
    assert V.check(ws, "0002")["left"] == 1
    again = V.check(ws, "0002")
    assert again["cached"] and again["left"] == 1
    V.check(ws, "0003")
    with pytest.raises(SystemExit, match="no vault checks left"):
        V.check(ws, "0004")
    assert "NO CHECKS LEFT" in V.status_line(ws, led.records())


def test_a_check_needs_vault_predictions_of_this_vault(sealed):
    ws, _ = sealed
    led = Ledger(ws)
    led.log("no-vault", 0.8, decision="baseline")
    _log(led, "child", np.zeros(V.meta(ws)["n_rows"]), parent="0001")
    with pytest.raises(SystemExit, match="no ancestor that predicted the vault"):
        V.check(ws, "0002")
    with pytest.raises(SystemExit, match="has no vault predictions"):
        V.check(ws, "0002", against="0001")
    led.update("0002", vault_hash="other")
    _log(led, "root2", np.zeros(V.meta(ws)["n_rows"]))
    with pytest.raises(SystemExit, match="predicted another vault"):
        V.check(ws, "0002", against="0003")


def test_the_vault_rule_lets_noise_through_rarely_and_real_gains_often():
    """A pipeline picked as the best of many seed-only variants on the folds has no real gain: the vault, which it
    never saw, passes it about as rarely as one-sided z >= 2 allows (~2%). A real gain passes almost always."""
    rng = np.random.default_rng(0)
    n, trials = 400, 120
    noise_pass = real_pass = 0
    for _ in range(trials):
        y = rng.integers(0, 2, n)
        base = y * 0.8 + rng.normal(0, 1, n)
        lucky = base + rng.normal(0, 0.3, n)  # the same model, another seed: no gain on unseen rows
        better = y * 1.6 + rng.normal(0, 1, n)
        noise_pass += V.judge(y, lucky, base, None, "auc", True)
        real_pass += V.judge(y, better, base, None, "auc", True)
    assert noise_pass / trials < 0.07
    assert real_pass / trials > 0.9


def test_a_candidate_worse_than_the_baseline_fails():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 600)
    root = rng.normal(0, 1, 600)
    cand = y * 1.0 + rng.normal(0, 1, 600)
    best = y * 3.0 + rng.normal(0, 1, 600)
    assert V.judge(y, cand, root, None, "auc", True)
    assert not V.judge(y, cand, root, best, "auc", True)


# ---------------------------------------------------------------------------- noise before sealing
def test_vault_size_measures_each_fraction_and_retires_the_probes(tmp_path, monkeypatch, capsys):
    df = _data(n=1500)
    ws = _init(tmp_path, monkeypatch, df)
    cli.main(FOLDS)
    rng = np.random.default_rng(3)
    y = df["target"].to_numpy()
    led = Ledger(ws)
    led.log("gbdt", 0.8, oof=y * 1.2 + rng.normal(0, 1, len(y)))
    led.log("linear", 0.7, oof=y * 0.6 + rng.normal(0, 1, len(y)))
    capsys.readouterr()
    cli.main(["vault", "size", "--exp", "0001", "--vs", "0002"])
    out = capsys.readouterr().out
    assert "| 10% |" in out and "| 30% |" in out and "VERDICT" in out
    assert all(r.get("pre_vault") for r in led.records())
    cli.main(["vault", "seal", "--frac", "0.2"])  # the probes are retired: no reset needed
    assert V.sealed(ws) and V.meta(ws)["frac"] == 0.2


# ---------------------------------------------------------------------------- ledger, blend, status, finals
def test_ledger_add_vault_pred_aligns_by_id(sealed):
    ws, _ = sealed
    vault = pd.read_csv(ws / V.FEATURES)
    pd.DataFrame({"id": vault["id"][::-1], "p": np.arange(len(vault))[::-1]}).to_csv(ws / "vp.csv", index=False)
    cli.main(["ledger", "add", "custom", "0.8", "--vault-pred", "vp.csv:p"])
    rec = Ledger(ws).get("0001")
    assert np.array_equal(Ledger(ws).load_vault(rec["id"]), np.arange(len(vault)))
    assert rec["vault_hash"] == V.meta(ws)["vault_hash"]


def test_a_blend_of_members_that_predicted_the_vault_predicts_it_too(sealed):
    ws, _ = sealed
    n_tr, n_v = len(pd.read_csv(ws / "data/train.csv")), V.meta(ws)["n_rows"]
    rng = np.random.default_rng(4)
    led = Ledger(ws)
    y = pd.read_csv(ws / "data/train.csv")["target"].to_numpy()
    for k in range(2):
        led.log(f"m{k}", 0.8, oof=y + rng.normal(0, 1, n_tr), test_pred=rng.random(50), vault_pred=rng.random(n_v))
    cli.main(["blend", "--exp", "0001", "0002", "--truth", "data/train.csv:target", "--method", "rank", "--name", "bl"])
    rec = Ledger(ws).records()[-1]
    assert rec["model"] == "blend" and len(Ledger(ws).load_vault(rec["id"])) == n_v
    assert len(Ledger(ws).load_test(rec["id"])) == 50


def test_status_and_the_final_gate(sealed, capsys):
    ws, _ = sealed
    y = _vault_truth(ws)
    led = Ledger(ws)
    _log(led, "root", y * 0.0 + 0.5, decision="baseline")
    _log(led, "good", y.astype(float), parent="0001", decision="keep")
    st = S.load(ws)
    st.final_picks = ["0002"]
    st.save(ws)
    gate = V.gate(V.meta(ws))
    sel = FN.selection(st, led.records(), (), gate)
    assert "BLOCKED" in FN.blocked_line(sel) and sel["honest"] == []
    V.check(ws, "0002")
    sel = FN.selection(st, led.records(), (), gate)
    assert FN.blocked_line(sel) is None and [i for i, _ in sel["honest"]] == ["0002_good"]
    st.final_picks = ["0001"]
    st.save(ws)
    cli.main(["final", "--accept-unvaulted", "0001", "--why", "deadline"])
    sel = FN.selection(S.load(ws), led.records(), (), V.gate(V.meta(ws)))
    assert FN.blocked_line(sel) is None
    assert "WITHOUT a vault pass: deadline" in (ws / "reports/final-selection.md").read_text()
    capsys.readouterr()
    cli.main(["status", "--offline"])
    assert "vault: sealed 10%" in capsys.readouterr().out


def test_no_vault_changes_nothing_in_the_final_candidates(tmp_path, monkeypatch):
    ws = _init(tmp_path, monkeypatch, _data())
    led = Ledger(ws)
    led.log("base", 0.8, decision="baseline")
    sel = FN.selection(S.load(ws), led.records(), (), V.gate(V.meta(ws)))
    assert [i for i, _ in sel["honest"]] == ["0001_base"] and FN.blocked_line(sel) is None


# ---------------------------------------------------------------------------- opening
def test_open_puts_every_row_back_where_it_was(sealed, capsys):
    ws, df = sealed
    n_v = V.meta(ws)["n_rows"]
    cli.main(["vault", "open"])
    back = pd.read_csv(ws / "data/train.csv")
    pd.testing.assert_frame_equal(back, df)
    folds = pd.read_csv(ws / "data/folds.csv")
    assert (folds["fold"] == -1).sum() == n_v and (folds["fold"] == V.VAULT).sum() == 0
    assert not (ws / V.FEATURES).exists() and V.meta(ws)["state"] == "opened"
    assert "opened" in V.status_line(ws)
    with pytest.raises(SystemExit, match="no sealed vault"):
        V.check(ws, "0001")


# ---------------------------------------------------------------------------- holdout+lb
def test_holdout_mode_carves_the_vault_from_the_holdout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli.main(["init", "demo", "--metric", "auc", "--target", "target", "--id-col", "id"])
    h = _data(n=3000)[["id", "target", "grp"]]
    h.to_csv(tmp_path / "data/holdout.csv", index=False)
    st = S.load(tmp_path)
    st.validation, st.eval_truth, st.eval_groups = "holdout+lb", "data/holdout.csv:target", "data/holdout.csv:grp"
    st.save(tmp_path)
    cli.main(["vault", "seal", "--frac", "0.2"])
    m = V.meta(tmp_path)
    rest, vault = pd.read_csv(tmp_path / "data/holdout.csv"), pd.read_csv(tmp_path / V.FEATURES)
    assert m["mode"] == "holdout" and len(rest) + len(vault) == 3000 and "target" not in vault.columns
    assert set(vault["grp"]).isdisjoint(rest["grp"])
    cli.main(["vault", "open"])
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "data/holdout.csv"), h)


# ---------------------------------------------------------------------------- the templates
def _run(args, cwd, plugin_root):
    env = dict(os.environ, KGKIT_HOME=str(plugin_root),
               PYTHONPATH=os.pathsep.join([str(plugin_root), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def test_the_gbdt_template_predicts_the_vault_and_never_trains_on_it(sealed, plugin_root):
    ws, df = sealed
    test = df.iloc[:60].drop(columns="target").assign(id=lambda d: d["id"] + 10_000)
    test.to_csv(ws / "data/test.csv", index=False)
    pd.DataFrame({"id": test["id"], "target": 0.5}).to_csv(ws / "data/sample_submission.csv", index=False)
    (ws / "src").mkdir(exist_ok=True)
    shutil.copy(plugin_root / "templates" / "train_gbdt.py", ws / "src" / "train_gbdt.py")
    args = ["src/train_gbdt.py", "--model", "hgb", "--params", '{"max_iter": 60}']
    out = _run([*args, "--name", "local"], ws, plugin_root)
    assert "vault: " in out
    led = Ledger(ws)
    local = led.records()[-1]
    m = V.meta(ws)
    assert local["vault_hash"] == m["vault_hash"] and len(led.load_vault(local["id"])) == m["n_rows"]
    assert len(led.load_test(local["id"])) == 60 and len(led.load_oof(local["id"])) == 4000 - m["n_rows"]
    # a remote run: the original train.csv with the vault rows (labels too), and no data/vault.csv
    visible = pd.read_csv(ws / "data/train.csv")
    shutil.move(str(ws / "data/train.csv"), str(ws / "visible.csv"))
    shutil.move(str(ws / V.FEATURES), str(ws / "vault_features.csv"))
    df.to_csv(ws / "data/train.csv", index=False)
    _run([*args, "--name", "remote"], ws, plugin_root)
    remote = Ledger(ws).records()[-1]
    assert remote["cv"] == pytest.approx(local["cv"])
    np.testing.assert_allclose(led.load_vault(remote["id"]), led.load_vault(local["id"]))
    np.testing.assert_allclose(led.load_oof(remote["id"]), led.load_oof(local["id"]))
    assert len(visible) == len(led.load_oof(remote["id"]))


# ---------------------------------------------------------------------------- the hooks
SH = shutil.which("sh")


def _hook(plugin_root, script, payload, env=None):
    r = subprocess.run([SH, str(plugin_root / "hooks" / "run.sh"), script], input=json.dumps(payload),
                       capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0, r.stderr
    out = r.stdout.strip()
    return json.loads(out)["hookSpecificOutput"] if out else None


@pytest.mark.skipif(SH is None, reason="needs a POSIX sh")
@pytest.mark.parametrize("tool, inp, denied", [
    ("Read", {"file_path": "~/.kaggle-gm/vault/demo-1234abcd/rows.csv"}, True),
    ("Read", {"file_path": "C:\\Users\\me\\.kaggle-gm\\vault\\demo-1234abcd\\rows.csv"}, True),
    ("Glob", {"pattern": "/home/me/.kaggle-gm/vault/**/*.csv"}, True),
    ("Grep", {"pattern": "target", "path": "/home/me/.kaggle-gm/vault"}, True),
    ("Edit", {"file_path": "/home/me/.kaggle-gm/vault/x/checks.jsonl", "old_string": "a", "new_string": "b"}, True),
    ("Read", {"file_path": "/ws/.kaggle-gm/vault.json"}, False),  # the workspace's metadata holds no label
    ("Read", {"file_path": "/ws/.kaggle-gm/ledger.jsonl"}, False),
    ("Grep", {"pattern": "kaggle-gm/vault", "path": "/ws/src"}, False),  # a search pattern is not a path
])
def test_the_file_tools_cannot_read_the_vault(plugin_root, tool, inp, denied):
    out = _hook(plugin_root, "pre_file.py", {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": inp})
    assert (out is not None and out["permissionDecision"] == "deny") == denied


@pytest.mark.skipif(SH is None, reason="needs a POSIX sh")
def test_the_vault_home_set_by_env_is_guarded_too(plugin_root, tmp_path):
    env = dict(os.environ, KG_VAULT_HOME=str(tmp_path / "labels"))
    out = _hook(plugin_root, "pre_file.py", {"tool_name": "Read", "tool_input": {
        "file_path": str(tmp_path / "labels" / "k" / "rows.csv")}}, env)
    assert out["permissionDecision"] == "deny"


@pytest.mark.skipif(SH is None, reason="needs a POSIX sh")
@pytest.mark.parametrize("cmd, denied", [
    ("cat ~/.kaggle-gm/vault/demo-1234abcd/rows.csv", True),
    ("python -c \"import pandas as pd; print(pd.read_csv('/home/me/.kaggle-gm/vault/k/rows.csv'))\"", True),
    ("Get-Content $env:USERPROFILE\\.kaggle-gm\\vault\\k\\rows.csv", True),
    ("kgkit vault check 0007", False),
    ("cat .kaggle-gm/vault.json", False),
])
def test_commands_cannot_read_the_vault(plugin_root, tmp_path, cmd, denied):
    out = _hook(plugin_root, "pre_tool.py", {"hook_event_name": "PreToolUse", "tool_name": "Bash",
                                             "tool_input": {"command": cmd}, "cwd": str(tmp_path)})
    assert (out is not None and out.get("permissionDecision") == "deny") == denied


def test_the_status_helpers_import_no_third_party_module():
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import kgkit.vault as V, kgkit.finals; "
            "bad = [m for m in ('numpy', 'pandas', 'sklearn') if m in sys.modules]; print(bad); "
            "sys.exit(1 if bad else 0)")
    from conftest import PLUGIN_ROOT

    r = subprocess.run([sys.executable, "-c", code, str(PLUGIN_ROOT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
