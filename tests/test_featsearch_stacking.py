"""Feature search, multi-level / residual stacking, library pruning, public-notebook review."""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from kgkit import ensemble as E
from kgkit import featsearch as FS
from kgkit import kernels as K
from kgkit import metrics as M


def _sig(z):
    return 1 / (1 + np.exp(-z))


# ----------------------------------------------------------------------------- feature search
@pytest.fixture
def interaction_data():
    """Target depends on the *group mean* of x by city and on a city x shop interaction: invisible
    to a model that sees only the raw columns at this sample size, easy for generated features."""
    rng = np.random.default_rng(0)
    n = 3000
    city = rng.integers(0, 60, n)
    shop = rng.integers(0, 4, n)
    x = rng.normal(size=n) + city * 0.05
    noise = rng.normal(size=n)
    city_mean = pd.Series(x).groupby(city).transform("mean").to_numpy()
    inter = ((city + shop) % 7 == 0).astype(float)
    logit = 2.5 * (x - city_mean) + 2.0 * inter - 0.5
    y = (rng.random(n) < _sig(logit)).astype(int)
    df = pd.DataFrame({"city": city.astype(str), "shop": shop.astype(str), "x": x, "noise": noise})
    folds = np.arange(n) % 4
    return df, y, folds


def test_generate_and_materialize(interaction_data):
    df, y, folds = interaction_data
    specs = FS.generate_specs(["city", "shop"], ["x", "noise"])
    names = {FS.spec_name(s) for s in specs}
    assert "cnt_city__shop" in names and "grp_x_diff_mean_by_city" in names and "num2_x_ratio_noise" in names
    assert len(names) == len(specs)
    test = df.iloc[:100].copy()
    tr, te = FS.materialize([s for s in specs if FS.spec_name(s) in
                             ("cnt_city__shop", "te_city", "grp_x_diff_mean_by_city", "num2_x_diff_noise")],
                            df, test, y, folds)
    assert tr.shape == (len(df), 4) and te.shape == (100, 4)
    expect = df["x"] - pd.concat([df["x"], test["x"]]).groupby(pd.concat([df["city"], test["city"]])).transform("mean").iloc[:len(df)].to_numpy()
    assert np.allclose(tr["grp_x_diff_mean_by_city"], expect)
    assert tr["te_city"].notna().all()
    with pytest.raises(ValueError):
        FS.generate_specs(["city"], ["x"], aggs=["bogus"])


def test_feature_search_finds_group_deviation(interaction_data):
    df, y, folds = interaction_data
    specs = FS.generate_specs(["city", "shop"], ["x", "noise"], kinds=("cnt", "te", "grp"),
                              aggs=("mean", "diff_mean", "std"), seed=1)
    res = FS.feature_search(df, y, folds, ["city", "shop"], ["x", "noise"], "auc", specs=specs, batch_size=6,
                            model="hgb", verbose=False, recheck_folds=(np.arange(len(y)) * 7 + 3) % 4)
    kept = {FS.spec_name(s) for s in res["kept"]}
    assert res["final_cv"] > res["base_cv"] + 0.01
    assert any(k.startswith(("grp_x_diff_mean_by_city", "grp_x_mean_by_city", "te_city")) for k in kept)
    assert res["recheck"]["gain"] > 0
    md = FS.report(res)
    assert "kept:" in md and "recheck" in md


def test_feature_search_cli(tmp_path, plugin_root, interaction_data):
    df, y, folds = interaction_data
    df = df.assign(y=y, id=np.arange(len(df)))
    df.to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"id": df["id"], "fold": folds}).to_csv(tmp_path / "folds.csv", index=False)
    env = dict(os.environ, PYTHONPATH=str(plugin_root))
    r = subprocess.run([sys.executable, "-m", "kgkit", "features", "search", "train.csv", "--target", "y",
                        "--id-col", "id", "--folds", "folds.csv:fold", "--metric", "auc", "--kinds", "cnt,grp",
                        "--aggs", "diff_mean", "--batch", "4", "--model", "hgb", "--recheck-seed", "5",
                        "--out", "fs.json"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    data = json.loads((tmp_path / "fs.json").read_text(encoding="utf-8"))
    assert data["kept"] and "recheck" in data
    assert FS.load_specs(tmp_path / "fs.json") == data["kept"]
    assert (tmp_path / "fs.md").is_file()


# ----------------------------------------------------------------------------- stacking
@pytest.fixture
def library():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n)
    s = y + rng.normal(0, 1, n)
    oofs, tests = {}, {}
    for i in range(6):
        oofs[f"m{i}"] = _sig(s + rng.normal(0, 1.0, n))
        tests[f"m{i}"] = rng.random(500)
    oofs["m0_dup"] = np.clip(oofs["m0"] + rng.normal(0, 1e-4, n), 0, 1)
    tests["m0_dup"] = tests["m0"]
    return y, oofs, tests, np.arange(n) % 5


def test_prune_library_drops_near_duplicates(library):
    y, oofs, _, _ = library
    pr = E.prune_library(oofs, y, "auc", max_models=4)
    assert len(pr["kept"]) == 4
    assert not {"m0", "m0_dup"} <= set(pr["kept"])
    assert any("corr" in why for why in pr["dropped"].values())


def test_multi_level_stack(library):
    y, oofs, tests, folds = library
    one = E.multi_level_stack(oofs, tests, y, folds, "auc", n_layers=1)
    two = E.multi_level_stack(oofs, tests, y, folds, "auc", n_layers=2)
    best_single = max(M.score("auc", y, p) for p in oofs.values())
    assert one["score"] > best_single
    assert set(one["layer_scores"][0]) == {"L2_lin", "L2_gbm"} and len(two["layer_scores"]) == 2
    assert one["final_honest"] is not None and one["test"].shape == (500,)


def test_stack_extra_features_and_residual():
    rng = np.random.default_rng(0)
    n = 3000
    region = rng.integers(0, 2, n)
    y = rng.normal(size=n) * 2
    good_everywhere = y + rng.normal(0, 1.0, n)
    # biased in region 1: a residual model with the region feature can learn the correction
    biased = y + rng.normal(0, 0.5, n) + 1.5 * region
    oofs = {"a": good_everywhere, "b": biased}
    tests = {"a": rng.normal(size=200), "b": rng.normal(size=200)}
    folds = np.arange(n) % 5
    rmse = M.get("rmse")
    res = E.residual_stack(oofs, tests, y, folds, base="b", X_extra=region[:, None],
                           X_extra_test=rng.integers(0, 2, (200, 1)))
    assert rmse(y, res["oof"]) < rmse(y, biased) * 0.8
    st = E.stack(oofs, tests, y, folds, X_extra=region[:, None], X_extra_test=rng.integers(0, 2, (200, 1)))
    assert rmse(y, st["oof"]) < min(rmse(y, good_everywhere), rmse(y, biased))
    with pytest.raises(ValueError):
        E.residual_stack({"a": np.zeros((n, 3))}, {"a": np.zeros((5, 3))}, y, folds, base="a")


# ----------------------------------------------------------------------------- public notebooks
def test_kernel_review_and_local_script(tmp_path):
    d = tmp_path / "ref" / "lgbm-starter"
    d.mkdir(parents=True)
    nb = {"cells": [
        {"cell_type": "code", "source": ["!pip install -q catboost\n", "import lightgbm as lgb\n",
                                         "from sklearn.model_selection import StratifiedGroupKFold\n"],
         "outputs": [{"text": ["CV AUC: 0.91234\n", "fold 0 score 0.9050\n"]}]},
        {"cell_type": "code", "source": "train = pd.read_csv('/kaggle/input/my-comp/train.csv')\n"
                                        "ext = pd.read_csv('/kaggle/input/orig-data/orig.csv')\n"
                                        "oof.to_csv('/kaggle/working/oof.csv')\n%matplotlib inline\n",
         "outputs": []},
        {"cell_type": "markdown", "source": "notes"}]}
    (d / "lgbm-starter.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({"dataset_sources": ["someone/orig-data"],
                                                         "competition_sources": ["my-comp"]}), encoding="utf-8")
    rev = K.analyse_dir(d, "author/lgbm-starter", "my-comp")
    assert rev["cv"] == ["StratifiedGroupKFold"]
    assert "LightGBM" in rev["libs"]
    assert any("0.91234" in s for s in rev["scores"])
    assert rev["other_inputs"] == ["orig-data"] and rev["external"]["datasets"] == ["someone/orig-data"]
    script = (d / "lgbm-starter_local.py").read_text(encoding="utf-8")
    assert "pd.read_csv('data/train.csv')" in script
    assert "data/ext/orig-data/orig.csv" in script and "artifacts/ref/lgbm-starter/oof.csv" in script
    assert "# !pip install" in script and "# %matplotlib" in script
    compile(script.replace("pd.read_csv", "print"), "x", "exec")
    review = (d / "REVIEW.md").read_text(encoding="utf-8")
    assert "Non-competition inputs" in review and "orig-data" in review


def test_parse_kernel_list_csv():
    text = "Warning: Looks like you're using an outdated API Version\nref,title,author,lastRunTime,totalVotes\n" \
           "a/b,Best LGBM,a,2026-10-01,120\nc/d,EDA,c,2026-09-01,80\n"
    rows = K.parse_list_csv(text)
    assert [r["ref"] for r in rows] == ["a/b", "c/d"] and rows[0]["totalVotes"] == "120"
