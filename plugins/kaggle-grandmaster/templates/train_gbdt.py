"""GBDT training template (LightGBM / XGBoost / CatBoost / sklearn HistGradientBoosting).

Copy to src/, edit CONFIG and add_features(), then:

    python src/train_gbdt.py                         # full run
    python src/train_gbdt.py --model cat --name cat_v1 --notes "catboost on base features"
    python src/train_gbdt.py --smoke                  # 1 fold, few rounds — crash test
    python src/train_gbdt.py --seeds 42 43 44         # seed averaging

What it guarantees:
  * uses the frozen folds in data/folds.csv (joined on id, or by row order when no id column)
  * OOF predictions + fold-averaged test predictions, scored with the competition metric
  * label metrics handled properly (threshold / argmax / optimised rounding fitted on OOF)
  * every run logged to the experiment ledger with OOF/test artefacts and a submission file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import kgkit  # noqa: F401  (vendored into src/ via `python -m kgkit vendor src/`)
except ImportError:
    sys.path.insert(0, os.environ.get("KGKIT_HOME", ""))
from kgkit import metrics as M
from kgkit import state as S
from kgkit.cv import split_indices
from kgkit.experiment import Ledger, seed_everything
from kgkit.thresholds import OptimizedRounder, best_threshold

# ----------------------------------------------------------------------------- config
CONFIG = dict(
    name="lgbm_baseline",
    train="data/train.csv",
    test="data/test.csv",
    sample="data/sample_submission.csv",
    folds="data/folds.csv",
    id_col=None,          # default: from .kaggle-gm/competition.json
    target=None,          # default: from .kaggle-gm/competition.json
    metric=None,          # default: from .kaggle-gm/competition.json
    model="lgbm",         # lgbm | xgb | cat | hgb
    task="auto",          # auto | binary | multiclass | regression
    drop=[],              # columns never used as features
    log_target=False,     # train on log1p(y) (RMSLE-style targets)
    seeds=[42],
    early_stopping=200,
    params={},            # overrides merged into the model defaults below
    notes="",
)

DEFAULTS = {
    "lgbm": dict(learning_rate=0.03, n_estimators=20000, num_leaves=63, min_child_samples=40, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0, verbose=-1, n_jobs=-1),
    "xgb": dict(learning_rate=0.03, n_estimators=20000, max_depth=6, min_child_weight=5, subsample=0.8,
                colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0, tree_method="hist", enable_categorical=True,
                max_bin=256),
    "cat": dict(learning_rate=0.05, iterations=20000, depth=6, l2_leaf_reg=3, verbose=0, allow_writing_files=False),
    "hgb": dict(learning_rate=0.05, max_iter=2000, max_leaf_nodes=63, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.1),
}


def add_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature engineering hook. Statistics that do not use the target may pool train+test.
    Target-based encodings must be computed out-of-fold (kgkit.features.oof_target_encode)."""
    return train, test


# ----------------------------------------------------------------------------- helpers
def infer_task(y: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 2:
        return "binary" if y.nunique() == 2 else "multiclass"
    if y.nunique() <= 20 and np.allclose(y % 1, 0):
        return "multiclass"
    return "regression"


def make_model(kind: str, task: str, params: dict, seed: int, n_classes: int):
    p = {**DEFAULTS[kind], **params}
    if kind == "lgbm":
        import lightgbm as lgb

        cls = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
        return cls(random_state=seed, **p)
    if kind == "xgb":
        import xgboost as xgb

        cls = xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier
        return cls(random_state=seed, **p)
    if kind == "cat":
        import catboost as cb

        cls = cb.CatBoostRegressor if task == "regression" else cb.CatBoostClassifier
        return cls(random_seed=seed, **p)
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    cls = HistGradientBoostingRegressor if task == "regression" else HistGradientBoostingClassifier
    return cls(random_state=seed, categorical_features="from_dtype", **p)


def fit(model, kind, X_tr, y_tr, X_va, y_va, cats, es):
    if kind == "lgbm":
        import lightgbm as lgb

        import inspect

        # LightGBM >= 4.7 renamed eval_set to eval_X/eval_y; Kaggle images may ship older versions
        if "eval_X" in inspect.signature(model.fit).parameters:
            evals = dict(eval_X=(X_va,), eval_y=(y_va,))
        else:
            evals = dict(eval_set=[(X_va, y_va)])
        model.fit(X_tr, y_tr, callbacks=[lgb.early_stopping(es, verbose=False)], **evals)
        return model, getattr(model, "best_iteration_", None)
    if kind == "xgb":
        model.set_params(early_stopping_rounds=es)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return model, getattr(model, "best_iteration", None)
    if kind == "cat":
        model.fit(X_tr, y_tr, cat_features=cats, eval_set=(X_va, y_va), early_stopping_rounds=es, use_best_model=True)
        return model, model.get_best_iteration()
    model.fit(X_tr, y_tr)
    return model, getattr(model, "n_iter_", None)


def predict(model, task, X):
    if task == "regression":
        return model.predict(X)
    p = model.predict_proba(X)
    return p[:, 1] if task == "binary" else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--model", choices=list(DEFAULTS))
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--params", type=json.loads, help='JSON overrides, e.g. \'{"num_leaves": 127}\'')
    ap.add_argument("--notes")
    ap.add_argument("--smoke", action="store_true", help="1 fold, 50 rounds, no ledger/submission")
    args = ap.parse_args()
    cfg = dict(CONFIG)
    for k in ("name", "model", "seeds", "notes"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    if args.params:
        cfg["params"] = {**cfg["params"], **args.params}

    st = S.load()
    id_col = cfg["id_col"] or (st.id_col if st else None) or None
    target = cfg["target"] or (st.target if st else None)
    metric = M.get(cfg["metric"] or (st.metric if st else "auc"))
    if not target:
        raise SystemExit("set CONFIG['target'] or run `python -m kgkit init` with --target")

    t0 = time.time()
    train, test = pd.read_csv(cfg["train"]), pd.read_csv(cfg["test"])
    folds_df = pd.read_csv(cfg["folds"])
    if id_col and id_col in folds_df.columns:
        train = train.merge(folds_df[[id_col, "fold"]], on=id_col, how="left", validate="one_to_one")
    else:
        if len(folds_df) != len(train):
            raise SystemExit("folds.csv has no id column and a different row count than train")
        train["fold"] = folds_df["fold"].to_numpy()
    if train["fold"].isna().any():
        raise SystemExit("some training rows have no fold — regenerate folds.csv")
    train, test = add_features(train, test)

    y_raw = train[target]
    task = infer_task(y_raw) if cfg["task"] == "auto" else cfg["task"]
    classes = None
    if task == "regression":
        y = np.log1p(y_raw.to_numpy(dtype=float)) if cfg["log_target"] else y_raw.to_numpy(dtype=float)
    else:
        classes, y = np.unique(y_raw.to_numpy(), return_inverse=True)
    n_classes = 1 if classes is None else len(classes)

    drop = set(cfg["drop"]) | {target, "fold"} | ({id_col} if id_col else set())
    feats = [c for c in train.columns if c not in drop and c in test.columns]
    cats = [c for c in feats if not pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_bool_dtype(train[c])]
    for c in cats:  # shared category vocabulary across train/test
        dtype = pd.CategoricalDtype(pd.concat([train[c], test[c]]).astype(str).unique())
        train[c], test[c] = train[c].astype(str).astype(dtype), test[c].astype(str).astype(dtype)
    if cfg["model"] == "cat":
        for c in cats:
            train[c], test[c] = train[c].astype(str), test[c].astype(str)
    X, X_test = train[feats], test[feats]
    folds = train["fold"].astype(int).to_numpy()

    params = dict(cfg["params"])
    if args.smoke:
        n_key = {"lgbm": "n_estimators", "xgb": "n_estimators", "cat": "iterations", "hgb": "max_iter"}[cfg["model"]]
        params[n_key] = 50
    oof = np.zeros((len(train), n_classes)) if task == "multiclass" else np.zeros(len(train))
    test_pred = np.zeros((len(test), n_classes)) if task == "multiclass" else np.zeros(len(test))
    fold_scores, best_iters = [], []
    fold_list = list(split_indices(folds))[: 1 if args.smoke else None]
    for seed in cfg["seeds"]:
        seed_everything(seed)
        for f, tr, va in fold_list:
            model = make_model(cfg["model"], task, params, seed, n_classes)
            model, it = fit(model, cfg["model"], X.iloc[tr], y[tr], X.iloc[va], y[va], cats, cfg["early_stopping"])
            best_iters.append(it)
            oof[va] += predict(model, task, X.iloc[va]) / len(cfg["seeds"])
            test_pred += predict(model, task, X_test) / (len(cfg["seeds"]) * len(fold_list))
    va_mask = np.isin(folds, [f for f, _, _ in fold_list])

    # ------------------------------------------------------------ metric-space conversion
    def decode(p_oof, p_test):
        """Map raw model outputs to what the metric (and submission) expects."""
        if task == "regression" and cfg["log_target"]:
            p_oof, p_test = np.expm1(p_oof), np.expm1(p_test)
        if metric.kind != "label":
            return p_oof, p_test, {}
        if task == "binary":
            t, _ = best_threshold(y[va_mask], p_oof[va_mask], metric.name)
            return (p_oof >= t).astype(int), (p_test >= t).astype(int), {"threshold": t}
        if task == "multiclass":
            return p_oof.argmax(1), p_test.argmax(1), {}
        r = OptimizedRounder(metric=metric.name).fit(p_oof[va_mask], y_raw.to_numpy()[va_mask])
        return r.predict(p_oof), r.predict(p_test), {"thresholds": r.thresholds_.tolist()}

    oof_m, test_m, post = decode(oof, test_pred)
    y_metric = y_raw.to_numpy() if task == "regression" else y
    for f, _, va in fold_list:
        fold_scores.append(metric(y_metric[va], oof_m[va]))
    cv = metric(y_metric[va_mask], oof_m[va_mask])
    print(f"[{cfg['name']}] {metric.name} CV {cv:.5f} +/- {np.std(fold_scores):.5f} | folds "
          + " ".join(f"{s:.4f}" for s in fold_scores) + f" | {time.time() - t0:.0f}s")
    if args.smoke:
        print("smoke run OK (not logged)")
        return

    rec = Ledger().log(
        cfg["name"], cv, fold_scores=fold_scores, params={**DEFAULTS[cfg["model"]], **params, **post},
        features=feats, model=cfg["model"], notes=cfg["notes"], oof=oof, test_pred=test_pred,
        task=task, seeds=cfg["seeds"], best_iterations=best_iters, runtime_s=round(time.time() - t0, 1),
    )
    sample = pd.read_csv(cfg["sample"])
    key = sample.columns[0]
    value_cols = list(sample.columns[1:])
    pred = pd.DataFrame(index=test.index)
    if task == "multiclass" and len(value_cols) == n_classes and metric.kind != "label":
        names = list(map(str, classes))
        order = [names.index(str(c)) if str(c) in names else i for i, c in enumerate(value_cols)]
        pred[value_cols] = test_pred[:, order]
    elif task != "regression" and metric.kind == "label":
        pred[value_cols[0]] = classes[test_m]
    else:
        pred[value_cols[0]] = test_m
    if key in test.columns:  # align to sample_submission's id order
        pred.insert(0, key, test[key].to_numpy())
        sub = sample[[key]].merge(pred, on=key, how="left")
    else:  # no id in test: rely on identical row order
        sub = pd.concat([sample[[key]].reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    Path("subs").mkdir(exist_ok=True)
    out = Path("subs") / f"{rec['id']}.csv"
    sub.to_csv(out, index=False)
    Ledger().update(rec["id"], submission=str(out).replace("\\", "/"))
    print(f"logged {rec['id']} -> {out}")


if __name__ == "__main__":
    main()
