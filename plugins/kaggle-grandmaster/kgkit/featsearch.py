"""Brute-force feature search: generate thousands of candidate features, keep the ones CV accepts.

This is how recent tabular / Playground competitions were won (e.g. 10,000+ groupby features
generated and screened, the best ~500 kept): the space of COL1 x COL2 x STAT combinations is too
large to reason through, but cheap to *test* with a fast GBDT on the frozen folds.

Candidate kinds (all leak-safe: target statistics are out-of-fold on the frozen folds; the
others are unsupervised and may pool train + test):

  cnt   count encoding of a categorical column or a pair of them
  te    out-of-fold smoothed target mean of a categorical column or a pair (binary/regression)
  grp   groupby aggregate of a numeric column by a categorical (or a pair): mean, std, min, max,
        median, nunique, value minus group mean, value / group mean, rank within group
  num2  pairwise arithmetic of numeric columns: ratio, difference, product, sum

Search: forward selection by *batches*. Each batch of candidates is added to the current set;
a batch that improves the paired fold score is kept, then pruned to its most important members
(LightGBM gain) and re-checked. Screening many batches on the same folds makes the final CV
optimistic, so pass ``recheck_folds`` (a different fold seed) for an honest before/after, and
confirm the kept set with the real model.

    from kgkit.featsearch import feature_search, materialize, save_specs, load_specs
    res = feature_search(train, y, folds, cat_cols=cats, num_cols=nums, metric="auc", test=test)
    save_specs("reports/featsearch.json", res)
    tr_new, te_new = materialize(load_specs("reports/featsearch.json"), train, test, y, folds)

CLI: ``python -m kgkit features search data/train.csv --target y --folds data/folds.csv:fold``.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import metrics as M
from .compare import compare_folds
from .features import oof_target_encode

GRP_AGGS = ("mean", "std", "min", "max", "median", "nunique", "diff_mean", "ratio_mean", "rank")
NUM_OPS = ("ratio", "diff", "prod", "sum")


# ----------------------------------------------------------------------------- candidate specs
def spec_name(spec: dict) -> str:
    k = spec["kind"]
    if k in ("cnt", "te"):
        return f"{k}_{'__'.join(spec['cols'])}"
    if k == "grp":
        return f"grp_{spec['col']}_{spec['agg']}_by_{'__'.join(spec['by'])}"
    if k == "num2":
        return f"num2_{spec['a']}_{spec['op']}_{spec['b']}"
    raise ValueError(f"unknown spec kind {k!r}")


def generate_specs(cat_cols: Sequence[str], num_cols: Sequence[str], kinds: Iterable[str] = ("cnt", "te", "grp", "num2"),
                   aggs: Sequence[str] = ("mean", "std", "nunique", "diff_mean", "rank"),
                   ops: Sequence[str] = NUM_OPS, max_order: int = 2, max_candidates: int | None = None,
                   seed: int = 0) -> list[dict]:
    """All candidate specs over the given columns (order 1 and pairs), shuffled reproducibly."""
    kinds = set(kinds)
    groups = [[c] for c in cat_cols]
    if max_order >= 2:
        groups += [list(p) for p in itertools.combinations(cat_cols, 2)]
    specs: list[dict] = []
    if "cnt" in kinds:
        specs += [{"kind": "cnt", "cols": g} for g in groups]
    if "te" in kinds:
        specs += [{"kind": "te", "cols": g} for g in groups]
    if "grp" in kinds:
        bad = set(aggs) - set(GRP_AGGS)
        if bad:
            raise ValueError(f"unknown aggs {sorted(bad)}; choose from {GRP_AGGS}")
        specs += [{"kind": "grp", "by": g, "col": c, "agg": a} for g in groups for c in num_cols for a in aggs]
    if "num2" in kinds:
        for a, b in itertools.combinations(num_cols, 2):
            for op in ops:
                specs.append({"kind": "num2", "a": a, "b": b, "op": op})
                if op in ("ratio", "diff"):  # non-commutative: both directions
                    specs.append({"kind": "num2", "a": b, "b": a, "op": op})
    rng = np.random.default_rng(seed)
    specs = [specs[i] for i in rng.permutation(len(specs))]
    return specs[:max_candidates] if max_candidates else specs


# ----------------------------------------------------------------------------- materialisation
def _codes(pool: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    return pool.groupby(list(cols), dropna=False, sort=False).ngroup()


def materialize(specs: Sequence[dict], train: pd.DataFrame, test: pd.DataFrame | None = None, y=None, folds=None,
                smoothing: float = 20.0) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Build the features of ``specs`` for train (and test). ``y`` and ``folds`` are needed for ``te``."""
    n = len(train)
    pool = pd.concat([train, test], ignore_index=True) if test is not None else train.reset_index(drop=True)
    out: dict[str, np.ndarray] = {}
    for spec in specs:
        name, k = spec_name(spec), spec["kind"]
        if k == "cnt":
            codes = _codes(pool, spec["cols"])
            out[name] = codes.map(codes.value_counts()).to_numpy(dtype=float)
        elif k == "te":
            if y is None or folds is None:
                raise ValueError("target encoding needs y and folds")
            tr = train[spec["cols"]].reset_index(drop=True).copy()
            tr["__y"] = np.asarray(y, dtype=float)
            te_df = test[spec["cols"]].reset_index(drop=True) if test is not None else None
            a, b = oof_target_encode(tr, te_df, spec["cols"], "__y", folds, smoothing=smoothing, name=name)
            out[name] = np.concatenate([a.to_numpy(), b.to_numpy()]) if b is not None else a.to_numpy()
        elif k == "grp":
            by, col, agg = list(spec["by"]), spec["col"], spec["agg"]
            g = pool.groupby(by, dropna=False, sort=False)[col]
            v = pool[col].astype(float)
            if agg in ("mean", "std", "min", "max", "median", "nunique"):
                r = g.transform(agg)
            elif agg == "diff_mean":
                r = v - g.transform("mean")
            elif agg == "ratio_mean":
                r = v / g.transform("mean").replace(0, np.nan)
            elif agg == "rank":
                r = g.rank(pct=True)
            else:
                raise ValueError(agg)
            out[name] = r.to_numpy(dtype=float)
        elif k == "num2":
            a, b = pool[spec["a"]].astype(float), pool[spec["b"]].astype(float)
            op = spec["op"]
            r = a / b.replace(0, np.nan) if op == "ratio" else a - b if op == "diff" else a * b if op == "prod" else a + b
            out[name] = r.to_numpy(dtype=float)
    df = pd.DataFrame(out)
    tr_out = df.iloc[:n].set_index(train.index)
    te_out = df.iloc[n:].set_index(test.index) if test is not None else None
    return tr_out, te_out


# ----------------------------------------------------------------------------- screening model
def _task(y: np.ndarray) -> str:
    u = np.unique(y)
    if len(u) == 2:
        return "binary"
    if len(u) <= 20 and np.allclose(u % 1, 0):
        return "multiclass"
    return "regression"


def _have_lgbm() -> bool:
    try:
        import lightgbm  # noqa: F401

        return True
    except ImportError:
        return False


def _to_metric_input(metric: M.Metric, task: str, p: np.ndarray) -> np.ndarray:
    if metric.kind != "label":
        return p
    if task == "binary":
        return (p >= 0.5).astype(int)
    if task == "multiclass":
        return p.argmax(1)
    return np.rint(p)


def cv_scores(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray, metric: str, model: str = "auto",
              fold_ids: Sequence[int] | None = None, seed: int = 0, n_estimators: int = 400) -> dict:
    """Fast fixed-parameter GBDT on the frozen folds: per-fold scores, OOF and gain importances.
    The parameters are deliberately fixed — the goal is a *relative* comparison of feature sets."""
    m = M.get(metric)
    task = _task(y)
    kind = ("lgbm" if _have_lgbm() else "hgb") if model == "auto" else model
    fold_ids = list(fold_ids) if fold_ids is not None else sorted(set(folds[folds >= 0].tolist()))
    n_classes = len(np.unique(y)) if task == "multiclass" else 1
    oof = np.full((len(y), n_classes) if n_classes > 1 else len(y), np.nan)
    scores, imp = [], np.zeros(X.shape[1])
    for f in fold_ids:
        tr, va = np.flatnonzero((folds != f) & (folds >= 0)), np.flatnonzero(folds == f)
        if kind == "lgbm":
            import lightgbm as lgb

            cls = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
            mdl = cls(n_estimators=n_estimators, learning_rate=0.1, num_leaves=31, min_child_samples=20,
                      colsample_bytree=0.8, subsample=0.8, subsample_freq=1, random_state=seed, verbose=-1)
            import inspect

            # LightGBM >= 4.7 renamed eval_set to eval_X/eval_y (same shim as templates/train_gbdt.py)
            if "eval_X" in inspect.signature(mdl.fit).parameters:
                evals = dict(eval_X=(X.iloc[va],), eval_y=(y[va],))
            else:
                evals = dict(eval_set=[(X.iloc[va], y[va])])
            mdl.fit(X.iloc[tr], y[tr], callbacks=[lgb.early_stopping(50, verbose=False)], **evals)
            imp += mdl.booster_.feature_importance("gain")
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

            cls = HistGradientBoostingRegressor if task == "regression" else HistGradientBoostingClassifier
            mdl = cls(learning_rate=0.1, max_iter=n_estimators, early_stopping=True, validation_fraction=0.1,
                      n_iter_no_change=30, random_state=seed)
            mdl.fit(X.iloc[tr], y[tr])
        if task == "regression":
            p = mdl.predict(X.iloc[va])
        else:
            p = mdl.predict_proba(X.iloc[va])
            p = p[:, 1] if task == "binary" else p
        oof[va] = p
        scores.append(m(y[va], _to_metric_input(m, task, p)))
    has_imp = kind == "lgbm"
    return {"scores": scores, "oof": oof, "importance": dict(zip(X.columns, imp)) if has_imp else None,
            "model": kind, "greater_is_better": m.greater_is_better}


def _prepare_base(train: pd.DataFrame, base_cols: Sequence[str], test: pd.DataFrame | None = None) -> pd.DataFrame:
    X = pd.DataFrame(index=train.index)
    for c in base_cols:
        s = train[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            X[c] = s.astype(float)
        else:
            pool = pd.concat([s, test[c]]) if test is not None and c in test else s
            cats = pd.Index(pool.astype(str).unique())
            X[c] = pd.Categorical(s.astype(str), categories=cats).codes.astype(float)
    return X


# ----------------------------------------------------------------------------- search
def feature_search(train: pd.DataFrame, y, folds, cat_cols: Sequence[str], num_cols: Sequence[str], metric: str,
                   base_cols: Sequence[str] | None = None, test: pd.DataFrame | None = None,
                   specs: Sequence[dict] | None = None, batch_size: int = 40, max_batches: int | None = None,
                   screen_folds: Sequence[int] | None = None, model: str = "auto", keep_frac: float = 0.5,
                   min_t: float = 1.0, time_budget_s: float | None = None, recheck_folds=None, seed: int = 0,
                   verbose: bool = True, **gen_kw) -> dict:
    """Forward batch selection over candidate features. Returns kept specs, scores and history.

    A batch is accepted when its paired gain over the current set is positive, most screened folds
    improve and the paired t-statistic is >= ``min_t`` (a lenient bar: survivors are re-checked on
    ``recheck_folds`` and must be confirmed with the real model).
    """
    y = np.asarray(y)
    folds = np.asarray(folds)
    if len(y) != len(train) or len(folds) != len(train):
        raise ValueError("train, y and folds must have the same length (same row order)")
    base_cols = list(base_cols) if base_cols is not None else list(dict.fromkeys([*cat_cols, *num_cols]))
    X_base = _prepare_base(train, base_cols, test)
    if specs is None:
        specs = generate_specs(cat_cols, num_cols, seed=seed, **gen_kw)
    task = _task(y)
    specs = [s for s in specs if not (s["kind"] == "te" and task == "multiclass")]
    t0 = time.time()

    def run(X, fids=screen_folds, f=folds):
        return cv_scores(X, y, f, metric, model=model, fold_ids=fids, seed=seed)

    cur = run(X_base)
    gib = cur["greater_is_better"]
    base_scores = list(cur["scores"])
    kept: list[dict] = []
    X_cur = X_base
    history = []
    batches = [specs[i:i + batch_size] for i in range(0, len(specs), batch_size)]
    if max_batches:
        batches = batches[:max_batches]
    for bi, batch in enumerate(batches):
        if time_budget_s and time.time() - t0 > time_budget_s:
            history.append({"batch": bi, "stopped": "time budget"})
            break
        F, _ = materialize(batch, train, None, y, folds)
        F = F.loc[:, F.nunique(dropna=False) > 1]  # drop constants
        F = F[[c for c in F.columns if c not in X_cur.columns]]
        if F.shape[1] == 0:
            continue
        trial = run(pd.concat([X_cur, F], axis=1))
        cmp = compare_folds(trial["scores"], cur["scores"], gib)
        ok = cmp["gain"] > 0 and cmp["folds_improved"] * 2 > cmp["k"] and (cmp["k"] < 2 or cmp["t"] >= min_t)
        entry = {"batch": bi, "n": int(F.shape[1]), "gain": cmp["gain"], "t": cmp["t"], "accepted": 0}
        if ok:
            chosen = list(F.columns)
            if trial["importance"] is not None and len(chosen) > 1:
                imp = sorted(((trial["importance"].get(c, 0.0), c) for c in chosen), reverse=True)
                top = [c for v, c in imp[: max(1, int(len(imp) * keep_frac))] if v > 0]
                if top and len(top) < len(chosen):
                    pruned = run(pd.concat([X_cur, F[top]], axis=1))
                    pc = compare_folds(pruned["scores"], cur["scores"], gib)
                    if pc["gain"] >= cmp["gain"] * 0.8:  # nearly all of the gain with fewer features
                        chosen, trial, cmp = top, pruned, pc
            by_name = {spec_name(s): s for s in batch}
            kept += [by_name[c] for c in chosen]
            X_cur = pd.concat([X_cur, F[chosen]], axis=1)
            cur = trial
            entry.update(accepted=len(chosen), gain=cmp["gain"], cv=float(np.mean(cur["scores"])))
        history.append(entry)
        if verbose:
            print(f"batch {bi + 1}/{len(batches)}: {entry['n']} cand, gain {entry['gain']:+.5f} "
                  f"(t={entry['t']:.2f}) -> kept {entry['accepted']}; total kept {len(kept)}", flush=True)
    res = {
        "metric": metric, "model": cur["model"], "screen_folds": list(screen_folds) if screen_folds else None,
        "base_cols": base_cols, "n_candidates": len(specs), "kept": kept,
        "base_scores": base_scores, "final_scores": list(cur["scores"]),
        "base_cv": float(np.mean(base_scores)), "final_cv": float(np.mean(cur["scores"])),
        "history": history, "elapsed_s": round(time.time() - t0, 1),
    }
    if recheck_folds is not None and kept:
        rf = np.asarray(recheck_folds)
        Fk, _ = materialize(kept, train, None, y, rf)  # target encodings rebuilt on the recheck split
        b = cv_scores(X_base, y, rf, metric, model=model, seed=seed)
        a = cv_scores(pd.concat([X_base, Fk], axis=1), y, rf, metric, model=model, seed=seed)
        res["recheck"] = compare_folds(a["scores"], b["scores"], gib)
    return res


def save_specs(path: str | Path, res: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return p


def load_specs(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["kept"] if isinstance(data, dict) else data


def report(res: dict) -> str:
    lines = [f"# Feature search ({res['model']}, metric {res['metric']})", "",
             f"- candidates screened: {res['n_candidates']} in {len(res['history'])} batches, {res['elapsed_s']}s",
             f"- kept: {len(res['kept'])} features",
             f"- screen CV: {res['base_cv']:.5f} -> {res['final_cv']:.5f}"
             + (f" on folds {res['screen_folds']}" if res.get("screen_folds") else "")]
    rc = res.get("recheck")
    if rc:
        lines.append(f"- recheck on an independent fold split: gain {rc['gain']:+.5f}, folds improved "
                     f"{rc['folds_improved']}/{rc['k']}, t={rc['t']:.2f}"
                     + ("" if rc["gain"] > 0 else "  <- NOT confirmed: the search overfit the screening folds"))
    else:
        lines.append("- no recheck split given: the screen CV is optimistic after many accept/reject decisions")
    kinds: dict[str, int] = {}
    for s in res["kept"]:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    lines.append("- kept by kind: " + (", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "none"))
    lines += ["", "Next: add `materialize(load_specs(...), train, test, y, folds)` to the training script's "
              "add_features(), run the real model, and decide with `kgkit ledger compare`."]
    lines += ["", "## Kept features", ""] + [f"- {spec_name(s)}" for s in res["kept"][:200]]
    return "\n".join(lines) + "\n"
