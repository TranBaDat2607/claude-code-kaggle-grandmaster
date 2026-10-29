"""Is experiment A really better than B? Paired comparisons on the same folds.

The tempting rule "accept if the gain beats one fold std" is the wrong yardstick: the fold std
is dominated by how *hard* each fold is, and that difficulty is shared by both models, so it
cancels in a paired comparison. The right noise scale is the spread of the per-fold
*differences* (and, with OOF predictions, a paired bootstrap over rows). Using the fold std
throws away real gains, which in the late game are exactly the gains you are stacking.

    from kgkit.compare import compare_folds, paired_bootstrap, verdict
    f = compare_folds(new["fold_scores"], base["fold_scores"], greater_is_better=True)
    b = paired_bootstrap(y, oof_new, oof_base, "auc", groups=patient_ids)
    print(verdict(f, b))

CLI: ``python -m kgkit ledger compare <new> <baseline> [--truth data/train.csv:target]``.

What each test does *not* see: the fold-paired test sees training noise only through K
numbers; the bootstrap sees row-sampling noise for *fixed* trained models, not seed noise. A
small gain that passes only the bootstrap deserves a second seed before it becomes the baseline.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Callable, Sequence

import numpy as np

from . import metrics as M
from .evalset import fmt_gain, fmt_se


def compare_folds(new_scores: Sequence[float], base_scores: Sequence[float], greater_is_better: bool = True) -> dict:
    """Paired per-fold comparison. ``gain`` is signed so that positive always means *new is better*."""
    a, b = np.asarray(new_scores, dtype=float), np.asarray(base_scores, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError(f"need per-fold scores of equal length, got {a.shape} vs {b.shape}")
    d = (a - b) if greater_is_better else (b - a)
    k = len(d)
    sd = float(np.std(d, ddof=1)) if k > 1 else float("nan")
    se = sd / math.sqrt(k) if k > 1 else float("nan")
    gain = float(d.mean())
    t = gain / se if k > 1 and se > 0 else (math.copysign(math.inf, gain) if gain else 0.0)
    return {
        "k": k,
        "gain": gain,
        "per_fold": d.tolist(),
        "folds_improved": int((d > 0).sum()),
        "diff_sd": sd,
        "se": se,
        "t": float(t),
        "base_fold_std": float(np.std(b)),
    }


METHODS = ("auto", "bootstrap", "bayes", "clt")
MAX_REDRAWS = 20
SMALL_N = 100
CLT_MIN_UNITS = 1000  # auto: the influence-function (CLT) fast path from this many units; the bootstrap below


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _metric(metric: str | Callable, greater_is_better: bool | None):
    """(registered Metric or None, fn, greater_is_better)."""
    if isinstance(metric, str):
        m = M.get(metric)
        return m, m.fn, m.greater_is_better
    if greater_is_better is None:
        raise ValueError("pass greater_is_better with a callable metric")
    return None, metric, greater_is_better


def _arr(x, dtype=None) -> np.ndarray:
    """An array; per-row lists (ranking targets and predictions) become a 1-D object array."""
    try:
        return np.asarray(x, dtype=dtype)
    except (ValueError, TypeError):
        out = np.empty(len(x), dtype=object)
        out[:] = list(x)
        return out


def _required(y: np.ndarray):
    """What every resample must keep so the metric measures the same thing as on the full set: each label column
    that has both classes (a 0/1 multilabel matrix; a macro average silently drops a column that loses one), or
    two classes of a 1-D class target (AUC is undefined on one). None for a continuous target."""
    try:
        if y.ndim == 2 and y.size and np.isin(y, (0, 1)).all():
            return "cols", np.flatnonzero(y.min(0) != y.max(0))
        if y.ndim == 1 and y.size:
            if y.dtype.kind in "biuOUS" or (y.dtype.kind == "f" and np.allclose(y, np.round(y))):
                if len(np.unique(y)) <= 50:
                    return "classes", None
    except TypeError:  # per-row lists (ranking): nothing to keep
        pass
    return None


def _keeps(y_idx: np.ndarray, req) -> bool:
    kind, cols = req
    if kind == "cols":
        sub = y_idx[:, cols]
        return bool((sub.min(0) != sub.max(0)).all())
    return len(np.unique(y_idx)) > 1


def _units(rows: np.ndarray, groups) -> tuple[list[np.ndarray] | None, np.ndarray | None]:
    """(rows of each resampling unit, unit of each row position); (None, None) when every row is its own unit."""
    if groups is None:
        return None, None
    g = np.asarray(groups)[rows]
    _, inv = np.unique(g, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    return np.split(rows[order], np.cumsum(np.bincount(inv))[:-1]), inv


def estimate(
    y_true,
    pred_new,
    pred_base=None,
    metric: str | Callable = "auc",
    *,
    greater_is_better: bool | None = None,
    method: str = "auto",
    n_boot: int | None = None,
    seed: int = 0,
    groups=None,
    mask=None,
    size: int | None = None,
    rope: float | None = None,
) -> dict:
    """Noise of a score (``pred_base=None``: question ``absolute``) or of a paired difference
    metric(new) - metric(base) (question ``paired``) on one evaluation set, with the exact metric.

    The metric is a black box: any registered or custom metric works, multilabel and ranking ones included.
    Rows are resampled whole per ``groups`` (patients, studies, queries, time blocks) when given. Both
    predictions are scored on the *same* resample, which is what makes it paired: the shared difficulty of the
    rows cancels, so a paired SE is usually far below the SE of one score. A resample that loses a class (or a
    label column its classes) is redrawn, so every resample measures the same metric as the full set.

    ``method``: ``bootstrap``; ``bayes`` (Bayesian bootstrap: Dirichlet weights per unit, so no resample loses
    a class; metrics that take ``sample_weight``); ``clt`` (the metric's influence function, instant at any
    size); ``auto`` = ``clt`` from ``CLT_MIN_UNITS`` units when the metric has one, else ``bootstrap``. A
    method the metric does not support falls back to the bootstrap, with a caveat.

    ``size`` projects to another sample size (e.g. the public LB's rows), assuming that sample looks like the
    evaluation rows. ``rope``: gains within +/- rope are "practically equivalent"; the result then carries the
    probabilities of a gain above, within and below it. ``mask`` restricts to some rows. Seed (training) noise
    is not in this number, fixed predictions are resampled: see :func:`with_seed_noise`.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    m, fn, gib = _metric(metric, greater_is_better)
    paired = pred_base is not None
    num = None if m is not None and m.kind == "ranking" else float
    y, pa = _arr(y_true), _arr(pred_new, num)
    pb = _arr(pred_base, num) if paired else None
    rows = np.arange(len(y)) if mask is None else np.flatnonzero(np.asarray(mask))
    sign = 1.0 if gib else -1.0

    def stat(idx, w=None):
        kw = {} if w is None else {"sample_weight": w}
        if paired:
            return sign * (fn(y[idx], pa[idx], **kw) - fn(y[idx], pb[idx], **kw))
        return float(fn(y[idx], pa[idx], **kw))

    value = float(stat(rows))
    members, inv = _units(rows, groups)
    n_units = len(rows) if members is None else len(members)
    ctx = dict(question="paired" if paired else "absolute", value=value, members=members, n_units=n_units,
               rows=rows, size=size, rope=rope)
    name = m.name if m is not None else "a custom metric"
    caveats = []
    if method == "auto":
        method = "clt" if m is not None and m.influence is not None and n_units >= CLT_MIN_UNITS else "bootstrap"
    if method == "clt":
        if m is None or m.influence is None:
            caveats.append(f"{name} has no influence function: measured by the bootstrap instead")
        else:
            try:
                return _clt(m, y, pa, pb, sign, inv, caveats, **ctx)
            except (NotImplementedError, ValueError, TypeError, IndexError) as e:
                caveats.append(f"influence function failed ({e}): measured by the bootstrap instead")
    rng = np.random.default_rng(seed)
    n_boot = n_boot or (1000 if len(rows) <= 20_000 else 300)
    if method == "bayes":
        if m is None or not m.weighted:
            caveats.append(f"{name} takes no sample weights: measured by the plain bootstrap instead")
        else:
            out, failed = [], 0
            for _ in range(n_boot):
                w = rng.dirichlet(np.ones(n_units))
                try:
                    out.append(stat(rows, w if inv is None else w[inv]))
                except (ValueError, ZeroDivisionError, IndexError):
                    failed += 1
            stats = np.asarray([s for s in out if np.isfinite(s)], dtype=float)
            if size:  # Dirichlet weights describe n units; rescale the spread to the projected sample size
                stats = value + (stats - value) * math.sqrt(len(rows) / size)
            return _result(method="bayesian bootstrap", stats=stats, failed=failed + len(out) - len(stats),
                           caveats=caveats, **ctx)
    m_units = n_units if not size else max(2, int(round(size * n_units / max(1, len(rows)))))
    req = _required(y[rows])
    out, redrawn, kept_degenerate, failed = [], 0, 0, 0
    for _ in range(n_boot):
        for _try in range(MAX_REDRAWS + 1):
            pick = rng.integers(0, n_units, m_units)
            idx = rows[pick] if members is None else np.concatenate([members[i] for i in pick])
            if req is None or _keeps(y[idx], req):
                break
            if _try == MAX_REDRAWS:
                kept_degenerate += 1
            else:
                redrawn += 1
        try:
            s = stat(idx)
        except (ValueError, ZeroDivisionError, IndexError):
            failed += 1
            continue
        if np.isfinite(s):
            out.append(s)
        else:
            failed += 1
    return _result(method="bootstrap", stats=np.asarray(out, dtype=float), redrawn=redrawn,
                   kept_degenerate=kept_degenerate, failed=failed, caveats=caveats, **ctx)


def _clt(m, y, pa, pb, sign, inv, caveats, *, rows, size, **ctx) -> dict:
    """Normal approximation from the metric's influence function: Var = G/(G-1) * sum over units of the squared
    summed contributions (cluster-robust when rows are grouped). Paired: the contributions are differenced per
    row first, so whatever the two models share cancels, as in the paired bootstrap."""
    _, c = m.influence(y[rows], pa[rows])
    c = np.asarray(c, dtype=float)
    if pb is not None:
        c = sign * (c - np.asarray(m.influence(y[rows], pb[rows])[1], dtype=float))
    per_unit = c if inv is None else np.bincount(inv, weights=c)
    g = len(per_unit)
    se = math.sqrt(g / (g - 1) * float(((per_unit - per_unit.mean()) ** 2).sum())) if g > 1 else float("nan")
    if size:
        se *= math.sqrt(len(rows) / size)
    return _result(method="clt", se=se, caveats=caveats, rows=rows, size=size, **ctx)


def _result(question: str, value: float, method: str, members, n_units: int, rows, size, rope=None,
            stats: np.ndarray | None = None, se: float | None = None, redrawn: int = 0, kept_degenerate: int = 0,
            failed: int = 0, caveats=()) -> dict:
    """The common result. ``stats``: resampled values (percentile CI, empirical probabilities); else a normal
    approximation around ``value`` with standard error ``se``."""
    caveats = list(caveats)
    if stats is not None:
        se = float(stats.std(ddof=1)) if len(stats) > 1 else float("nan")
    unit = "rows" if members is None else "groups"
    if n_units < SMALL_N:
        caveats.append(f"small evaluation set ({n_units} {unit}): the interval is approximate")
    if size:
        caveats.append(f"projected to {size} rows: assumes those rows look like the evaluation rows")
    if kept_degenerate:
        caveats.append(f"{kept_degenerate} resamples still lacked a class or label after {MAX_REDRAWS} redraws")
    if failed > 0.05 * max(1, (len(stats) if stats is not None else 0) + failed):
        caveats.append(f"{failed} resamples could not be scored")
    ok = se == se and se > 0
    if stats is not None:
        ci = [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))] if len(stats) else None
    else:
        ci = [value - 1.96 * se, value + 1.96 * se] if ok else None
    res = {
        "question": question,
        "value": float(value),
        "se": se,
        "ci95": ci,
        "method": method,
        "unit": unit,
        "n_units": int(n_units),
        "n_rows": int(len(rows)),
        "size": int(size) if size else None,
        "n_boot": int(len(stats)) if stats is not None else 0,
        "n_redrawn": int(redrawn),
        "n_failed": int(failed),
        "caveats": caveats,
    }
    if question == "paired":
        res["gain"] = float(value)
        res["z"] = float(value / se) if ok else float("nan")
        if stats is not None:
            res["p_better"] = float((stats > 0).mean()) if len(stats) else float("nan")
        else:
            res["p_better"] = _phi(value / se) if ok else float("nan")
        if rope:
            _rope(res, rope, stats)
    return res


def _rope(res: dict, rope: float, stats: np.ndarray | None = None) -> None:
    """P(gain > rope), P(|gain| <= rope), P(gain < -rope): from the resamples, or the normal approximation."""
    rope = abs(float(rope))
    if stats is not None and len(stats):
        above, below = float((stats > rope).mean()), float((stats < -rope).mean())
    elif res["se"] == res["se"] and res["se"] > 0:
        above = 1 - _phi((rope - res["gain"]) / res["se"])
        below = _phi((-rope - res["gain"]) / res["se"])
    else:
        return
    res.update(rope=rope, p_above_rope=above, p_rope=max(0.0, 1 - above - below), p_below_rope=below)


def paired_bootstrap(
    y_true,
    pred_new,
    pred_base,
    metric: str | Callable,
    greater_is_better: bool | None = None,
    n_boot: int = 300,
    seed: int = 0,
    groups=None,
    mask=None,
) -> dict:
    """Paired bootstrap of metric(new) - metric(base) over rows (or over ``groups`` when rows are
    clustered, e.g. several images per patient — resample whole groups, like the CV does).

    ``mask`` restricts the comparison to some rows, e.g. one fold for a 1-fold screen.
    Both predictions are scored on the *same* resample each time, which is what makes it paired.
    See :func:`estimate` for the general form (absolute scores, other sample sizes).
    """
    return estimate(y_true, pred_new, pred_base, metric, greater_is_better=greater_is_better, method="bootstrap",
                    n_boot=n_boot, seed=seed, groups=groups, mask=mask)


def describe(res: dict) -> str:
    """One line that names the question and the method, e.g.
    'paired SE of the difference 0.0034 (bootstrap, 1000 resamples over 58 rows)'."""
    what = "paired SE of the difference" if res["question"] == "paired" else "SE of the score"
    at = f" at n={res['size']}" if res.get("size") else ""
    how = res["method"]
    if how.startswith("clt"):
        how = how.replace("clt", "influence-function CLT", 1)
    how += f", {res['n_boot']} resamples" if "bootstrap" in res["method"] else ""
    src = f"of {res['size']} rows drawn from" if res.get("size") else "over"
    redrawn = f"; {res['n_redrawn']} redrawn to keep every class/label" if res.get("n_redrawn") else ""
    return f"{what}{at} {fmt_se(res['se'])} ({how} {src} {res['n_units']} {res['unit']}{redrawn})"


def verdict(folds: dict | None, boot: dict | None = None, z: float = 2.0) -> dict:
    """KEEP / DISCARD / INCONCLUSIVE from the paired fold test and/or the paired bootstrap.

    KEEP needs a positive gain that is >= ``z`` standard errors from zero in one test, with the
    other test (if available) not pointing the other way and most folds improving.
    A gain far above the baseline's fold std is flagged: audit it for leakage before accepting.
    """
    if folds is None and boot is None:
        raise ValueError("need fold scores or OOF predictions")
    gain = folds["gain"] if folds is not None else boot["gain"]
    reasons, flags = [], []
    fold_strong = folds is not None and folds["k"] > 1 and folds["t"] >= z
    majority = folds is None or folds["folds_improved"] * 2 > folds["k"]
    boot_strong = boot is not None and boot["z"] == boot["z"] and boot["z"] >= z
    boot_against = boot is not None and boot["gain"] < 0
    if folds is not None:
        reasons.append(f"folds: gain {fmt_gain(folds['gain'])}, improved {folds['folds_improved']}/{folds['k']}, "
                       f"paired SE {fmt_se(folds['se'])} (t={folds['t']:.2f}); baseline fold std "
                       f"{fmt_se(folds['base_fold_std'])}")
        if folds["base_fold_std"] > 0 and gain > 3 * folds["base_fold_std"]:
            flags.append("gain > 3x the baseline fold std: audit for leakage (validation-auditor) before accepting")
    if boot is not None:
        reasons.append(f"{boot.get('method', 'bootstrap')}: gain {fmt_gain(boot['gain'])} +/- {fmt_se(boot['se'])} "
                       f"(z={boot['z']:.2f}, P(better)={boot['p_better']:.2f}, "
                       f"{boot.get('n_units', boot['n_rows'])} {boot.get('unit', 'rows')})")
    equivalent = boot is not None and boot.get("p_rope") is not None and boot["p_rope"] >= 0.95
    if boot is not None and boot.get("p_rope") is not None:
        reasons.append(f"ROPE +/-{boot['rope']:g}: P(better by more) {boot['p_above_rope']:.2f}, P(practically "
                       f"equivalent) {boot['p_rope']:.2f}, P(worse by more) {boot['p_below_rope']:.2f}")
    if equivalent:
        decision = "INCONCLUSIVE"
        reasons.append("practically equivalent: keep whichever is simpler or cheaper")
    elif gain <= 0:
        decision = "DISCARD"
    elif (fold_strong or boot_strong) and majority and not boot_against:
        decision = "KEEP"
    else:
        decision = "INCONCLUSIVE"
        reasons.append("positive but within noise: rerun with a second seed (or more folds) before "
                       "accepting, or keep only if it is free (no extra complexity / runtime)")
    return {"decision": decision, "gain": gain, "reasons": reasons, "flags": flags}


def format_comparison(new_id: str, base_id: str, res: dict) -> str:
    lines = [f"{new_id} vs {base_id}: {res['decision']} (gain {fmt_gain(res['gain'])}, positive = better)"]
    lines += [f"  {r}" for r in res["reasons"]]
    lines += [f"  WARNING: {f}" for f in res["flags"]]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- seed (training) noise
SEED_KEYS = {"seed", "seeds", "random_state", "random_seed", "name", "notes", "run_name", "exp_name"}


def replica_key(rec: dict) -> str | None:
    """Records that differ only in their seed(s) share a key: the explicit ``variant`` (``ledger add --variant``),
    else the model, its params without seed keys and its features, on one clean git commit. None when unknown."""
    if rec.get("variant"):
        return f"v:{rec['variant']}"
    params = rec.get("params") or {}
    if not params or not rec.get("git_commit") or rec.get("git_dirty"):
        return None
    core = {k: v for k, v in params.items() if k not in SEED_KEYS}
    blob = json.dumps({"model": rec.get("model"), "params": core, "features": rec.get("features"),
                       "n_features": rec.get("n_features"), "commit": rec["git_commit"]}, sort_keys=True, default=str)
    return "p:" + hashlib.sha1(blob.encode()).hexdigest()[:12]


def n_seeds(rec: dict) -> int:
    """How many seeds a record's predictions average (templates log ``seeds``)."""
    return len(rec.get("seeds") or ()) or int(rec.get("n_seeds") or 1)


def seed_variance(records: Sequence[dict], eval_key: Callable[[dict], str | None] | None = None) -> dict | None:
    """Pooled single-seed SD of the score across replicas: records of one pipeline that differ only in their
    seed, scored on the same evaluation set. None without at least one pipeline run twice."""
    from .evalset import own_score

    groups: dict = defaultdict(list)
    for r in records:
        k = replica_key(r)
        if k is None or not own_score(r) or r.get("recheck"):
            continue
        groups[(k, eval_key(r) if eval_key else None, n_seeds(r))].append(float(r["cv"]))
    ss = df = 0.0
    variants = 0
    for (_, _, r_seeds), scores in groups.items():
        if len(scores) > 1:
            ss += float(np.var(scores, ddof=1)) * (len(scores) - 1) * r_seeds  # back to a single seed's variance
            df += len(scores) - 1
            variants += 1
    if not df:
        return None
    return {"sd": math.sqrt(ss / df), "df": int(df), "n_variants": variants}


def with_seed_noise(res: dict, seed: dict, r_new: int = 1, r_base: int = 1) -> dict:
    """Add training noise to a paired result: se_total^2 = se^2 + sd_seed^2 (1/r_new + 1/r_base), where r is
    the number of seeds each side averages. The resampling SE stays in ``se_rows``."""
    out = dict(res, caveats=list(res.get("caveats") or ()))
    se = math.sqrt(res["se"] ** 2 + seed["sd"] ** 2 * (1 / max(1, r_new) + 1 / max(1, r_base)))
    gain = res["gain"]
    out.update(se=se, se_rows=res["se"], z=gain / se if se > 0 else float("nan"),
               p_better=_phi(gain / se) if se > 0 else float("nan"),
               ci95=[gain - 1.96 * se, gain + 1.96 * se],
               method=f"{res['method']} + seed noise (sd {fmt_se(seed['sd'])}, df {seed['df']})")
    for k in ("rope", "p_rope", "p_above_rope", "p_below_rope"):
        out.pop(k, None)
    if res.get("rope"):
        _rope(out, res["rope"])
    return out
