"""Per-unit scores: paired comparisons for metrics that are not one array of row predictions (#136).

``ledger compare`` resamples stored OOF arrays. For segmentation (Dice / IoU), detection (F-beta at an IoU or a
distance), particle picking (F-beta per tomogram and particle type), multi-stage medical pipelines, LLM problem sets or
simulation matches, the OOF is huge or not an array at all. Such a record carries a *unit table* instead: one row per
thing the metric is computed over (an image, a study, a tomogram, a query, a problem, a match) with the values the
metric is built from. Two tables of the same units are compared paired, resampling units (or whole groups of them:
patients, sites) and scoring both on the same draw, as the OOF bootstrap does with rows::

    from kgkit import units as U
    u = U.make(tomo_ids, counts={"tp": tp, "fp": fp, "fn": fn}, params={"beta": 4}, fold=folds)
    led.log_units("unet_v3", u, parent="0012")          # CV, fold scores and the table, from the table
    U.compare(u, led.load_units("0012"))["z"]           # paired, positive = better

Aggregates (the column names say which one a table uses; ``agg`` names it explicitly):

    mean       score [, weight]          weighted mean of per-unit scores: accuracy per problem, Dice or IoU per image,
                                         MAE per series, NDCG@k per query, win rate per match
    ratio      num, den                  sum(num) / sum(den)
    iou        inter, union              dataset-level IoU from summed pixel counts (empty = 1)
    dice       inter, pred, true         dataset-level Dice, 2 sum(inter) / (sum(pred) + sum(true)) (empty = 1)
    fbeta      tp, fp, fn                micro F-beta from summed counts (params: beta, default 1). With columns
                                         tp_<class>, fp_<class>, fn_<class>: F-beta per class from its summed counts,
                                         then their mean weighted by params class_weights (default equal): CZII-style
    precision  tp, fp                    from summed counts
    recall     tp, fn                    from summed counts
    custom     any numeric columns       params fn = "module:function" called as fn(cols, w) -> float, ``w`` the draw
                                         weight of each unit (0 = not drawn); greater_is_better must be given

A count-based aggregate is exact, not an approximation: the metric of a resample is recomputed from its summed counts.
A metric that is no function of per-unit values (a global ranking over all detections) goes through ``custom`` with
rows that hold what it needs (one row per detection, ``group`` = its image) and is resampled per group.

Reserved columns: ``id`` (required, unique), ``group`` (resampled whole) and ``fold`` (per-fold pairing).
"""

from __future__ import annotations

import hashlib
import importlib
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

RESERVED = ("id", "group", "fold")
AGGS = ("mean", "ratio", "iou", "dice", "fbeta", "precision", "recall", "custom")
_NEEDS = {"mean": ("score",), "ratio": ("num", "den"), "iou": ("inter", "union"), "dice": ("inter", "pred", "true"),
          "fbeta": ("tp", "fp", "fn"), "precision": ("tp", "fp"), "recall": ("tp", "fn"), "custom": ()}
CHUNK = 20_000_000  # resamples x units per block of the weight matrix


def _div(num, den, empty=np.nan):
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, float(empty))
    np.divide(num, den, out=out, where=den != 0)
    return out


def _fbeta(t, p, n, beta: float):
    b2 = float(beta) ** 2
    return _div((1 + b2) * t, (1 + b2) * t + b2 * n + p, empty=1.0)  # nothing to find, nothing claimed: perfect


def _classes(cols: dict) -> list[str]:
    return sorted(c[3:] for c in cols if c.startswith("tp_") and len(c) > 3)


def infer(cols) -> str:
    """The aggregate a table's columns imply."""
    names = set(cols)
    if "score" in names:
        return "mean"
    if {"tp", "fp", "fn"} <= names or _classes(names):
        return "fbeta"
    if {"inter", "union"} <= names:
        return "iou"
    if {"inter", "pred", "true"} <= names:
        return "dice"
    if {"num", "den"} <= names:
        return "ratio"
    raise ValueError(f"cannot tell the aggregate from the columns {sorted(names)}: name it (agg=..., one of "
                     f"{', '.join(AGGS)}), or use the column names of one (score; tp/fp/fn; inter/union; ...)")


def _custom(params: dict) -> Callable:
    spec = params.get("fn")
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError('custom needs params fn = "module:function", called as fn(cols, w) -> float')
    mod, _, name = spec.rpartition(":")
    added = [p for p in (str(Path.cwd()), str(Path.cwd() / "src")) if p not in sys.path]
    sys.path[:0] = added
    try:
        return getattr(importlib.import_module(mod), name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"custom aggregate {spec}: {e}")
    finally:
        for p in added:
            sys.path.remove(p)


@dataclass
class Units:
    """One record's unit table. ``ids`` are strings (numbers are written as text, so a table read back pairs with
    the one that was logged); ``cols`` the numeric columns its aggregate is built from."""
    ids: np.ndarray
    cols: dict
    agg: str
    params: dict = field(default_factory=dict)
    greater_is_better: bool | None = None
    group: np.ndarray | None = None
    fold: np.ndarray | None = None

    def __post_init__(self):
        self.ids = np.asarray([str(i) for i in np.asarray(self.ids).ravel()])
        n = len(self.ids)
        if n == 0:
            raise ValueError("a unit table needs at least one unit")
        if len(set(self.ids.tolist())) != n:
            raise ValueError("duplicate unit ids: each unit appears once (resample whole groups with group=...)")
        cols = {}
        for k, v in self.cols.items():
            arr = np.asarray(v, dtype=float).ravel()
            if len(arr) != n:
                raise ValueError(f"column {k} has {len(arr)} values for {n} units")
            if not np.isfinite(arr).all():
                raise ValueError(f"column {k} has missing or infinite values")
            cols[str(k)] = arr
        self.cols = cols
        if self.group is not None:
            self.group = np.asarray([str(g) for g in np.asarray(self.group).ravel()])
        if self.fold is not None:
            self.fold = np.asarray(self.fold).astype(int).ravel()
        for name, arr in (("group", self.group), ("fold", self.fold)):
            if arr is not None and len(arr) != n:
                raise ValueError(f"{name} has {len(arr)} values for {n} units")
        if self.agg not in AGGS:
            raise ValueError(f"unknown aggregate {self.agg!r}: one of {', '.join(AGGS)}")
        self.params = dict(self.params or {})
        if self.agg == "fbeta" and "tp" not in self.cols and _classes(self.cols):
            missing = [f"{k}_{c}" for c in _classes(self.cols) for k in ("fp", "fn") if f"{k}_{c}" not in self.cols]
            weights = self.params.get("class_weights") or {}
            unknown = [c for c in _classes(self.cols) if weights and c not in weights]
            if unknown:
                raise ValueError(f"class_weights has no weight for {', '.join(unknown)}")
        else:
            missing = [c for c in _NEEDS[self.agg] if c not in self.cols]
        if missing:
            raise ValueError(f"{self.agg} needs the columns {', '.join(_NEEDS[self.agg])}"
                             + (" (or tp_<class>, fp_<class>, fn_<class>)" if self.agg == "fbeta" else "")
                             + f"; missing: {', '.join(missing)}")
        if self.agg == "custom":
            _custom(self.params)
            if self.greater_is_better is None:
                raise ValueError("a custom aggregate needs greater_is_better: kgkit cannot tell its direction")

    # ------------------------------------------------------------------ scores
    @property
    def gib(self) -> bool:
        return True if self.greater_is_better is None else bool(self.greater_is_better)

    def scores(self, w: np.ndarray) -> np.ndarray:
        """The aggregate under each row of ``w`` (draws x units): the weight of each unit in that draw."""
        w = np.atleast_2d(np.asarray(w, dtype=float))
        c, p = self.cols, self.params
        if self.agg == "custom":
            fn = _custom(p)
            return np.asarray([float(fn(c, row)) for row in w], dtype=float)

        def s(name):
            return w @ c[name]

        if self.agg == "mean":
            weight = c.get("weight", np.ones(len(self.ids)))
            return _div(w @ (c["score"] * weight), w @ weight)
        if self.agg == "ratio":
            return _div(s("num"), s("den"))
        if self.agg == "iou":
            return _div(s("inter"), s("union"), empty=1.0)
        if self.agg == "dice":
            return _div(2 * s("inter"), s("pred") + s("true"), empty=1.0)
        if self.agg == "precision":
            return _div(s("tp"), s("tp") + s("fp"))
        if self.agg == "recall":
            return _div(s("tp"), s("tp") + s("fn"))
        beta = float(p.get("beta", 1))
        if "tp" in c:
            return _fbeta(s("tp"), s("fp"), s("fn"), beta)
        weights = p.get("class_weights") or {}
        classes = _classes(c)
        total = sum(float(weights.get(k, 1)) for k in classes)
        return sum(float(weights.get(k, 1)) * _fbeta(s(f"tp_{k}"), s(f"fp_{k}"), s(f"fn_{k}"), beta)
                   for k in classes) / total

    def score(self, w=None) -> float:
        return float(self.scores(np.ones(len(self.ids)) if w is None else w)[0])

    def fold_scores(self) -> list[float] | None:
        if self.fold is None:
            return None
        return [self.score((self.fold == k).astype(float)) for k in sorted(set(self.fold.tolist()))]

    def digest(self) -> str:
        """Fingerprint of the unit set (not its order): two tables compare only on the same units."""
        return hashlib.sha1("\n".join(sorted(self.ids.tolist())).encode()).hexdigest()[:16]

    def reordered(self, ids: np.ndarray) -> "Units":
        pos = {u: i for i, u in enumerate(self.ids.tolist())}
        idx = np.asarray([pos[u] for u in np.asarray(ids).tolist()])
        return Units(self.ids[idx], {k: v[idx] for k, v in self.cols.items()}, self.agg, self.params,
                     self.greater_is_better, None if self.group is None else self.group[idx],
                     None if self.fold is None else self.fold[idx])

    # ------------------------------------------------------------------ files
    def spec(self) -> dict:
        out = {"agg": self.agg, "params": self.params, "greater_is_better": self.gib, "n": len(self.ids),
               "hash": self.digest(), "columns": sorted(self.cols)}
        if self.group is not None:
            out["n_groups"] = len(set(self.group.tolist()))
        if self.fold is not None:
            out["n_folds"] = len(set(self.fold.tolist()))
        return out

    def save(self, path) -> dict:
        """Write the table (.npz, no pickled objects) and return its spec for the ledger record."""
        arrays = {"id": self.ids, **self.cols}
        if self.group is not None:
            arrays["group"] = self.group
        if self.fold is not None:
            arrays["fold"] = self.fold
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.savez_compressed(f, **arrays)
        return self.spec()


def make(ids, *, score=None, weight=None, counts: dict | None = None, agg: str | None = None,
         params: dict | None = None, group=None, fold=None, greater_is_better: bool | None = None) -> Units:
    """A unit table from per-unit ``score`` (mean, optional ``weight``) or ``counts`` (columns of summed values)."""
    cols = dict(counts or {})
    if score is not None:
        cols["score"] = score
        if weight is not None:
            cols["weight"] = weight
    return Units(np.asarray(list(ids)), cols, agg or infer(cols), params or {}, greater_is_better, group, fold)


def load(path, spec: dict) -> Units:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return Units(arrays.pop("id"), {k: v for k, v in arrays.items() if k not in RESERVED}, spec["agg"],
                 spec.get("params") or {}, spec.get("greater_is_better"), arrays.get("group"), arrays.get("fold"))


def read(path, agg: str | None = None, params: dict | None = None, greater_is_better: bool | None = None) -> Units:
    """A unit table from a file: CSV / TSV / parquet with an ``id`` column, or an .npz written by :meth:`Units.save`."""
    p = Path(path)
    if p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    else:
        import pandas as pd

        df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(
            p, sep="\t" if p.suffix.lower() == ".tsv" else ",")
        bad = [c for c in df.columns if c not in RESERVED and not pd.api.types.is_numeric_dtype(df[c])]
        if bad:
            raise ValueError(f"{p}: columns other than id/group/fold must be numeric; not: {', '.join(map(str, bad))}")
        arrays = {str(c): df[c].to_numpy() for c in df.columns}
    if "id" not in arrays:
        raise ValueError(f"{p}: a unit table needs an id column (one row per image, study, tomogram, query, ...)")
    cols = {k: v for k, v in arrays.items() if k not in RESERVED}
    return Units(arrays["id"], cols, agg or infer(cols), params or {}, greater_is_better, arrays.get("group"),
                 arrays.get("fold"))


# ---------------------------------------------------------------------------- paired comparison
def check_pair(ua: Units, ub: Units) -> Units:
    """``ub`` in ``ua``'s unit order; ValueError when the two are not comparable."""
    if (ua.agg, ua.params) != (ub.agg, ub.params):
        raise ValueError(f"the tables use a different aggregate ({ua.agg} {ua.params or ''} vs {ub.agg} "
                         f"{ub.params or ''}): they score different things")
    a, b = set(ua.ids.tolist()), set(ub.ids.tolist())
    if a != b:
        raise ValueError(f"the tables do not hold the same units ({len(a - b)} only in the first, {len(b - a)} only "
                         "in the second): compare runs scored on the same units (a screen with its control on the "
                         "same subset)")
    return ub.reordered(ua.ids)


def paired_folds(ua: Units, ub: Units) -> tuple[list[float], list[float]] | None:
    """Per-fold scores of both tables when they assign every unit to the same fold, with two folds or more."""
    if ua.fold is None or ub.fold is None:
        return None
    ub = check_pair(ua, ub)
    if not np.array_equal(ua.fold, ub.fold) or len(set(ua.fold.tolist())) < 2:
        return None
    fa, fb = ua.fold_scores(), ub.fold_scores()
    if not all(math.isfinite(x) for x in [*fa, *fb]):
        return None
    return fa, fb


def compare(ua: Units, ub: Units | None = None, *, n_boot: int | None = None, seed: int = 0,
            rope: float | None = None) -> dict:
    """Noise of one table's score (``ub=None``) or of the paired gain ``ua - ub`` (positive = better), by
    resampling units, or whole groups when the first table has a ``group`` column. Same result fields as
    :func:`kgkit.compare.estimate`."""
    from .compare import _result

    paired = ub is not None
    if paired:
        ub = check_pair(ua, ub)
    sign = 1.0 if ua.gib else -1.0
    n = len(ua.ids)
    group = ua.group if ua.group is not None else (ub.group if paired else None)
    inv = None
    if group is not None:
        _, inv = np.unique(group, return_inverse=True)
    n_units = n if inv is None else int(inv.max()) + 1
    n_boot = n_boot or (1000 if n <= 20_000 else 300)
    value = sign * (ua.score() - ub.score()) if paired else ua.score()
    rng = np.random.default_rng(seed)
    out, failed, done = [], 0, 0
    block = max(1, min(n_boot, CHUNK // max(1, n)))
    while done < n_boot:
        k = min(block, n_boot - done)
        w = rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=k).astype(float)
        if inv is not None:
            w = w[:, inv]
        s = sign * (ua.scores(w) - ub.scores(w)) if paired else ua.scores(w)
        ok = np.isfinite(s)
        failed += int((~ok).sum())
        out.append(s[ok])
        done += k
    stats = np.concatenate(out) if out else np.asarray([], dtype=float)
    caveats = [] if ua.agg != "custom" else [f"custom aggregate {ua.params.get('fn')}: exact if it is a function of "
                                             "the drawn units' rows"]
    return _result(question="paired" if paired else "absolute", value=value, method="unit bootstrap",
                   members=None if inv is None else [], n_units=n_units, rows=np.arange(n), size=None, rope=rope,
                   stats=stats, failed=failed, caveats=caveats, unit="units" if inv is None else "groups")
