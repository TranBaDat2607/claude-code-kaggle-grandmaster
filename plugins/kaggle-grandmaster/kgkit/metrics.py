"""A registry of Kaggle evaluation metrics.

Every metric knows its direction and what it consumes, so generic code (hill climbing,
early stopping, the ledger) never has to guess::

    from kgkit import metrics
    m = metrics.get("qwk")
    m.greater_is_better, m.kind   # (True, 'label')
    metrics.score("rmsle", y, pred)

``kind`` is one of:
    value  – continuous predictions (regression)
    proba  – probabilities / scores (ranking metrics like AUC accept any monotone score)
    label  – hard class labels (threshold / argmax first; see kgkit.thresholds)
    ranking – per-row ranked lists of predicted items (MAP@K, NDCG@K, recall@K)
    mask   – binary masks (segmentation)

Always re-implement the *exact* competition metric (read the evaluation page and the
host's metric notebook) and test it against a known value before trusting CV.

Two optional fields let ``kgkit.compare.estimate`` measure a metric's noise faster; any metric,
custom ones included, works without them through the plain bootstrap:
    influence – ``(y, p) -> (score, c)``, ``c`` the per-row linearised contribution: the variance of
                the score is about the sum of squared ``c`` (summed per group first). Means of per-row
                or per-query scores, smooth functions of one (RMSE, delta method), AUC (DeLong).
    weighted  – the function takes ``sample_weight`` (Bayesian bootstrap: no resample ever loses a class)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats
from sklearn import metrics as skm


@dataclass(frozen=True)
class Metric:
    name: str
    fn: Callable[..., float]
    greater_is_better: bool
    kind: str
    description: str = ""
    influence: Callable | None = None
    weighted: bool = False

    def __call__(self, y_true, y_pred, **kw) -> float:
        return float(self.fn(y_true, y_pred, **kw))

    def is_better(self, a: float, b: float) -> bool:
        """True if score ``a`` beats score ``b``."""
        return a > b if self.greater_is_better else a < b

    def worst(self) -> float:
        return -np.inf if self.greater_is_better else np.inf


_REGISTRY: dict[str, Metric] = {}
_ALIASES: dict[str, str] = {}


def register(name, greater_is_better, kind, description="", aliases=(), influence=None, weighted=False):
    def deco(fn):
        _REGISTRY[name] = Metric(name, fn, greater_is_better, kind, description, influence, weighted)
        for a in aliases:
            _ALIASES[a] = name
        return fn

    return deco


def get(name: str) -> Metric:
    key = name.lower().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(key, key)
    if key not in _REGISTRY:
        raise KeyError(f"unknown metric {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def score(name: str, y_true, y_pred, **kw) -> float:
    return get(name)(y_true, y_pred, **kw)


def list_metrics() -> list[Metric]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def _a(x) -> np.ndarray:
    return np.asarray(x)


def _rows(x) -> np.ndarray:
    """Per-row values (extra dimensions, e.g. several outputs or labels per row, averaged)."""
    x = np.asarray(x, dtype=float)
    return x.reshape(len(x), -1).mean(1) if x.ndim > 1 else x


def _mean_infl(per_row):
    """Influence of a metric that is the mean of per-row (per-query) scores."""
    def influence(y, p):
        u = _rows(per_row(y, p))
        theta = float(u.mean())
        return theta, (u - theta) / len(u)

    return influence


def _sqrt_mean_infl(per_row):
    """Influence of the square root of a mean of per-row scores, e.g. RMSE (delta method)."""
    def influence(y, p):
        u = _rows(per_row(y, p))
        m = float(u.mean())
        theta = m ** 0.5
        return theta, ((u - m) / len(u) / (2 * theta) if theta > 0 else np.zeros(len(u)))

    return influence


# ----------------------------------------------------------------------------- regression
def _sq_err(y, p):
    return (_a(y) - _a(p)) ** 2


def _abs_err(y, p):
    return np.abs(_a(y) - _a(p))


def _sq_log_err(y, p):
    p = np.clip(_a(p).astype(float), 0, None)
    return (np.log1p(_a(y)) - np.log1p(p)) ** 2


def _smape_rows(y, p):
    y, p = _a(y).astype(float), _a(p).astype(float)
    denom = np.abs(y) + np.abs(p)
    return np.where(denom == 0, 0.0, 2 * np.abs(p - y) / np.where(denom == 0, 1, denom)) * 100


@register("rmse", False, "value", "root mean squared error", influence=_sqrt_mean_infl(_sq_err), weighted=True)
def rmse(y, p, sample_weight=None):
    return np.sqrt(np.average(_rows(_sq_err(y, p)), weights=sample_weight))


@register("mse", False, "value", "mean squared error", influence=_mean_infl(_sq_err), weighted=True)
def mse(y, p, sample_weight=None):
    return np.average(_rows(_sq_err(y, p)), weights=sample_weight)


@register("mae", False, "value", "mean absolute error (optimal constant: median)", influence=_mean_infl(_abs_err),
          weighted=True)
def mae(y, p, sample_weight=None):
    return np.average(_rows(_abs_err(y, p)), weights=sample_weight)


@register("medae", False, "value", "median absolute error")
def medae(y, p):
    return np.median(np.abs(_a(y) - _a(p)))


@register("rmsle", False, "value", "RMSE on log1p — train on log1p(y) with RMSE", aliases=("rmsle_",),
          influence=_sqrt_mean_infl(_sq_log_err), weighted=True)
def rmsle(y, p, sample_weight=None):
    return np.sqrt(np.average(_rows(_sq_log_err(y, p)), weights=sample_weight))


@register("msle", False, "value", "mean squared log error", influence=_mean_infl(_sq_log_err), weighted=True)
def msle(y, p, sample_weight=None):
    return np.average(_rows(_sq_log_err(y, p)), weights=sample_weight)


@register("mape", False, "value", "mean absolute percentage error (zeros in y excluded)")
def mape(y, p):
    y, p = _a(y).astype(float), _a(p).astype(float)
    m = y != 0
    return np.mean(np.abs((y[m] - p[m]) / y[m])) * 100


@register("smape", False, "value", "symmetric MAPE in percent (0/0 counted as 0)", influence=_mean_infl(_smape_rows),
          weighted=True)
def smape(y, p, sample_weight=None):
    return np.average(_rows(_smape_rows(y, p)), weights=sample_weight)


@register("r2", True, "value", "coefficient of determination", weighted=True)
def r2(y, p, sample_weight=None):
    return skm.r2_score(y, p, sample_weight=sample_weight)


@register("pearson", True, "value", "Pearson correlation")
def pearson(y, p):
    return stats.pearsonr(_a(y).ravel(), _a(p).ravel())[0]


@register("spearman", True, "value", "Spearman rank correlation")
def spearman(y, p):
    return stats.spearmanr(_a(y).ravel(), _a(p).ravel())[0]


# ------------------------------------------------------------------ probabilistic / scores
def _binary(y, s):
    y, s = _a(y).ravel(), _a(s).astype(float).ravel()
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"ROC AUC needs exactly two classes in y_true, got {len(classes)}")
    if not np.isfinite(s).all():
        raise ValueError("ROC AUC: scores contain NaN or inf")
    return y == classes[1], s


def binary_auc(y, s, sample_weight=None) -> float:
    """Binary ROC AUC from midranks (the Mann-Whitney U); equal to sklearn's ``roc_auc_score`` with ties, ~20x
    faster on small sets, which matters inside a bootstrap. The larger label is the positive class."""
    pos, s = _binary(y, s)
    if sample_weight is not None:  # weighted pairs: negatives below each positive, ties counted half
        w = _a(sample_weight).astype(float).ravel()
        _, inv = np.unique(s, return_inverse=True)
        wp = np.bincount(inv, weights=np.where(pos, w, 0.0))
        wn = np.bincount(inv, weights=np.where(pos, 0.0, w))
        return float((wp * (np.cumsum(wn) - wn + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
    n_pos = int(pos.sum())
    r = stats.rankdata(s)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(s) - n_pos)))


def _auc_infl(y, s):
    """DeLong placement values: each positive's share of the negatives scored below it, each negative's share
    of the positives scored above it (ties count half). Their means are the AUC."""
    pos, s = _binary(y, s)
    m, n = int(pos.sum()), int((~pos).sum())
    r = stats.rankdata(s)
    v = np.empty(len(s))
    v[pos] = (r[pos] - stats.rankdata(s[pos])) / n
    v[~pos] = 1 - (r[~pos] - stats.rankdata(s[~pos])) / m
    theta = float(v[pos].mean())
    return theta, np.where(pos, (v - theta) / m, (v - theta) / n)


def _auc_influence(y, p):
    p = _a(p)
    if p.ndim == 2 and p.shape[1] > 1:
        raise NotImplementedError("one-vs-rest multiclass AUC has no influence function here: bootstrap it")
    return _auc_infl(y, p)


def _gini_influence(y, p):
    theta, c = _auc_infl(y, p)
    return 2 * theta - 1, 2 * c


def _mean_col_auc_influence(y, p):
    y, p = _a(y), _a(p)
    parts = [_auc_infl(y[:, j], p[:, j]) for j in range(y.shape[1]) if len(np.unique(y[:, j])) > 1]
    return float(np.mean([t for t, _ in parts])), np.mean([c for _, c in parts], axis=0)


@register("auc", True, "proba", "ROC AUC (binary, or multiclass one-vs-rest macro)", aliases=("roc_auc", "rocauc"),
          influence=_auc_influence, weighted=True)
def auc(y, p, sample_weight=None):
    p = _a(p)
    if p.ndim == 2 and p.shape[1] > 1:
        return skm.roc_auc_score(y, p, multi_class="ovr", average="macro", sample_weight=sample_weight)
    return binary_auc(y, p, sample_weight)


@register("gini", True, "proba", "normalized Gini = 2*AUC - 1", aliases=("normalized_gini",),
          influence=_gini_influence, weighted=True)
def gini(y, p, sample_weight=None):
    return 2 * binary_auc(y, p, sample_weight) - 1


@register("pr_auc", True, "proba", "average precision", aliases=("average_precision", "ap"), weighted=True)
def pr_auc(y, p, sample_weight=None):
    return skm.average_precision_score(y, p, sample_weight=sample_weight)


@register("mean_col_auc", True, "proba", "mean column-wise ROC AUC (multilabel)", aliases=("mcauc",),
          influence=_mean_col_auc_influence, weighted=True)
def mean_col_auc(y, p, sample_weight=None):
    y, p = _a(y), _a(p)
    cols = [binary_auc(y[:, j], p[:, j], sample_weight) for j in range(y.shape[1]) if len(np.unique(y[:, j])) > 1]
    return float(np.mean(cols))


def _logloss_rows(y, p, eps=1e-15):
    p = np.clip(_a(p).astype(float), eps, 1 - eps)
    y = _a(y)
    if p.ndim == 2 and p.shape[1] > 1:
        p = p / p.sum(axis=1, keepdims=True)
        return -np.log(p[np.arange(len(y)), y.astype(int)])
    p = p.ravel()
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


@register("logloss", False, "proba", "binary / multiclass log loss (clipped 1e-15)", aliases=("log_loss", "cross_entropy", "bce"),
          influence=_mean_infl(_logloss_rows), weighted=True)
def logloss(y, p, eps=1e-15, sample_weight=None):
    return np.average(_rows(_logloss_rows(y, p, eps)), weights=sample_weight)


@register("balanced_logloss", False, "proba", "class-balanced log loss (each class weighted equally)")
def balanced_logloss(y, p, eps=1e-15):
    y = _a(y).astype(int)
    p = np.clip(_a(p).astype(float), eps, 1 - eps)
    if p.ndim == 1 or p.shape[1] == 1:
        p = p.ravel()
        p = np.stack([1 - p, p], axis=1)
    p = p / p.sum(axis=1, keepdims=True)
    losses = [-np.mean(np.log(p[y == c, c])) for c in range(p.shape[1]) if (y == c).any()]
    return float(np.mean(losses))


def _brier_rows(y, p):
    return (_a(y).astype(float).ravel() - _a(p).astype(float).ravel()) ** 2


@register("brier", False, "proba", "Brier score (binary)", influence=_mean_infl(_brier_rows), weighted=True)
def brier(y, p, sample_weight=None):
    return skm.brier_score_loss(y, _a(p).ravel(), sample_weight=sample_weight)


def _multilabel_logloss_rows(y, p, eps=1e-15):
    y = _a(y).astype(float)
    p = np.clip(_a(p).astype(float), eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


@register("multilabel_logloss", False, "proba", "mean binary log loss over label columns",
          influence=_mean_infl(_multilabel_logloss_rows), weighted=True)
def multilabel_logloss(y, p, eps=1e-15, sample_weight=None):
    return np.average(_rows(_multilabel_logloss_rows(y, p, eps)), weights=sample_weight)


# -------------------------------------------------------------------------------- labels
def _correct(y, p):
    eq = _a(y) == _a(p)
    return eq.all(axis=1) if eq.ndim > 1 else eq


@register("accuracy", True, "label", "accuracy", aliases=("acc",), influence=_mean_infl(_correct), weighted=True)
def accuracy(y, p, sample_weight=None):
    return skm.accuracy_score(y, p, sample_weight=sample_weight)


@register("balanced_accuracy", True, "label", "balanced accuracy", weighted=True)
def balanced_accuracy(y, p, sample_weight=None):
    return skm.balanced_accuracy_score(y, p, sample_weight=sample_weight)


@register("f1", True, "label", "binary F1 (tune the threshold!)", weighted=True)
def f1(y, p, sample_weight=None):
    return skm.f1_score(y, p, sample_weight=sample_weight)


@register("f1_macro", True, "label", "macro F1", weighted=True)
def f1_macro(y, p, sample_weight=None):
    return skm.f1_score(y, p, average="macro", sample_weight=sample_weight)


@register("f1_micro", True, "label", "micro F1", weighted=True)
def f1_micro(y, p, sample_weight=None):
    return skm.f1_score(y, p, average="micro", sample_weight=sample_weight)


@register("f1_weighted", True, "label", "weighted F1", weighted=True)
def f1_weighted(y, p, sample_weight=None):
    return skm.f1_score(y, p, average="weighted", sample_weight=sample_weight)


@register("fbeta2", True, "label", "binary F-beta with beta=2 (recall-heavy)", aliases=("f2",), weighted=True)
def fbeta2(y, p, sample_weight=None):
    return skm.fbeta_score(y, p, beta=2, sample_weight=sample_weight)


@register("mcc", True, "label", "Matthews correlation coefficient", weighted=True)
def mcc(y, p, sample_weight=None):
    return skm.matthews_corrcoef(y, p, sample_weight=sample_weight)


@register("qwk", True, "label", "quadratic weighted kappa (optimise rounding thresholds!)",
          aliases=("quadratic_weighted_kappa", "kappa_quadratic"), weighted=True)
def qwk(y, p, sample_weight=None):
    return skm.cohen_kappa_score(_a(y).astype(int), _a(p).astype(int), weights="quadratic",
                                 sample_weight=sample_weight)


@register("cohen_kappa", True, "label", "unweighted Cohen's kappa", weighted=True)
def cohen_kappa(y, p, sample_weight=None):
    return skm.cohen_kappa_score(y, p, sample_weight=sample_weight)


# ------------------------------------------------------------------------------- ranking
def apk(actual: Sequence, predicted: Sequence, k: int = 10) -> float:
    if not len(actual):
        return 0.0
    predicted = list(predicted)[:k]
    actual_set = set(actual)
    hits, total, seen = 0, 0.0, set()
    for i, p in enumerate(predicted):
        if p in actual_set and p not in seen:
            hits += 1
            total += hits / (i + 1.0)
        seen.add(p)
    return total / min(len(actual_set), k)


def _apk_rows(k):
    return lambda actual, predicted: [apk(a, p, k) for a, p in zip(actual, predicted)]


def _mapk(actual, predicted, k, sample_weight=None):
    return float(np.average(_apk_rows(k)(actual, predicted), weights=sample_weight))


@register("map@5", True, "ranking", "mean average precision at 5", aliases=("map5",),
          influence=_mean_infl(_apk_rows(5)), weighted=True)
def map5(actual, predicted, sample_weight=None):
    return _mapk(actual, predicted, 5, sample_weight)


@register("map@10", True, "ranking", "mean average precision at 10", aliases=("map10",),
          influence=_mean_infl(_apk_rows(10)), weighted=True)
def map10(actual, predicted, sample_weight=None):
    return _mapk(actual, predicted, 10, sample_weight)


@register("map@12", True, "ranking", "mean average precision at 12 (H&M-style)", aliases=("map12",),
          influence=_mean_infl(_apk_rows(12)), weighted=True)
def map12(actual, predicted, sample_weight=None):
    return _mapk(actual, predicted, 12, sample_weight)


def mapk(actual, predicted, k=10) -> float:
    return _mapk(actual, predicted, k)


def _ndcg_rows(actual, predicted, k=10) -> list[float]:
    scores = []
    for a, p in zip(actual, predicted):
        a = set(a)
        dcg = sum(1.0 / np.log2(i + 2) for i, x in enumerate(list(p)[:k]) if x in a)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(a), k)))
        scores.append(dcg / idcg if idcg else 0.0)
    return scores


def ndcgk(actual, predicted, k=10) -> float:
    return float(np.mean(_ndcg_rows(actual, predicted, k)))


@register("ndcg@10", True, "ranking", "binary-relevance NDCG at 10", aliases=("ndcg10",),
          influence=_mean_infl(_ndcg_rows), weighted=True)
def ndcg10(actual, predicted, sample_weight=None):
    return float(np.average(_ndcg_rows(actual, predicted, 10), weights=sample_weight))


def _recall20_rows(actual, predicted) -> list[float]:
    return [len(set(a) & set(list(p)[:20])) / max(1, len(set(a))) for a, p in zip(actual, predicted)]


@register("recall@20", True, "ranking", "recall at 20 (candidate generation quality)", aliases=("recall20",),
          influence=_mean_infl(_recall20_rows), weighted=True)
def recall20(actual, predicted, sample_weight=None):
    return float(np.average(_recall20_rows(actual, predicted), weights=sample_weight))


# --------------------------------------------------------------------------- segmentation
@register("dice", True, "mask", "Dice coefficient over binary masks (empty-empty = 1)")
def dice(y, p, eps=1e-7):
    y, p = _a(y).astype(bool), _a(p).astype(bool)
    inter = np.logical_and(y, p).sum()
    s = y.sum() + p.sum()
    return 1.0 if s == 0 else (2 * inter + eps) / (s + eps)


@register("iou", True, "mask", "intersection over union over binary masks (empty-empty = 1)", aliases=("jaccard",))
def iou(y, p, eps=1e-7):
    y, p = _a(y).astype(bool), _a(p).astype(bool)
    union = np.logical_or(y, p).sum()
    return 1.0 if union == 0 else (np.logical_and(y, p).sum() + eps) / (union + eps)
