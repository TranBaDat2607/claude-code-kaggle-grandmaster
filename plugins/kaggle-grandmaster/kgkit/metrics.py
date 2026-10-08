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

    def __call__(self, y_true, y_pred, **kw) -> float:
        return float(self.fn(y_true, y_pred, **kw))

    def is_better(self, a: float, b: float) -> bool:
        """True if score ``a`` beats score ``b``."""
        return a > b if self.greater_is_better else a < b

    def worst(self) -> float:
        return -np.inf if self.greater_is_better else np.inf


_REGISTRY: dict[str, Metric] = {}
_ALIASES: dict[str, str] = {}


def register(name, greater_is_better, kind, description="", aliases=()):
    def deco(fn):
        _REGISTRY[name] = Metric(name, fn, greater_is_better, kind, description)
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


# ----------------------------------------------------------------------------- regression
@register("rmse", False, "value", "root mean squared error")
def rmse(y, p):
    return np.sqrt(np.mean((_a(y) - _a(p)) ** 2))


@register("mse", False, "value", "mean squared error")
def mse(y, p):
    return np.mean((_a(y) - _a(p)) ** 2)


@register("mae", False, "value", "mean absolute error (optimal constant: median)")
def mae(y, p):
    return np.mean(np.abs(_a(y) - _a(p)))


@register("medae", False, "value", "median absolute error")
def medae(y, p):
    return np.median(np.abs(_a(y) - _a(p)))


@register("rmsle", False, "value", "RMSE on log1p — train on log1p(y) with RMSE", aliases=("rmsle_",))
def rmsle(y, p):
    p = np.clip(_a(p).astype(float), 0, None)
    return np.sqrt(np.mean((np.log1p(_a(y)) - np.log1p(p)) ** 2))


@register("msle", False, "value", "mean squared log error")
def msle(y, p):
    p = np.clip(_a(p).astype(float), 0, None)
    return np.mean((np.log1p(_a(y)) - np.log1p(p)) ** 2)


@register("mape", False, "value", "mean absolute percentage error (zeros in y excluded)")
def mape(y, p):
    y, p = _a(y).astype(float), _a(p).astype(float)
    m = y != 0
    return np.mean(np.abs((y[m] - p[m]) / y[m])) * 100


@register("smape", False, "value", "symmetric MAPE in percent (0/0 counted as 0)")
def smape(y, p):
    y, p = _a(y).astype(float), _a(p).astype(float)
    denom = np.abs(y) + np.abs(p)
    out = np.where(denom == 0, 0.0, 2 * np.abs(p - y) / np.where(denom == 0, 1, denom))
    return np.mean(out) * 100


@register("r2", True, "value", "coefficient of determination")
def r2(y, p):
    return skm.r2_score(y, p)


@register("pearson", True, "value", "Pearson correlation")
def pearson(y, p):
    return stats.pearsonr(_a(y).ravel(), _a(p).ravel())[0]


@register("spearman", True, "value", "Spearman rank correlation")
def spearman(y, p):
    return stats.spearmanr(_a(y).ravel(), _a(p).ravel())[0]


# ------------------------------------------------------------------ probabilistic / scores
@register("auc", True, "proba", "ROC AUC (binary, or multiclass one-vs-rest macro)", aliases=("roc_auc", "rocauc"))
def auc(y, p):
    p = _a(p)
    if p.ndim == 2 and p.shape[1] > 1:
        return skm.roc_auc_score(y, p, multi_class="ovr", average="macro")
    return skm.roc_auc_score(y, p.ravel())


@register("gini", True, "proba", "normalized Gini = 2*AUC - 1", aliases=("normalized_gini",))
def gini(y, p):
    return 2 * skm.roc_auc_score(y, _a(p).ravel()) - 1


@register("pr_auc", True, "proba", "average precision", aliases=("average_precision", "ap"))
def pr_auc(y, p):
    return skm.average_precision_score(y, p)


@register("mean_col_auc", True, "proba", "mean column-wise ROC AUC (multilabel)", aliases=("mcauc",))
def mean_col_auc(y, p):
    y, p = _a(y), _a(p)
    cols = [skm.roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if len(np.unique(y[:, j])) > 1]
    return float(np.mean(cols))


@register("logloss", False, "proba", "binary / multiclass log loss (clipped 1e-15)", aliases=("log_loss", "cross_entropy", "bce"))
def logloss(y, p, eps=1e-15):
    p = np.clip(_a(p).astype(float), eps, 1 - eps)
    y = _a(y)
    if p.ndim == 2 and p.shape[1] > 1:
        p = p / p.sum(axis=1, keepdims=True)
        return -np.mean(np.log(p[np.arange(len(y)), y.astype(int)]))
    p = p.ravel()
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


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


@register("brier", False, "proba", "Brier score (binary)")
def brier(y, p):
    return skm.brier_score_loss(y, _a(p).ravel())


@register("multilabel_logloss", False, "proba", "mean binary log loss over label columns")
def multilabel_logloss(y, p, eps=1e-15):
    y = _a(y).astype(float)
    p = np.clip(_a(p).astype(float), eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# -------------------------------------------------------------------------------- labels
@register("accuracy", True, "label", "accuracy", aliases=("acc",))
def accuracy(y, p):
    return skm.accuracy_score(y, p)


@register("balanced_accuracy", True, "label", "balanced accuracy")
def balanced_accuracy(y, p):
    return skm.balanced_accuracy_score(y, p)


@register("f1", True, "label", "binary F1 (tune the threshold!)")
def f1(y, p):
    return skm.f1_score(y, p)


@register("f1_macro", True, "label", "macro F1")
def f1_macro(y, p):
    return skm.f1_score(y, p, average="macro")


@register("f1_micro", True, "label", "micro F1")
def f1_micro(y, p):
    return skm.f1_score(y, p, average="micro")


@register("f1_weighted", True, "label", "weighted F1")
def f1_weighted(y, p):
    return skm.f1_score(y, p, average="weighted")


@register("fbeta2", True, "label", "binary F-beta with beta=2 (recall-heavy)", aliases=("f2",))
def fbeta2(y, p):
    return skm.fbeta_score(y, p, beta=2)


@register("mcc", True, "label", "Matthews correlation coefficient")
def mcc(y, p):
    return skm.matthews_corrcoef(y, p)


@register("qwk", True, "label", "quadratic weighted kappa (optimise rounding thresholds!)",
          aliases=("quadratic_weighted_kappa", "kappa_quadratic"))
def qwk(y, p):
    return skm.cohen_kappa_score(_a(y).astype(int), _a(p).astype(int), weights="quadratic")


@register("cohen_kappa", True, "label", "unweighted Cohen's kappa")
def cohen_kappa(y, p):
    return skm.cohen_kappa_score(y, p)


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


def _mapk(actual, predicted, k):
    return float(np.mean([apk(a, p, k) for a, p in zip(actual, predicted)]))


@register("map@5", True, "ranking", "mean average precision at 5", aliases=("map5",))
def map5(actual, predicted):
    return _mapk(actual, predicted, 5)


@register("map@10", True, "ranking", "mean average precision at 10", aliases=("map10",))
def map10(actual, predicted):
    return _mapk(actual, predicted, 10)


@register("map@12", True, "ranking", "mean average precision at 12 (H&M-style)", aliases=("map12",))
def map12(actual, predicted):
    return _mapk(actual, predicted, 12)


def mapk(actual, predicted, k=10) -> float:
    return _mapk(actual, predicted, k)


def ndcgk(actual, predicted, k=10) -> float:
    scores = []
    for a, p in zip(actual, predicted):
        a = set(a)
        dcg = sum(1.0 / np.log2(i + 2) for i, x in enumerate(list(p)[:k]) if x in a)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(a), k)))
        scores.append(dcg / idcg if idcg else 0.0)
    return float(np.mean(scores))


@register("ndcg@10", True, "ranking", "binary-relevance NDCG at 10", aliases=("ndcg10",))
def ndcg10(actual, predicted):
    return ndcgk(actual, predicted, 10)


@register("recall@20", True, "ranking", "recall at 20 (candidate generation quality)", aliases=("recall20",))
def recall20(actual, predicted):
    return float(np.mean([len(set(a) & set(list(p)[:20])) / max(1, len(set(a))) for a, p in zip(actual, predicted)]))


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
