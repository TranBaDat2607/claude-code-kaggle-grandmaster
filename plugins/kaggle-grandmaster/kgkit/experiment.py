"""The experiment ledger: one JSON line per experiment in ``.kaggle-gm/ledger.jsonl``.

Grandmasters win by running many small, well-recorded experiments. Every training run
should end with ``Ledger().log(...)`` so that CV, fold scores, params, the git commit,
and the OOF / test predictions needed for ensembling are never lost::

    from kgkit.experiment import Ledger, seed_everything
    seed_everything(42)
    ...
    exp = Ledger().log("lgbm_te_v3", cv=0.8123, fold_scores=scores, params=params,
                       features=feature_cols, oof=oof, test_pred=test_pred,
                       notes="added OOF target encoding of city x month")
    # after submitting:  python -m kgkit ledger lb <exp_id> 0.8150

The ledger also remembers *decisions*: each experiment can name its ``parent`` (the baseline it
was compared against) and carry a ``decision`` (baseline / keep / discard / inconclusive). The
*accepted baseline* is the latest kept experiment, not simply the highest CV, so a lucky seed
or a leaky run never silently becomes the thing every new idea is measured against::

    python -m kgkit ledger compare 0012 0009 --truth data/train.csv:target
    python -m kgkit ledger decide 0012 keep "focal loss, +0.0021, 5/5 folds"

Only the ledger (not the artefacts) should be committed to git.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import state as S


def seed_everything(seed: int = 42, deterministic_torch: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _git_info(cwd: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True,
                                timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True,
                                    timeout=5).stdout.strip())
        return {"git_commit": commit or None, "git_dirty": dirty if commit else None}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}


DECISIONS = ("baseline", "keep", "discard", "inconclusive")
ACCEPTED = ("baseline", "keep")
# Records without a cross-validated score of their own: blends with public submission files, forks of
# public inference notebooks. They carry an LB score only and never enter the CV-LB correlation.
LB_ONLY_TAGS = ("lb-only", "lb-blend")
MIN_DISTINCT_CV = 4


def has_own_cv(r: dict) -> bool:
    return isinstance(r.get("cv"), (int, float)) and not set(r.get("tags") or ()) & set(LB_ONLY_TAGS)


def fmt_cv(r: dict, digits: int = 5) -> str:
    return f"{r['cv']:.{digits}f}" if isinstance(r.get("cv"), (int, float)) else "LB-only"


def _check_decision(d: str) -> str:
    d = d.lower()
    if d not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, got {d!r}")
    return d


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()[:60] or "exp"


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


class Ledger:
    def __init__(self, root: str | Path | None = None):
        if root is None:
            root = S.find_root() or Path.cwd()
        self.root = Path(root)
        self.dir = self.root / S.STATE_DIR
        self.path = self.dir / "ledger.jsonl"
        self.artifacts = self.root / "artifacts"

    # ------------------------------------------------------------------ io
    def records(self) -> list[dict]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def _write_all(self, recs: list[dict]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, exp_id: str) -> dict:
        for r in self.records():
            num = r["id"].split("_")[0]
            if r["id"] == exp_id or r.get("name") == exp_id or (exp_id.isdigit() and num.isdigit() and int(num) == int(exp_id)):
                return r
        raise KeyError(f"experiment {exp_id!r} not found in {self.path}")

    # ------------------------------------------------------------------ write
    def log(
        self,
        name: str,
        cv: float | None,
        fold_scores: Sequence[float] | None = None,
        params: dict | None = None,
        features: Sequence[str] | None = None,
        model: str | None = None,
        notes: str = "",
        oof: np.ndarray | None = None,
        test_pred: np.ndarray | None = None,
        tags: Sequence[str] = (),
        metric: str | None = None,
        folds: Sequence[int] | None = None,
        parent: str | None = None,
        decision: str | None = None,
        **extra,
    ) -> dict:
        """Append an experiment. ``folds_hash`` fingerprints the CV split (from ``folds`` if given,
        else the bytes of ``data/folds.csv``) so blends can detect members trained on different splits.
        ``cv=None`` logs an LB-only record (tagged ``lb-only``): something that cannot be cross-validated,
        such as a blend with public submission files."""
        recs = self.records()
        if cv is None and not set(tags) & set(LB_ONLY_TAGS):
            tags = [*tags, "lb-only"]
        num = 1 + max([int(r["id"].split("_")[0]) for r in recs if r["id"].split("_")[0].isdigit()] or [0])
        exp_id = f"{num:04d}_{_slug(name)}"
        st = S.load(self.root)
        rec = {
            "id": exp_id,
            "name": name,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "metric": metric or (st.metric if st else None),
            "cv": float(cv) if cv is not None else None,
            "cv_std": float(np.std(fold_scores)) if fold_scores is not None and len(fold_scores) else None,
            "fold_scores": [float(s) for s in fold_scores] if fold_scores is not None else None,
            "model": model,
            "params": _jsonable(params or {}),
            "n_features": len(features) if features is not None else None,
            "features": list(features) if features is not None else None,
            "tags": list(tags),
            "notes": notes,
            "lb_public": None,
            "lb_private": None,
            "submission": None,
            "folds_hash": self._folds_hash(folds),
            "recheck": os.environ.get("KG_RECHECK") or None,  # set by `kgkit recheck`: not on the frozen folds
            "parent": self._resolve_id(parent) if parent else None,
            "decision": _check_decision(decision) if decision else None,
            **_git_info(self.root),
            **_jsonable(extra),
        }
        if oof is not None or test_pred is not None:
            d = self.artifacts / exp_id
            d.mkdir(parents=True, exist_ok=True)
            if oof is not None:
                np.save(d / "oof.npy", np.asarray(oof))
                rec["oof_path"] = str((d / "oof.npy").relative_to(self.root)).replace("\\", "/")
            if test_pred is not None:
                np.save(d / "test.npy", np.asarray(test_pred))
                rec["test_path"] = str((d / "test.npy").relative_to(self.root)).replace("\\", "/")
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def _resolve_id(self, ref: str) -> str:
        """Canonical id for a ref; unknown refs are kept verbatim (e.g. a local id named inside a remote
        Kaggle run, whose own ledger starts empty)."""
        try:
            return self.get(ref)["id"]
        except KeyError:
            return str(ref)

    def _folds_hash(self, folds: Sequence[int] | None) -> str | None:
        if folds is not None:
            return hashlib.sha1(np.asarray(folds, dtype=np.int64).tobytes()).hexdigest()[:12]
        f = Path(os.environ.get("KG_FOLDS_FILE") or "data/folds.csv")  # recheck runs use a re-drawn split
        f = f if f.is_absolute() else self.root / f
        if f.is_file():
            return hashlib.sha1(f.read_bytes()).hexdigest()[:12]
        return None

    def update(self, exp_id: str, **fields) -> dict:
        recs = self.records()
        target = self.get(exp_id)["id"]
        for r in recs:
            if r["id"] == target:
                r.update(_jsonable(fields))
                self._write_all(recs)
                return r
        raise KeyError(exp_id)

    def attach_lb(self, exp_id: str, public: float | None = None, private: float | None = None,
                  submission: str | None = None) -> dict:
        fields: dict[str, Any] = {}
        if public is not None:
            fields["lb_public"] = float(public)
        if private is not None:
            fields["lb_private"] = float(private)
        if submission is not None:
            fields["submission"] = submission
        return self.update(exp_id, **fields)

    def decide(self, exp_id: str, decision: str, note: str = "", parent: str | None = None) -> dict:
        """Record the keep/discard decision (and optionally the baseline it was compared against)."""
        fields: dict[str, Any] = {"decision": _check_decision(decision), "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if note:
            fields["decision_note"] = note
        if parent:
            fields["parent"] = self.get(parent)["id"]  # locally the baseline must exist
        return self.update(exp_id, **fields)

    def import_preds(self, name: str, oof: np.ndarray, test_pred: np.ndarray | None, y_true=None,
                     folds: Sequence[int] | None = None, metric: str | None = None, cv: float | None = None,
                     notes: str = "", source: str = "external") -> dict:
        """Log predictions produced elsewhere (a teammate, a reproduced public notebook) so they can be
        compared and blended like any other experiment. With ``y_true`` and ``folds`` the CV and fold
        scores are recomputed here, on *our* folds, instead of trusting a reported number."""
        st = S.load(self.root)
        metric = metric or (st.metric if st else None)
        fold_scores = None
        oof = np.asarray(oof)
        if y_true is not None and metric:
            from . import metrics as M

            m = M.get(metric)
            y = np.asarray(y_true)
            if len(y) != len(oof):
                raise ValueError(f"OOF has {len(oof)} rows but the target has {len(y)}")
            if folds is not None:
                f = np.asarray(folds)
                fold_scores = [m(y[f == k], oof[f == k]) for k in sorted(set(f[f >= 0].tolist()))]
                cv = m(y[f >= 0], oof[f >= 0])
            else:
                cv = m(y, oof)
        if cv is None:
            raise ValueError("pass cv, or y_true (+ folds) so it can be computed")
        return self.log(name, cv, fold_scores=fold_scores, oof=oof, test_pred=test_pred, notes=notes,
                        tags=[source], metric=metric, model=source)

    # ------------------------------------------------------------------ read
    def baseline(self, fallback: bool = True) -> dict | None:
        """The accepted baseline: the most recent experiment decided ``keep`` or ``baseline``.
        Falls back to the highest CV only when nothing has been accepted yet."""
        recs = self.records()
        for r in reversed(recs):
            if r.get("decision") in ACCEPTED and not r.get("recheck"):
                return r
        if fallback:
            best = self.best(1)
            return best[0] if best else None
        return None

    def lineage(self, exp_id: str) -> list[dict]:
        """Chain of parents back to the root, oldest first — the list of changes a reverse ablation
        should re-test (each kept step that was only marginally significant is a suspect)."""
        chain, seen = [], set()
        rec: dict | None = self.get(exp_id)
        while rec is not None and rec["id"] not in seen:
            chain.append(rec)
            seen.add(rec["id"])
            try:
                rec = self.get(rec["parent"]) if rec.get("parent") else None
            except KeyError:
                rec = None
        return chain[::-1]

    def decisions_on_folds(self, folds_hash: str | None) -> int:
        """How many keep/discard decisions were taken on one fold split — every decision spends a
        little of the split's ability to give unbiased estimates."""
        return sum(1 for r in self.records() if r.get("decision") in ("keep", "discard", "inconclusive")
                   and r.get("folds_hash") == folds_hash)

    def load_oof(self, exp_id: str) -> np.ndarray:
        return np.load(self.root / self.get(exp_id)["oof_path"])

    def load_test(self, exp_id: str) -> np.ndarray:
        return np.load(self.root / self.get(exp_id)["test_path"])

    def _gib(self) -> bool:
        st = S.load(self.root)
        if st is not None:
            return st.greater_is_better
        return True

    def best(self, n: int = 5, key: str = "cv") -> list[dict]:
        recs = [r for r in self.records() if r.get(key) is not None and not r.get("recheck")
                and (key != "cv" or has_own_cv(r))]
        return sorted(recs, key=lambda r: r[key], reverse=self._gib())[:n]

    def cv_lb_pairs(self) -> list[tuple[float, float]]:
        """(CV, public LB) of submissions that have a cross-validated score of their own."""
        return [(r["cv"], r["lb_public"]) for r in self.records()
                if r.get("lb_public") is not None and has_own_cv(r) and not r.get("recheck")]

    def cv_lb_correlation(self) -> dict | None:
        """None unless >= MIN_DISTINCT_CV submissions with distinct own CVs: with fewer, a correlation of
        1.00 is an artefact (two LB-only blends logged with their base stack's CV gave exactly that)."""
        pairs = self.cv_lb_pairs()
        if len(pairs) < 3 or len({round(c, 12) for c, _ in pairs}) < MIN_DISTINCT_CV:
            return None
        cv, lb = np.array(pairs).T
        if np.ptp(lb) == 0:
            return None
        from scipy.stats import pearsonr, spearmanr

        return {"n": len(pairs), "pearson": float(pearsonr(cv, lb)[0]), "spearman": float(spearmanr(cv, lb)[0])}

    def holdout_note(self) -> str:
        """For a holdout+lb workspace: how noisy the 'CV' column is."""
        st = S.load(self.root)
        if st is None or not st.holdout:
            return ""
        se = st.holdout_se
        n = st.holdout_n
        if se is None and n:
            cvs = [r["cv"] for r in self.records() if has_own_cv(r) and 0 < r["cv"] < 1]
            if cvs:  # binomial approximation; a bootstrap of the holdout metric is better (holdout_se)
                p = float(np.median(cvs))
                se = (p * (1 - p) / n) ** 0.5
        size = f"n={n}" if n else "size unknown"
        noise = f", SE ~{se:.3f}: differences under ~{2 * se:.3f} are noise" if se else ""
        return f"CV here is a fixed holdout ({size}{noise}); the public LB is the second opinion"

    def cv_lb_summary(self) -> str:
        """One line on CV-LB agreement, or why it is not reported."""
        corr = self.cv_lb_correlation()
        pairs = self.cv_lb_pairs()
        lb_only = sum(1 for r in self.records() if r.get("lb_public") is not None and not has_own_cv(r))
        skipped = f"; {lb_only} LB-only submission(s) left out" if lb_only else ""
        if corr:
            line = (f"CV-LB correlation over {corr['n']} subs: pearson {corr['pearson']:.2f}, "
                    f"spearman {corr['spearman']:.2f}{skipped}")
        elif pairs:
            line = (f"CV-LB correlation: not reported ({len(pairs)} subs with their own CV, "
                    f"{len({round(c, 12) for c, _ in pairs})} distinct CVs; needs >= {MIN_DISTINCT_CV}){skipped}")
        else:
            return ""
        note = self.holdout_note()
        return line + (f"\n  {note}" if note else "")

    def table(self, n: int = 15, sort: str = "time") -> str:
        recs = self.records()
        if sort == "cv":
            recs = self.best(n)
        else:
            recs = recs[-n:]
        if not recs:
            return "(ledger is empty)"
        lines = ["| id | cv | ±std | LB | decision | model | notes |", "|---|---|---|---|---|---|---|"]
        for r in recs:
            std = f"{r['cv_std']:.4f}" if r.get("cv_std") is not None else ""
            lb = f"{r['lb_public']:.5f}" if r.get("lb_public") is not None else ""
            note = (r.get("notes") or "").replace("|", "/").replace("\n", " ")[:70]
            lines.append(f"| {r['id']} | {fmt_cv(r)} | {std} | {lb} | {r.get('decision') or ''} | "
                         f"{r.get('model') or ''} | {note} |")
        return "\n".join(lines)
