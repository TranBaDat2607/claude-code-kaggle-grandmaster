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
        cv: float,
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
        **extra,
    ) -> dict:
        """Append an experiment. ``folds_hash`` fingerprints the CV split (from ``folds`` if given,
        else the bytes of ``data/folds.csv``) so blends can detect members trained on different splits."""
        recs = self.records()
        num = 1 + max([int(r["id"].split("_")[0]) for r in recs if r["id"].split("_")[0].isdigit()] or [0])
        exp_id = f"{num:04d}_{_slug(name)}"
        st = S.load(self.root)
        rec = {
            "id": exp_id,
            "name": name,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "metric": metric or (st.metric if st else None),
            "cv": float(cv),
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

    def _folds_hash(self, folds: Sequence[int] | None) -> str | None:
        if folds is not None:
            return hashlib.sha1(np.asarray(folds, dtype=np.int64).tobytes()).hexdigest()[:12]
        f = self.root / "data" / "folds.csv"
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

    # ------------------------------------------------------------------ read
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
        recs = [r for r in self.records() if r.get(key) is not None]
        return sorted(recs, key=lambda r: r[key], reverse=self._gib())[:n]

    def cv_lb_correlation(self) -> dict | None:
        pairs = [(r["cv"], r["lb_public"]) for r in self.records() if r.get("lb_public") is not None]
        if len(pairs) < 3:
            return None
        cv, lb = np.array(pairs).T
        from scipy.stats import pearsonr, spearmanr

        return {"n": len(pairs), "pearson": float(pearsonr(cv, lb)[0]), "spearman": float(spearmanr(cv, lb)[0])}

    def table(self, n: int = 15, sort: str = "time") -> str:
        recs = self.records()
        if sort == "cv":
            recs = self.best(n)
        else:
            recs = recs[-n:]
        if not recs:
            return "(ledger is empty)"
        lines = ["| id | cv | ±std | LB | model | notes |", "|---|---|---|---|---|---|"]
        for r in recs:
            std = f"{r['cv_std']:.4f}" if r.get("cv_std") is not None else ""
            lb = f"{r['lb_public']:.5f}" if r.get("lb_public") is not None else ""
            note = (r.get("notes") or "").replace("|", "/").replace("\n", " ")[:70]
            lines.append(f"| {r['id']} | {r['cv']:.5f} | {std} | {lb} | {r.get('model') or ''} | {note} |")
        return "\n".join(lines)
