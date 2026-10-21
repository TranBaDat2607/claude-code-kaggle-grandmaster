"""The evaluation set that scores are compared on, and the noise measured on it.

Two scores are only comparable when they were computed on the same rows: the out-of-fold predictions over
``data/folds.csv`` in a K-fold workspace, the labelled holdout in a ``holdout+lb`` one. Ledger records carry a
fingerprint of that set (``folds_hash``, or ``eval_hash`` for holdout predictions), and every
``kgkit ledger compare`` appends what it measured to ``.kaggle-gm/comparisons.jsonl``. ``kgkit status`` and the
session hook report that measured, paired noise; they never derive a threshold from the set's size, because the
noise of a *difference* between two models scored on the same rows is far below the noise of one score::

    key = current_key(root, state)
    recent_noise(root, key)   # {"se": 0.0034, "method": "bootstrap", "n_units": 58, ...} or None

This module must stay standard-library only: hooks import it.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from . import state as S

# lb-only: no score of its own at all (logged with cv=None). lb-blend: blended with public submission files; in a
# K-fold workspace such a blend has no OOF, so a CV logged for it was borrowed from a component.
LB_ONLY_TAGS = ("lb-only", "lb-blend")
DECIDED = ("keep", "discard", "inconclusive")
COMPARISONS = "comparisons.jsonl"


def split_spec(spec: str) -> tuple[str, list[str]]:
    """'file.csv:a,b' -> ('file.csv', ['a', 'b'])."""
    path, _, cols = str(spec).rpartition(":")
    if not path or (len(path) == 1 and path.isalpha()):  # windows drive letter, no column
        raise ValueError(f"expected FILE:COLUMN[,COLUMN...], got {spec!r}")
    names = [c.strip() for c in cols.split(",") if c.strip()]
    if not names:
        raise ValueError(f"no column in {spec!r}")
    return path, names


def _resolve(root: str | Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(root) / p


def spec_hash(root: str | Path, spec: str) -> str | None:
    """Fingerprint of an evaluation set given as FILE:COLS (the file's bytes and the columns); None if unreadable."""
    try:
        path, cols = split_spec(spec)
        h = hashlib.sha1(_resolve(root, path).read_bytes())
    except (OSError, ValueError):
        return None
    h.update(("|" + ",".join(cols)).encode())
    return h.hexdigest()[:12]


def folds_file_hash(root: str | Path) -> str | None:
    """Fingerprint of the frozen fold split (``KG_FOLDS_FILE`` during a recheck, else ``data/folds.csv``)."""
    f = Path(os.environ.get("KG_FOLDS_FILE") or "data/folds.csv")
    f = f if f.is_absolute() else Path(root) / f
    try:
        return hashlib.sha1(f.read_bytes()).hexdigest()[:12] if f.is_file() else None
    except OSError:
        return None


def current_key(root: str | Path, st: S.CompetitionState | None) -> str | None:
    """Fingerprint of the workspace's evaluation set: the holdout labels, or the fold split."""
    if st is not None and st.holdout:
        return spec_hash(root, st.eval_truth) if st.eval_truth else None
    return folds_file_hash(root)


def record_key(rec: dict, holdout: bool) -> str | None:
    return rec.get("eval_hash") if holdout else rec.get("folds_hash")


def own_score(rec: dict, holdout: bool = False) -> bool:
    """True when the record's score is its own on the evaluation set.

    K-fold: a record tagged ``lb-only`` or ``lb-blend`` has none (a blend with public files has no OOF, so its CV
    was borrowed from a component). Fixed holdout (``holdout=True``): the blend itself is scored on the holdout, so
    ``lb-blend`` only says it includes public files; only ``lb-only`` means no score of its own. A score recomputed
    from the record's own stored holdout predictions is always its own."""
    if not isinstance(rec.get("cv"), (int, float)):
        return False
    if rec.get("holdout_path"):
        return True
    tags = set(rec.get("tags") or ())
    return "lb-only" not in tags if holdout else not tags & set(LB_ONLY_TAGS)


def decisions_on(records: list[dict], key: str | None, holdout: bool) -> int:
    """Keep/discard decisions taken on one evaluation set. On a fixed holdout, records decided before their
    predictions were stored count too: the holdout never changes."""
    keys = (key, None) if holdout else (key,)
    return sum(1 for r in records if r.get("decision") in DECIDED and not r.get("recheck")
               and record_key(r, holdout) in keys)


# ---------------------------------------------------------------------------- measured noise
def comparisons_path(root: str | Path) -> Path:
    return Path(root) / S.STATE_DIR / COMPARISONS


def log_comparison(root: str | Path, entry: dict) -> dict:
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
    p = comparisons_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def comparisons(root: str | Path) -> list[dict]:
    p = comparisons_path(root)
    out = []
    try:
        lines = p.read_text(encoding="utf-8").splitlines() if p.is_file() else []
    except OSError:
        return []
    for line in lines:
        try:
            c = json.loads(line)
        except ValueError:
            continue
        if isinstance(c, dict):
            out.append(c)
    return out


def recent_noise(root: str | Path, key: str | None, k: int = 5) -> dict | None:
    """Median paired SE of the last ``k`` comparisons measured on the evaluation set ``key``, or None."""
    rows = [c for c in comparisons(root) if c.get("question") == "paired" and c.get("eval") == key
            and isinstance(c.get("se"), (int, float)) and c["se"] > 0]
    if not rows:
        return None
    last = rows[-k:]
    return {"se": float(statistics.median(c["se"] for c in last)), "n": len(last),
            "methods": sorted({str(c.get("method") or "?") for c in last}), "unit": last[-1].get("unit"),
            "n_units": last[-1].get("n_units"), "last": f"{last[-1].get('new')} vs {last[-1].get('base')}"}


def format_noise(noise: dict) -> str:
    """'paired SE of differences ~0.0034 (bootstrap over 58 rows, median of 3 compares)'."""
    units = f" over {noise['n_units']} {noise.get('unit') or 'rows'}" if noise.get("n_units") else ""
    many = f", median of {noise['n']} compares" if noise.get("n", 1) > 1 else ""
    return f"paired SE of differences ~{noise['se']:.4f} ({'/'.join(noise['methods'])}{units}{many})"
