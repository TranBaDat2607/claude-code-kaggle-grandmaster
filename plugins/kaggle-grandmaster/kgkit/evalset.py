"""The evaluation set that scores are compared on, and the noise measured on it.

Two scores are only comparable when they were computed on the same rows: the out-of-fold predictions over
``data/folds.csv`` in a K-fold workspace, the labelled holdout in a ``holdout+lb`` one. Ledger records carry a
fingerprint of that set (``folds_hash``, or ``eval_hash`` for holdout predictions), and every
``kgkit ledger compare`` appends what it measured to ``.kaggle-gm/comparisons.jsonl``. ``kgkit status`` and the
session hook report that measured, paired noise; they never derive a threshold from the set's size, because the
noise of a *difference* between two models scored on the same rows is far below the noise of one score::

    key = current_key(root, state)
    recent_noise(root, key)   # {"se": 0.0034, "method": "bootstrap", "n_units": 58, ...} or None

A fingerprint is of the content, not of the file: the fold column of the split file (the digest
``Ledger.log(folds=...)`` takes of an array) and the label columns of a CSV holdout. The same split written on
Windows (CRLF) or Linux (LF), quoted, or with other columns gets one fingerprint. kgkit 1.7.1 and older hashed the
file's bytes; ``upgrades`` maps those fingerprints of the current files to the content fingerprint.

This module must stay standard-library only: hooks import it.
"""

from __future__ import annotations

import array
import csv
import hashlib
import io
import json
import os
import statistics
import sys
import time
from itertools import islice
from operator import itemgetter
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


# ---------------------------------------------------------------------------- fingerprints
def split_digest(pieces) -> str:
    """Fingerprint of a fold assignment: one little-endian int64 per row, in pieces (bytes-like) in row order. An
    array passed to ``Ledger.log(folds=...)`` and the fold column of the split file get it alike."""
    h = hashlib.sha1()
    for piece in pieces:
        h.update(piece)
    return h.hexdigest()[:12]


def _rows(data: bytes):
    """A CSV reader over a file's bytes, decoded as it reads: any line endings, a BOM, no copy of the file."""
    return csv.reader(io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig", errors="replace", newline=""))


def _int(v: str) -> int:
    try:
        return int(v)
    except ValueError:
        x = float(v)  # "3.0": pandas writes a column with gaps as floats
        if not x.is_integer():
            raise
        return int(x)


def _fold_batches(data: bytes, size: int = 1 << 16):
    """The ``fold`` column of a split file (its bytes) in row order, as int64 arrays of up to ``size`` rows, so a
    split of any size is read in little memory; blank lines are skipped. ValueError when there is no such column or
    a fold is not an integer."""
    rows = _rows(data)
    ints: dict[str, int] = {}
    try:
        header = [h.strip() for h in next(rows, [])]
        if "fold" not in header:
            raise ValueError("no 'fold' column")
        pick = itemgetter(header.index("fold"))
        while True:
            before = rows.line_num
            vals = list(map(pick, filter(None, islice(rows, size))))
            if rows.line_num == before:
                return
            for v in set(vals) - ints.keys():
                try:
                    ints[v] = _int(v)
                except ValueError:
                    raise ValueError(f"fold {v.strip()!r} is not an integer") from None
            yield array.array("q", map(ints.__getitem__, vals))
    except IndexError:
        raise ValueError("a row has no fold") from None
    except csv.Error as e:
        raise ValueError(f"not a readable CSV ({e})") from None
    except OverflowError:
        raise ValueError("a fold does not fit in int64") from None


def fold_ids(data: bytes) -> list[int]:
    """The folds of a split file (its bytes), sorted; ValueError as ``_fold_batches``."""
    ids: set[int] = set()
    for a in _fold_batches(data):
        ids.update(a)
    return sorted(ids)


def _little_endian(a: array.array) -> array.array:
    if sys.byteorder == "big":
        a.byteswap()
    return a


def _finish(h, cols: list[str] | None) -> str:
    if cols is not None:  # an evaluation set FILE:COLS
        h.update(("|" + ",".join(cols)).encode())
    return h.hexdigest()[:12]


def _bytes_digest(data: bytes, cols: list[str] | None = None) -> str:
    """The fingerprint kgkit <= 1.7.1 took: sha1 of the bytes (then of '|' and the columns of an evaluation set)."""
    return _finish(hashlib.sha1(data), cols)


def _eol_digests(data: bytes, cols: list[str] | None = None) -> tuple[str, ...]:
    """``_bytes_digest`` of the bytes as they are, with LF and with CRLF line endings; piece by piece, so a big file
    is not copied whole."""
    raw, lf, crlf = hashlib.sha1(data), hashlib.sha1(), hashlib.sha1()
    pos = 0
    while pos < len(data):
        end = pos + (1 << 24)
        if data[end - 1:end] == b"\r":
            end += 1  # a CRLF pair stays in one piece
        piece = data[pos:end].replace(b"\r\n", b"\n")
        lf.update(piece)
        crlf.update(piece.replace(b"\n", b"\r\n"))
        pos = end
    return tuple(_finish(h, cols) for h in (raw, lf, crlf))


def _fold_digest(data: bytes) -> str:
    """The split file's fold column (``split_digest``); for a file without a readable integer fold column, its bytes
    with LF line endings."""
    try:
        return split_digest(_little_endian(a) for a in _fold_batches(data))
    except ValueError:
        return _eol_digests(data)[1]


def _cell(v: str) -> str:
    """A label as the metric reads it: '1', '1.0' and ' 1 ' are one value; blank and NaN are both missing."""
    v = v.strip()
    try:
        x = float(v)
    except ValueError:
        return v
    return repr(x) if x == x else ""


def _labels_digest(data: bytes, cols: list[str]) -> str:
    """The label columns ``cols`` of a CSV, in row order; for a file without them, its bytes with LF line endings."""
    rows = _rows(data)
    h = hashlib.sha1()
    try:
        header = [c.strip() for c in next(rows, [])]
        idx = [header.index(c) for c in cols]
        for r in rows:
            if r:
                h.update(("\x1f".join(_cell(r[i]) for i in idx) + "\x1e").encode())
    except (ValueError, IndexError, csv.Error):
        return _eol_digests(data, cols)[1]
    return h.hexdigest()[:12]


_DIGESTS: dict[tuple, str] = {}  # (sha1 of the bytes, columns) -> content fingerprint: a file is parsed once


def _content_digest(data: bytes, cols: list[str] | None = None) -> str:
    """Fingerprint of a file's content: the split file's fold column (``cols`` None) or the columns ``cols``."""
    key = (hashlib.sha1(data).digest(), None if cols is None else tuple(cols))
    if key not in _DIGESTS:
        _DIGESTS[key] = _fold_digest(data) if cols is None else _labels_digest(data, cols)
    return _DIGESTS[key]


def _by_content(f: Path, cols: list[str] | None) -> bool:
    """The split file (``cols`` None) and a CSV evaluation set are fingerprinted by their content. A parquet,
    feather or compressed file keeps the fingerprint of its bytes: it has no line endings to differ in."""
    return cols is None or f.suffix.lower() == ".csv"


def _spec_file(root: str | Path, spec: str) -> tuple[Path, list[str]]:
    path, cols = split_spec(spec)
    return _resolve(root, path), cols


def spec_hash(root: str | Path, spec: str) -> str | None:
    """Fingerprint of an evaluation set given as FILE:COLS: the values in those columns, row by row (of a file that
    is not a CSV, its bytes); None if unreadable."""
    try:
        f, cols = _spec_file(root, spec)
        data = f.read_bytes()
    except (OSError, ValueError):
        return None
    return _content_digest(data, cols) if _by_content(f, cols) else _bytes_digest(data, cols)


def folds_path(root: str | Path) -> Path:
    """The frozen fold split: ``KG_FOLDS_FILE`` during a recheck, else ``data/folds.csv``."""
    f = Path(os.environ.get("KG_FOLDS_FILE") or "data/folds.csv")
    return f if f.is_absolute() else Path(root) / f


def folds_file_hash(root: str | Path) -> str | None:
    """Fingerprint of the frozen fold split (``folds_path``): its fold column, so neither line endings, quoting nor
    the other columns matter. Row order does: without an id column it is how rows get their fold."""
    try:
        return _content_digest(folds_path(root).read_bytes())
    except OSError:
        return None


def upgrades(root: str | Path, keys, st: S.CompetitionState | None = None) -> dict[str, str]:
    """{old: new} for the fingerprints in ``keys`` that kgkit <= 1.7.1 took of the current split file or, in a
    holdout+lb workspace, the holdout labels. It hashed the file's bytes, so the same split written with other line
    endings looked like another split; the bytes as written with LF and with CRLF are both recognised. ``new`` is the
    content fingerprint a record gets now."""
    keys = {k for k in keys if isinstance(k, str)}
    if not keys:
        return {}
    st = st if st is not None else S.load(root)
    sources = [(folds_path(root), None)]
    if st is not None and st.holdout and st.eval_truth:
        try:
            sources.append(_spec_file(root, st.eval_truth))
        except ValueError:
            pass
    out = {}
    for f, cols in sources:
        if not _by_content(f, cols):
            continue  # still fingerprinted by its bytes: nothing changed
        try:
            data = f.read_bytes()
        except OSError:
            continue
        hit = keys & set(_eol_digests(data, cols))
        if hit:
            new = _content_digest(data, cols)
            out.update({k: new for k in hit if k != new})
    return out


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
    """The logged comparisons; an ``eval`` fingerprint kgkit <= 1.7.1 took of the current files is upgraded."""
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
    up = upgrades(root, (c.get("eval") for c in out))
    for c in out:
        if isinstance(c.get("eval"), str) and c["eval"] in up:
            c["eval"] = up[c["eval"]]
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
