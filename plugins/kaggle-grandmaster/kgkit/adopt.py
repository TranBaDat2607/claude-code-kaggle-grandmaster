"""Adopt an existing Kaggle project that has no ``.kaggle-gm/`` workspace yet.

``detect`` looks for signs of Kaggle work (``kernel-metadata.json`` files and their
``competition_sources``, competition URLs in the README or git config) so the SessionStart hook
can point a mid-competition repo at ``/kg-adopt`` instead of the full ``/kg-start`` bootstrap.
``import_submissions`` merges the competition's submission history (``kaggle competitions
submissions <slug> --format json``) into ``.kaggle-gm/submissions.jsonl`` so the daily budget and
the status brief count what was already submitted.

This module must stay standard-library only and fast: the SessionStart hook imports it.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from . import state as S

URL_RE = re.compile(r"kaggle\.com/(?:c|competitions)/([a-z0-9][a-z0-9-]{1,80})", re.IGNORECASE)
EXP_ID_RE = re.compile(r"\b(\d{4}_[a-z0-9-]+)\b")
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env", "site-packages",
             "data", "input", "artifacts", "wandb", "mlruns", "lightning_logs", "checkpoints", "outputs"}
URL_NOT_SLUGS = {"discussion", "leaderboard", "data", "overview", "rules", "code", "submissions", "team"}


def _kernel_metas(root: Path, max_depth: int, max_dirs: int, max_seconds: float) -> list[tuple[Path, dict]]:
    out, seen, start = [], 0, time.monotonic()
    base_depth = len(root.parts)
    for d, dirs, files in os.walk(root):
        seen += 1
        if seen > max_dirs or time.monotonic() - start > max_seconds:
            break
        depth = len(Path(d).parts) - base_depth
        dirs[:] = [x for x in dirs if depth < max_depth and x not in SKIP_DIRS and not x.startswith(".")]
        if "kernel-metadata.json" in files:
            p = Path(d) / "kernel-metadata.json"
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(meta, dict):
                out.append((p, meta))
    return out


def _url_slugs(root: Path) -> list[str]:
    texts = []
    for name in ("README.md", "README.rst", "README.txt", "README", "CLAUDE.md", ".git/config"):
        p = root / name
        try:
            if p.is_file():
                texts.append(p.read_text(encoding="utf-8", errors="replace")[:200_000])
        except OSError:
            pass
    return [m.lower() for t in texts for m in URL_RE.findall(t) if m.lower() not in URL_NOT_SLUGS]


def _truthy(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return None


def detect(start: str | Path = ".", max_depth: int = 3, max_dirs: int = 2000, max_seconds: float = 0.5) -> dict | None:
    """Signs of Kaggle work under ``start``, or None when there are none.

    Returns ``{"slugs": [most likely first], "kernels": [relative folders], "internet_off": bool,
    "evidence": [short strings]}``. ``internet_off`` is True when every kernel that uses a competition
    source has internet disabled (typical of a code competition's inference notebook).
    """
    root = Path(start).resolve()
    metas = _kernel_metas(root, max_depth, max_dirs, max_seconds)
    votes: Counter = Counter()
    internet = []
    for _, meta in metas:
        comps = [str(c).strip().lower() for c in meta.get("competition_sources") or [] if str(c).strip()]
        votes.update(comps)
        if comps:
            internet.append(_truthy(meta.get("enable_internet")))
    url_slugs = _url_slugs(root)
    votes.update(url_slugs)
    if not metas and not votes:
        return None
    evidence = []
    if metas:
        evidence.append(f"{len(metas)} kernel-metadata.json")
    if url_slugs:
        evidence.append("competition URL in README/git config")
    kernels = sorted({p.parent.relative_to(root).as_posix() or "." for p, _ in metas})
    return {"slugs": [s for s, _ in votes.most_common()], "kernels": kernels,
            "internet_off": bool(internet) and all(v is False for v in internet), "evidence": evidence}


# ---------------------------------------------------------------------------- submission history
def _get(row: dict, *keys):
    low = {re.sub(r"[^a-z]", "", str(k).lower()): v for k, v in row.items()}
    for k in keys:
        v = low.get(re.sub(r"[^a-z]", "", k.lower()))
        if v not in (None, "", "None"):
            return v
    return None


def _time(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().replace("Z", "+00:00")
    try:
        t = dt.datetime.fromisoformat(s.replace(" ", "T", 1))
    except ValueError:
        return None
    if t.tzinfo is not None:
        t = t.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _score(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _status(v) -> str:
    s = str(v or "").lower()
    if "error" in s or "fail" in s:
        return "failed"
    if "complete" in s:
        return "scored"
    return "submitted"  # pending / running: still uses a slot of the daily budget


def parse_submissions(text: str, competition: str) -> list[dict]:
    """Records in the ``submissions.jsonl`` format from ``kaggle competitions submissions`` JSON or CSV output.

    Lines before the JSON (CLI warnings such as an outdated-version notice) are skipped.
    """
    lines = text.strip().splitlines()
    first = next((i for i, l in enumerate(lines) if l.lstrip()[:1] in ("[", "{")), None)
    rows: list = []
    if first is not None:
        data = json.loads("\n".join(lines[first:]))
        rows = data if isinstance(data, list) else data.get("submissions") or []
    elif lines:
        rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        when = _time(_get(row, "date", "submittedDate", "dateSubmitted"))
        if when is None:
            continue
        desc = _get(row, "description", "message")
        file = _get(row, "fileName", "file")
        m = EXP_ID_RE.search(str(desc or "")) or EXP_ID_RE.search(str(file or ""))
        out.append({"time_utc": when, "competition": competition, "file": file, "kernel": None, "message": desc,
                    "exp_id": m.group(1) if m else None, "status": _status(_get(row, "status")),
                    "public_score": _score(_get(row, "publicScore")), "private_score": _score(_get(row, "privateScore")),
                    "via": "kaggle-import"})
    return sorted(out, key=lambda r: r["time_utc"])


def _seconds(t: str) -> float:
    return dt.datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").timestamp()


def import_submissions(root: str | Path, records: list[dict], window_s: float = 900) -> tuple[int, int]:
    """Append records not already in ``.kaggle-gm/submissions.jsonl``; returns (added, skipped).

    A record is a duplicate when one with the same message (or file) is logged within ``window_s``
    seconds: the hook logs a submission when the CLI returns, Kaggle dates it a little earlier.
    """
    log = S.submissions_log(root)
    existing = S.submission_records(root)

    def key(r):
        return r.get("message") or r.get("file") or ""

    known = [(key(r), _seconds(r["time_utc"])) for r in existing if _time(r.get("time_utc"))]
    added = []
    for r in records:
        t = _seconds(r["time_utc"])
        if any(k == key(r) and abs(t - kt) <= window_s for k, kt in known):
            continue
        added.append(r)
        known.append((key(r), t))
    if added:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            for r in added:
                f.write(json.dumps(r) + "\n")
    return len(added), len(records) - len(added)
