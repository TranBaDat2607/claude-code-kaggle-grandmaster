"""Shared helpers for kaggle-grandmaster hooks. Standard library only; every entry point fails open."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import shlex
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from kgkit import state as S  # noqa: E402  (stdlib-only module)

SUBMIT_RE = re.compile(r"kaggle(?:\.exe)?\s+(?:competitions|c)\s+submit\b(.*)", re.IGNORECASE | re.DOTALL)


def read_input() -> dict:
    try:
        return json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, ValueError):
        return {}


def emit(event: str, **fields) -> None:
    sys.stdout.write(json.dumps({"hookSpecificOutput": {"hookEventName": event, **fields}}))
    sys.stdout.flush()


def command_of(payload: dict) -> str:
    ti = payload.get("tool_input") or {}
    return ti.get("command") or ""


def workspace(payload: dict) -> Path | None:
    cwd = payload.get("cwd") or os.getcwd()
    try:
        return S.find_root(cwd)
    except OSError:
        return None


def parse_submit(command: str) -> dict | None:
    """Extract competition / file / message / kernel from a `kaggle competitions submit` command."""
    m = SUBMIT_RE.search(command)
    if not m:
        return None
    rest = m.group(1).replace("\\", "/")  # Windows paths: POSIX lexing would treat \ as an escape
    try:
        lex = shlex.shlex(rest, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        toks = list(lex)
    except ValueError:
        toks = rest.split()
    # stop at the next shell separator (quote-aware: separators inside quotes stay in their token)
    for i, t in enumerate(toks):
        if t in {"&&", "||", ";", "|", "&", ";;"}:
            toks = toks[:i]
            break
    out = {"competition": None, "file": None, "message": None, "kernel": None, "version": None}
    flags = {"-f": "file", "--file": "file", "-m": "message", "--message": "message",
             "-k": "kernel", "--kernel": "kernel", "-v": "version", "--version": "version"}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in flags and i + 1 < len(toks):
            out[flags[t]] = toks[i + 1]
            i += 2
            continue
        if "=" in t and t.split("=", 1)[0] in flags:
            k, v = t.split("=", 1)
            out[flags[k]] = v
        elif not t.startswith("-") and out["competition"] is None:
            out["competition"] = t
        i += 1
    return out


def find_sample(root: Path) -> Path | None:
    for pat in ("data/sample_submission.csv", "data/*/sample_submission.csv", "data/sample_submission*.csv",
                "data/*sample*.csv", "sample_submission.csv"):
        hits = sorted(root.glob(pat))
        if hits:
            return hits[0]
    return None


def _read_csv(path: Path, max_rows: int = 5_000_000):
    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        header = next(r, [])
        ids, blanks, n = [], 0, 0
        for row in r:
            n += 1
            if row:
                ids.append(row[0])
            if any(c.strip() == "" or c.strip().lower() == "nan" for c in row) or len(row) != len(header):
                blanks += 1
            if n >= max_rows:
                break
    return header, ids, blanks, n


def validate_csv(sub: Path, sample: Path) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    try:
        h, ids, blanks, n = _read_csv(sub)
        rh, rids, _, rn = _read_csv(sample)
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        return [f"could not read CSV: {e}"], []
    if [c.strip() for c in h] != [c.strip() for c in rh]:
        if any(c == "" or c.startswith("Unnamed") for c in h):
            errors.append("header has an empty/'Unnamed' first column — the DataFrame index was written (use index=False)")
        errors.append(f"header {h} != sample header {rh}")
    if n != rn:
        errors.append(f"{n} rows but sample_submission has {rn}")
    if blanks:
        errors.append(f"{blanks} rows with empty/NaN values or wrong column count")
    if len(set(ids)) != len(ids):
        errors.append(f"{len(ids) - len(set(ids))} duplicated ids")
    if set(ids) != set(rids):
        errors.append(f"id set differs from sample ({len(set(rids) - set(ids))} missing, {len(set(ids) - set(rids))} unexpected)")
    elif ids != rids:
        warnings.append("ids are in a different order than sample_submission")
    return errors, warnings


def submissions_log(root: Path) -> Path:
    return root / S.STATE_DIR / "submissions.jsonl"


def today_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def submissions_today(root: Path) -> int:
    p = submissions_log(root)
    if not p.is_file():
        return 0
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(rec.get("time_utc", "")).startswith(today_utc()) and rec.get("status") != "failed":
            n += 1
    return n


def ledger_records(root: Path) -> list[dict]:
    p = root / S.STATE_DIR / "ledger.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


EXP_ID_RE = re.compile(r"\b(\d{4}_[a-z0-9-]+)\b")
