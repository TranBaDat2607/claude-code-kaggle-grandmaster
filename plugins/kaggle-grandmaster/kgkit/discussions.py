"""Competition discussions from Meta Kaggle, Kaggle's official daily export of its forums.

The Kaggle CLI has no discussion commands, and discussion pages are hard to fetch. Meta Kaggle
(dataset ``kaggle/meta-kaggle``, refreshed daily) has every topic (``ForumTopics.csv``, ~70 MB), the
competition -> forum mapping plus rule facts (``Competitions.csv``) and, optionally, every message
(``ForumMessages.csv``, ~1.8 GB). Files are cached once per machine in ``~/.kaggle-gm/meta-kaggle``
(override with ``KGKIT_META_DIR``) and refreshed when older than ``--max-age`` days.

    python -m kgkit discussions sync                    # index this competition's topics -> reports/discussions.csv
    python -m kgkit discussions top -n 20 --sort votes  # votes | recent | replies | views
    python -m kgkit discussions search "leak|shake|cv.*lb"
    python -m kgkit discussions solutions --competition <finished-slug>   # "1st place solution" write-ups
    python -m kgkit discussions read <topic_id>         # full thread (needs `sync --messages`), else its URL

Meta Kaggle lags by up to a day; for the last hours' threads, fetch the URL. A competition that is too
new may have no forum id in the export yet: then use the website (WebFetch) or a Kaggle MCP server.
"""

from __future__ import annotations

import html
import os
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import pandas as pd

TOPIC_COLS = ["Id", "ForumId", "KernelId", "CreationDate", "LastCommentDate", "Title", "IsSticky", "TotalViews",
              "Score", "TotalMessages", "TotalReplies"]
COMP_COLS = ["Id", "Slug", "Title", "ForumId", "DeadlineDate", "TeamMergerDeadlineDate", "MaxTeamSize",
             "BanTeamMergers", "MaxDailySubmissions", "EvaluationAlgorithmName", "HostSegmentTitle"]
DATE_FMT = "%m/%d/%Y %H:%M:%S"
SOLUTION_RX = re.compile(r"(?i)\b(?:\d+(?:st|nd|rd|th)|first|second|third|top[- ]?\d+|gold)\b.{0,30}\b(?:place|"
                         r"solution|write-?up|approach)\b|\bsolution\b|\bwrite-?up\b")
SORTS = {"votes": "Score", "replies": "TotalReplies", "views": "TotalViews", "recent": "Created"}


def meta_dir() -> Path:
    return Path(os.environ.get("KGKIT_META_DIR") or Path.home() / ".kaggle-gm" / "meta-kaggle")


def _fresh(p: Path, max_age_days: float) -> bool:
    return p.is_file() and (time.time() - p.stat().st_mtime) < max_age_days * 86400


def ensure(files: list[str], max_age_days: float = 3, force: bool = False) -> Path:
    """Download (or refresh) Meta Kaggle files into the cache via the Kaggle CLI."""
    d = meta_dir()
    d.mkdir(parents=True, exist_ok=True)
    for name in files:
        p = d / name
        if not force and _fresh(p, max_age_days):
            continue
        exe = shutil.which("kaggle")
        if exe is None:
            if p.is_file():
                print(f"kaggle CLI not found; using the cached (stale) {name}")
                continue
            raise SystemExit("kaggle CLI not found: pip install kaggle (and authenticate; see skill kaggle-cli)")
        print(f"downloading kaggle/meta-kaggle {name} ...", flush=True)
        r = subprocess.run([exe, "datasets", "download", "kaggle/meta-kaggle", "-f", name, "-p", str(d), "--force"],
                           capture_output=True, text=True, timeout=3600)
        z = d / f"{name}.zip"
        if z.is_file():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(d)
            z.unlink()
        if r.returncode != 0 or not p.is_file():
            if p.is_file():
                print(f"download of {name} failed; using the cached copy")
                continue
            raise SystemExit(f"could not download {name}: {(r.stderr or r.stdout).strip()[-500:]}")
        os.utime(p)
    return d


def competition_row(slug: str, d: Path | None = None) -> dict:
    d = d or meta_dir()
    comps = pd.read_csv(d / "Competitions.csv", usecols=lambda c: c in COMP_COLS)
    row = comps[comps["Slug"] == slug]
    if row.empty:
        raise SystemExit(f"{slug!r} is not in Meta Kaggle's Competitions.csv (new competition? refresh with --force)")
    return row.iloc[0].to_dict()


def index(slug: str, d: Path | None = None) -> pd.DataFrame:
    """All discussion topics of one competition's forum (notebook comment threads excluded)."""
    d = d or meta_dir()
    comp = competition_row(slug, d)
    fid = comp.get("ForumId")
    if fid is None or pd.isna(fid):
        raise SystemExit(f"Meta Kaggle has no forum id for {slug!r} yet (very new competition). Use "
                         f"https://www.kaggle.com/competitions/{slug}/discussion via WebFetch or a Kaggle MCP server.")
    topics = pd.read_csv(d / "ForumTopics.csv", usecols=lambda c: c in TOPIC_COLS)
    t = topics[(topics["ForumId"] == int(fid)) & topics["KernelId"].isna()].copy()
    t["Created"] = pd.to_datetime(t["CreationDate"], format=DATE_FMT, errors="coerce")
    t["Url"] = [f"https://www.kaggle.com/competitions/{slug}/discussion/{i}" for i in t["Id"]]
    t["Title"] = t["Title"].fillna("").astype(str)
    return t.drop(columns=["KernelId"]).reset_index(drop=True)


def top(t: pd.DataFrame, n: int = 20, sort: str = "votes") -> pd.DataFrame:
    if sort not in SORTS:
        raise ValueError(f"sort must be one of {sorted(SORTS)}")
    return t.sort_values(SORTS[sort], ascending=False, na_position="last").head(n)


def search(t: pd.DataFrame, pattern: str) -> pd.DataFrame:
    return t[t["Title"].str.contains(pattern, case=False, regex=True, na=False)].sort_values("Score", ascending=False)


def solutions(t: pd.DataFrame) -> pd.DataFrame:
    """Write-up threads ("1st place solution", "gold solution", ...), best-voted first."""
    return t[t["Title"].str.contains(SOLUTION_RX, na=False)].sort_values("Score", ascending=False)


def _strip_html(s: str) -> str:
    s = re.sub(r"(?is)<(script|style).*?</\1>", "", s or "")
    s = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</h\d>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\n{3,}", "\n\n", html.unescape(s)).strip()


def read_thread(topic_id: int, d: Path | None = None, chunksize: int = 500_000) -> list[dict]:
    """Messages of one topic from the cached ForumMessages.csv (streamed in chunks; ~1.8 GB)."""
    d = d or meta_dir()
    p = d / "ForumMessages.csv"
    if not p.is_file():
        raise FileNotFoundError(p)
    header = pd.read_csv(p, nrows=0).columns
    text_col = "RawMarkdown" if "RawMarkdown" in header else "Message"
    cols = [c for c in ("Id", "ForumTopicId", "PostDate", "Medal", text_col) if c in header]
    out = []
    for chunk in pd.read_csv(p, usecols=cols, chunksize=chunksize):
        hit = chunk[chunk["ForumTopicId"] == topic_id]
        if not hit.empty:
            out += hit.to_dict("records")
    for m in out:
        raw = m.get(text_col)
        m["text"] = _strip_html(raw if isinstance(raw, str) else "")
    out.sort(key=lambda m: pd.to_datetime(m.get("PostDate"), format=DATE_FMT, errors="coerce") or pd.Timestamp(0))
    return out


def format_table(t: pd.DataFrame) -> str:
    if t.empty:
        return "(no topics)"
    lines = ["| votes | replies | created | title | id |", "|---|---|---|---|---|"]
    for _, r in t.iterrows():
        created = r["Created"].strftime("%Y-%m-%d") if pd.notna(r["Created"]) else ""
        title = str(r["Title"]).replace("|", "/")[:90]
        lines.append(f"| {int(r['Score']) if pd.notna(r['Score']) else ''} | "
                     f"{int(r['TotalReplies']) if pd.notna(r['TotalReplies']) else ''} | {created} | {title} | {r['Id']} |")
    return "\n".join(lines)


def facts(comp: dict) -> str:
    """Rule facts the recon and teaming advice need, straight from Kaggle's export."""
    keep = {"DeadlineDate": "deadline", "TeamMergerDeadlineDate": "merger deadline", "MaxTeamSize": "max team size",
            "BanTeamMergers": "team mergers banned", "MaxDailySubmissions": "daily submissions",
            "EvaluationAlgorithmName": "metric", "HostSegmentTitle": "category"}
    return "; ".join(f"{v}: {comp[k]}" for k, v in keep.items() if k in comp and pd.notna(comp[k]))
