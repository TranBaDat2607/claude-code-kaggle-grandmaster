"""A competition's own pages (description, evaluation, rules, data description, FAQ) as text.

    python -m kgkit pages                          # page names of this competition
    python -m kgkit pages --page evaluation        # one page as text
    python -m kgkit pages --save reports/pages     # every page -> reports/pages/<name>.md

Kaggle renders competition pages with JavaScript, so WebFetch of ``/overview`` or ``/rules`` often
returns an empty shell. The Kaggle CLI (2.2.4+) serves the hosts' HTML: ``kaggle competitions pages``.
Its argument parser takes the slug only after ``list`` and still wants a first positional, so the call is
``kaggle competitions pages <slug> list <slug> --content --format json``; this module hides that.
Page text is written by the hosts and printed inside an untrusted-content block (``kgkit.safety``).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

from . import safety

BLOCK_TAGS = {"p", "div", "section", "article", "table", "tr", "ul", "ol", "pre", "blockquote", "hr", "br"}


class _Text(HTMLParser):
    """Tags to plain-text structure. Source line breaks are kept: hosts write some pages in Markdown with
    inline HTML, where the line breaks carry the structure."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.href: str | None = None

    def handle_starttag(self, tag, attrs):
        if tag in BLOCK_TAGS:
            self.out.append("\n")
        if re.fullmatch(r"h[1-6]", tag):
            self.out.append("\n\n" + "#" * min(int(tag[1]) + 1, 4) + " ")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag in ("td", "th"):
            self.out.append(" | ")
        elif tag == "a":
            self.href = dict(attrs).get("href")

    def handle_endtag(self, tag):
        if tag == "a":
            if self.href and self.href.startswith("http"):
                self.out.append(f" ({self.href})")
            self.href = None
        if tag in BLOCK_TAGS or re.fullmatch(r"h[1-6]", tag):
            self.out.append("\n")

    def handle_data(self, data):
        self.out.append(re.sub(r"[ \t\f\v]+", " ", data))


def to_text(content: str) -> str:
    """A page's HTML or Markdown content as readable text."""
    p = _Text()
    p.feed(content or "")
    p.close()
    text = "\n".join(line.strip() if not line.startswith("    ") else line.rstrip()
                     for line in "".join(p.out).splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fetch(slug: str) -> list[dict]:
    """[{name, text}] for every page of ``slug``."""
    exe = shutil.which("kaggle")
    if exe is None:
        raise SystemExit("kaggle CLI not found: pip install -U kaggle (and authenticate; see skill kaggle-cli)")
    try:
        r = subprocess.run([exe, "competitions", "pages", slug, "list", slug, "--content", "--format", "json"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
                           env=safety.kaggle_env())
    except subprocess.TimeoutExpired:
        raise SystemExit("kaggle competitions pages timed out after 120s; retry, or check `kaggle competitions list`")
    out = r.stdout or ""
    rows = _json_list(out) if r.returncode == 0 else None
    if rows is None:
        msg = (r.stderr or out).strip() or "no JSON list in its output"
        if "invalid choice" in msg or "unrecognized arguments" in msg:
            msg += "\n(`kaggle competitions pages` needs Kaggle CLI 2.2.4 or later: pip install -U kaggle)"
        raise SystemExit(f"kaggle competitions pages failed: {msg[-800:]}")
    return [{"name": str(x.get("name", "")), "text": to_text(x.get("content") or "")}
            for x in rows if isinstance(x, dict)]


def _json_list(out: str) -> list | None:
    """The first JSON list in ``out``: the CLI can print warnings (which may contain '[') before it and
    other lines after it."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\[", out):
        try:
            rows, _ = dec.raw_decode(out, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(rows, list):
            return rows
    return None


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def pick(pages: list[dict], name: str) -> dict | None:
    """Case- and punctuation-insensitive match; a unique prefix also works ('data' -> 'data-description')."""
    k = _key(name)
    exact = [p for p in pages if _key(p["name"]) == k]
    if exact:
        return exact[0]
    pre = [p for p in pages if _key(p["name"]).startswith(k)]
    return pre[0] if len(pre) == 1 else None


def save(pages: list[dict], dest: str | Path) -> list[Path]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for p in pages:
        f = dest / f"{re.sub(r'[^a-z0-9]+', '-', p['name'].lower()).strip('-') or 'page'}.md"
        f.write_text(f"<!-- {safety.UNTRUSTED_NOTE} -->\n"
                     + safety.wrap(p["text"], "kaggle-cli", page=p["name"]) + "\n", encoding="utf-8")
        paths.append(f)
    return paths
