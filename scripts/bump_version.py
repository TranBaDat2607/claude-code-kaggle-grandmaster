"""Check or bump the kaggle-grandmaster version in every place it is recorded.

    python scripts/bump_version.py --check                 # exit 1 if the versions disagree
    python scripts/bump_version.py --current               # print the version
    python scripts/bump_version.py --part patch --note "Fix X (#12)"

A bump rewrites plugin.json, the marketplace entry and kgkit.__version__, and turns the
``## Unreleased`` section of CHANGELOG.md into ``## X.Y.Z — YYYY-MM-DD`` (or adds that heading
with ``--note`` as its only entry). Run by .github/workflows/release.yml on every merge to main.
Standard library only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = "kaggle-grandmaster"
SEMVER = r"(\d+)\.(\d+)\.(\d+)"


def files(root: Path) -> dict[str, Path]:
    return {
        "plugin.json": root / "plugins" / PLUGIN / ".claude-plugin" / "plugin.json",
        "marketplace.json": root / ".claude-plugin" / "marketplace.json",
        "kgkit": root / "plugins" / PLUGIN / "kgkit" / "__init__.py",
        "changelog": root / "CHANGELOG.md",
    }


def versions(root: Path) -> dict[str, str | None]:
    f = files(root)
    market = json.loads(f["marketplace.json"].read_text(encoding="utf-8"))
    entry = next((p for p in market.get("plugins", []) if p.get("name") == PLUGIN), {})
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', f["kgkit"].read_text(encoding="utf-8"), re.MULTILINE)
    return {
        "plugin.json": json.loads(f["plugin.json"].read_text(encoding="utf-8")).get("version"),
        "marketplace.json": entry.get("version"),
        "kgkit": m.group(1) if m else None,
    }


def current(root: Path) -> str:
    v = versions(root)
    if len(set(v.values())) != 1 or not re.fullmatch(SEMVER, v["plugin.json"] or ""):
        raise SystemExit(f"versions disagree or are not X.Y.Z: {v}")
    return v["plugin.json"]


def next_version(version: str, part: str) -> str:
    major, minor, patch = (int(x) for x in re.fullmatch(SEMVER, version).groups())
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def _sub_once(pattern: str, repl: str, text: str, what: str, flags: int = 0) -> str:
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        raise SystemExit(f"could not find the version in {what}")
    return new


def update_changelog(text: str, new: str, note: str | None, today: str) -> str:
    heading = f"## {new} — {today}"
    if re.search(r"^## Unreleased[ \t]*$", text, re.MULTILINE):
        return re.sub(r"^## Unreleased[ \t]*$", heading, text, count=1, flags=re.MULTILINE)
    entry = f"{heading}\n\n- {note or 'Maintenance release.'}\n\n"
    m = re.search(r"^## ", text, re.MULTILINE)
    if m:
        return text[:m.start()] + entry + text[m.start():]
    return text.rstrip("\n") + "\n\n" + entry


def bump(root: Path, part: str, note: str | None = None, today: str | None = None) -> str:
    old = current(root)
    new = next_version(old, part)
    f = files(root)
    ver_line = r'("version"\s*:\s*")' + re.escape(old) + '"'
    f["plugin.json"].write_text(_sub_once(ver_line, rf'\g<1>{new}"', f["plugin.json"].read_text(encoding="utf-8"),
                                          "plugin.json"), encoding="utf-8")
    market = f["marketplace.json"].read_text(encoding="utf-8")
    # only the kaggle-grandmaster entry: the first "version" after its "name"
    market = _sub_once(r'("name"\s*:\s*"' + PLUGIN + r'"[^{}]*?"version"\s*:\s*")' + re.escape(old) + '"',
                       rf'\g<1>{new}"', market, "marketplace.json", flags=re.DOTALL)
    f["marketplace.json"].write_text(market, encoding="utf-8")
    f["kgkit"].write_text(_sub_once(r'^(__version__\s*=\s*")' + re.escape(old) + '"', rf'\g<1>{new}"',
                                    f["kgkit"].read_text(encoding="utf-8"), "kgkit/__init__.py", re.MULTILINE),
                          encoding="utf-8")
    if f["changelog"].is_file():
        today = today or dt.date.today().isoformat()
        f["changelog"].write_text(update_changelog(f["changelog"].read_text(encoding="utf-8"), new, note, today),
                                  encoding="utf-8")
    if current(root) != new:
        raise SystemExit("bump left the versions out of sync")
    return new


def release_notes(root: Path, version: str) -> str:
    """The CHANGELOG section of ``version`` (without its heading)."""
    text = files(root)["changelog"].read_text(encoding="utf-8")
    m = re.search(rf"^## {re.escape(version)}\b[^\n]*\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="exit 1 if the versions disagree")
    g.add_argument("--current", action="store_true", help="print the current version")
    g.add_argument("--part", choices=["patch", "minor", "major"], help="bump this part and print the new version")
    g.add_argument("--notes", metavar="VERSION", help="print the CHANGELOG section of VERSION")
    ap.add_argument("--note", help="CHANGELOG entry when there is no '## Unreleased' section")
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args(argv)
    if a.check:
        v = versions(a.root)
        if len(set(v.values())) != 1:
            print(f"versions disagree: {v}", file=sys.stderr)
            return 1
        print(f"versions in sync: {v['plugin.json']}")
        return 0
    if a.current:
        print(current(a.root))
    elif a.notes:
        print(release_notes(a.root, a.notes))
    else:
        print(bump(a.root, a.part, a.note))
    return 0


if __name__ == "__main__":
    sys.exit(main())
