#!/usr/bin/env python3
"""What the plugin adds over plain Claude, as a markdown table built from `claude plugin eval` results.

    python scripts/eval_table.py plugins/kaggle-grandmaster/evals/results/<run>/aggregate-result.json [more.json ...]

Each result file is one run with the baseline arm (`--ablation with-without`, the default): every case was answered
with the plugin and again without it. Several files (the `portable` and the `shell` sets, or Haiku and Sonnet) are
merged; a case in two files of the same model keeps the latest. The table gives, per model, each case's score with
and without the plugin and the difference (`Δ`), then the mean per tag (`negative`, `regression`, `capability` and
each area tag) and overall, with the plugin version, judge model and run dates, so a reader can tell which numbers
are comparable. A run without the baseline arm (`--ablation none`) has no `W/OUT` column and is refused.

Tags are read from each case's `prompt.md` front matter (`--evals`, default the plugin's `evals/`), as the result
files do not carry them. Standard library only, so it runs on any Python (Windows included).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "plugins" / "kaggle-grandmaster" / "evals"
KINDS = ("negative", "regression", "capability")
PLATFORM = {"portable", "shell"}


def case_tags(evals: Path, name: str) -> list[str]:
    """The tags of case ``name`` (``tags: [a, b]`` in its prompt.md front matter); [] when it has none."""
    p = evals / name / "prompt.md"
    if not p.is_file():
        return []
    m = re.search(r"^tags:\s*\[([^\]]*)\]", p.read_text(encoding="utf-8"), re.MULTILINE)
    return [t.strip() for t in m.group(1).split(",") if t.strip()] if m else []


def load(paths: list[Path]) -> dict:
    """{model: {"cases": {name: row}, "versions": set, "judges": set, "dates": set}} from the result files."""
    by_model: dict = {}
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        suite = d.get("suite") or {}
        if suite.get("ablation") == "none":
            raise SystemExit(f"{p}: run with --ablation none (no baseline arm): nothing to compare")
        model = suite.get("modelOverride") or "default"
        m = by_model.setdefault(model, {"cases": {}, "versions": set(), "judges": set(), "dates": set()})
        m["versions"] |= {pl.get("version") for pl in suite.get("plugins") or [] if pl.get("version")}
        m["judges"].add(suite.get("judgeModel") or "default")
        started = str(d.get("startedAt") or "")
        m["dates"].add(started[:10])
        for c in d.get("cases") or []:
            a = c.get("aggregates") or {}
            if a.get("scoreWithout") is None:
                continue
            old = m["cases"].get(c["name"])
            if old is None or old["started"] <= started:
                m["cases"][c["name"]] = {"with": float(a.get("score") or 0), "without": float(a["scoreWithout"]),
                                         "delta": float(a.get("delta") or 0), "started": started}
    return by_model


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def table(by_model: dict, evals: Path) -> str:
    out = []
    for model, m in sorted(by_model.items()):
        cases = m["cases"]
        if not cases:
            continue
        tags = {name: case_tags(evals, name) for name in cases}
        out += [f"### {model} ({len(cases)} cases; plugin {', '.join(sorted(m['versions'])) or '?'}; judge "
                f"{', '.join(sorted(m['judges']))}; run {', '.join(sorted(m['dates']))})", "",
                "| group | cases | with plugin | without | Δ |", "|---|---|---|---|---|"]
        groups = [(k, [n for n in cases if k in tags[n]]) for k in KINDS]
        areas = sorted({t for ts in tags.values() for t in ts} - set(KINDS) - PLATFORM)
        groups += [(f"area: {a}", [n for n in cases if a in tags[n]]) for a in areas]
        groups.append(("**all**", list(cases)))
        for label, names in groups:
            if names:
                out.append(f"| {label} | {len(names)} | {mean([cases[n]['with'] for n in names]):.2f} | "
                           f"{mean([cases[n]['without'] for n in names]):.2f} | "
                           f"{mean([cases[n]['delta'] for n in names]):+.2f} |")
        out += ["", "| case | kind | with plugin | without | Δ |", "|---|---|---|---|---|"]
        for name in sorted(cases, key=lambda n: (-cases[n]["delta"], n)):
            kind = next((k for k in KINDS if k in tags[name]), "-")
            c = cases[name]
            out.append(f"| {name} | {kind} | {c['with']:.2f} | {c['without']:.2f} | {c['delta']:+.2f} |")
        out.append("")
    if not out:
        raise SystemExit("no case with a baseline score in these files")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", nargs="+", type=Path, help="aggregate-result.json files of with/without runs")
    ap.add_argument("--evals", type=Path, default=EVALS, help="the eval cases' folder (for their tags)")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # Markdown with a delta sign: UTF-8 even on a cp1252 Windows console
        sys.stdout.reconfigure(encoding="utf-8")
    print(table(load(a.results), a.evals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
