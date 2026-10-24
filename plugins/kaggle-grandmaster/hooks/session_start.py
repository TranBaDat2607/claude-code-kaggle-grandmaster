"""SessionStart: expose kgkit to the session and, inside a competition workspace, inject a compact
status brief (facts, deadline countdown, best CV/LB, accepted baseline, CV-LB agreement, recent
experiments, top of the idea backlog, submissions today, lessons from past competitions). Outside a
workspace, a repo with signs of Kaggle work (kernel-metadata.json, competition URLs) is pointed at
/kg-adopt rather than the full /kg-start bootstrap."""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

try:
    import _common as C
    from kgkit import __version__ as KGKIT_VERSION
except Exception:  # pragma: no cover - fail open
    sys.exit(0)

KGKIT_HOME = C.PLUGIN_ROOT.as_posix()
VERSION_RE = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']""", re.MULTILINE)


def _export_env() -> None:
    env_file = os.environ.get("CLAUDE_ENV_FILE")
    if not env_file:
        return
    try:
        with open(env_file, "a", encoding="utf-8") as f:
            f.write(f'export KGKIT_HOME="{KGKIT_HOME}"\n')
    except OSError:
        pass


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _own_cv(r: dict, holdout: bool = False) -> bool:
    return C.EV.own_score(r, holdout)


def vendored_note(root: Path) -> str:
    """Copies of kgkit in the workspace or a folder of it (`kgkit vendor src`) whose version is not the plugin's:
    scripts that import one, and `python -m kgkit` run from one, run that version, which a plugin update leaves as
    it was."""
    found = []
    for d in [root, *sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))]:
        init = d / "kgkit" / "__init__.py"
        try:
            if not init.is_file() or init.resolve().parent == C.PLUGIN_ROOT / "kgkit":
                continue
            m = VERSION_RE.search(init.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        version = m.group(1) if m else "of no known version"
        if version != KGKIT_VERSION:
            found.append((d.relative_to(root).as_posix(), version))
    if not found:
        return ""
    copies = "; ".join(f"{'' if d == '.' else d + '/'}kgkit is {v}" for d, v in found)
    one = len(found) == 1
    return (f"Vendored kgkit not at the plugin's version {KGKIT_VERSION}: {copies}. Scripts that import a copy, and "
            f"`python -m kgkit` run from one, run its version. Refresh {'it' if one else 'each'} with the plugin's "
            f"kgkit: `PYTHONPATH=\"{KGKIT_HOME}\" python -m kgkit vendor {found[0][0] if one else '<folder>'}` (later "
            "experiments then run the new code).")


def brief(root: Path) -> str:
    st = C.S.load(root)
    lines = []
    if st is None:
        return ""
    direction = "higher" if st.greater_is_better else "lower"
    lines.append(f"Competition workspace: **{st.slug}** — metric {st.metric or '?'} ({direction} is better), "
                 f"task {st.task or '?'}, target `{st.target or '?'}`; {C.S.deadline_line(st)}.")
    if st.code_competition:
        lines.append(f"CODE COMPETITION: runtime limit {st.runtime_limit_hours or '?'}h, internet "
                     f"{'allowed' if st.internet_allowed else 'OFF'} — budget inference early.")
    holdout = getattr(st, "holdout", False)
    if holdout:
        noise = C.EV.recent_noise(root, C.EV.current_key(root, st))
        judge = (f"measured {C.EV.format_noise(noise)} (`kgkit ledger compare`)" if noise else
                 "judge differences with `kgkit ledger compare` on stored holdout predictions (paired), not by the "
                 "holdout size")
        lines.append(f"Validation: fixed holdout" + (f" (n={st.holdout_n})" if st.holdout_n else "")
                     + f" + public LB, no OOF/K-fold: {judge}; error analysis runs on the holdout predictions; "
                       "public notebooks are judged on inputs and public score.")
    recs = C.ledger_records(root)
    if recs:
        scored = [r for r in recs if _own_cv(r, holdout) and not r.get("recheck")]
        best = sorted(scored, key=lambda r: r["cv"], reverse=st.greater_is_better)[:1]
        with_lb = [r for r in recs if isinstance(r.get("lb_public"), (int, float))]
        s = f"Ledger: {len(recs)} experiments"
        if best:
            s += f"; best {'holdout' if holdout else 'CV'} {best[0]['cv']:.5f} ({best[0]['id']})"
        if with_lb:
            bl = sorted(with_lb, key=lambda r: r["lb_public"], reverse=st.greater_is_better)[0]
            s += f"; best public LB {bl['lb_public']:.5f} ({bl['id']})"
            pairs = [r for r in with_lb if _own_cv(r, holdout) and not r.get("recheck")]
            corr = _pearson([r["cv"] for r in pairs], [r["lb_public"] for r in pairs])
            if corr is not None and len({round(r["cv"], 12) for r in pairs}) >= 4:
                s += (f"; CV-LB pearson {corr:.2f} over {len(pairs)} subs"
                      + (" (holdout CV: noisy)" if holdout else ""))
        lines.append(s + ".")
        accepted = next((r for r in reversed(recs) if r.get("decision") in ("baseline", "keep")
                         and not r.get("recheck")), None)
        if accepted is not None:
            cv = f"CV {accepted['cv']:.5f}" if isinstance(accepted.get("cv"), (int, float)) else "LB-only"
            lines.append(f"Accepted baseline (compare new ideas against this): {accepted['id']} {cv}.")
        elif len(scored) >= 2:
            lines.append("No accepted baseline yet: decide one (`kgkit ledger decide <id> baseline`) so experiments "
                         "are compared against a deliberate choice, not the luckiest CV.")
        for r in recs[-3:]:
            note = (r.get("notes") or "").replace("\n", " ")[:80]
            cv = f"{r['cv']:.5f}" if isinstance(r.get("cv"), (int, float)) else "?"
            lines.append(f"  - {r['id']}: CV {cv} {('— ' + note) if note else ''}")
    else:
        lines.append("Ledger is empty — next step is usually " + ("/kg-baseline (log holdout scores with `kgkit ledger "
                     "add`)." if holdout else "/kg-cv then /kg-baseline."))
    top = C.B.Backlog(root).ranked()[:3]
    if top:
        lines.append("Backlog top: " + "; ".join(f"#{i['id']} {i['idea'][:60]}" for i in top) + ".")
    lines.append(C.S.submissions_line(root, st) + ".")
    gq = C.G.quota_line(root)
    if gq:
        lines.append(gq)
    if not holdout and not (root / "data" / "folds.csv").exists():
        lines.append("No data/folds.csv yet — freeze the CV scheme before comparing experiments (/kg-cv). If "
                     "decisions are made on a fixed holdout + the public LB instead, record that: "
                     "`python -m kgkit config validation=holdout+lb holdout_n=<rows>`.")
    try:
        vendored = vendored_note(root)
    except OSError:
        vendored = ""
    if vendored:
        lines.append(vendored)
    return "\n".join(lines)


def lessons() -> str:
    p = Path.home() / ".kaggle-gm" / "lessons.md"
    if not p.is_file():
        return ""
    text = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not text:
        return ""
    return "Lessons from your past competitions (~/.kaggle-gm/lessons.md, latest):\n" + "\n".join(text[-12:])


def no_workspace(cwd: str) -> str:
    """Point a mid-competition repo at adoption; /kg-start is the full bootstrap for a new competition."""
    try:
        found = C.A.detect(cwd)
    except (OSError, ValueError):
        found = None
    if not found:
        return "Not in a competition workspace. Start one with /kaggle-grandmaster:kg-start <competition-slug>."
    slug = found["slugs"][0] if len(found["slugs"]) == 1 else "<competition-slug>"
    seen = ", ".join(found["evidence"])
    if found["slugs"]:
        seen += f"; competition {' / '.join(found['slugs'][:3])}"
    flags = " --code-competition --runtime-hours <H> --no-internet" if found["internet_off"] else ""
    return (f"This repo looks like existing Kaggle work ({seen}) but has no .kaggle-gm/ workspace, so submission "
            f"logging, the daily-budget checks, this status brief and `kgkit status`/`ledger` are off. Adopt it in "
            f"place with /kaggle-grandmaster:kg-adopt {slug} (writes only .kaggle-gm/, imports past submissions; "
            f"or `python -m kgkit adopt {slug}{flags} --deadline <date>`). Use /kg-start only for a new competition: "
            f"it bootstraps data, folds and a baseline from scratch.")


def main() -> None:
    payload = C.read_input()
    _export_env()
    parts = [f"[kaggle-grandmaster] kgkit home: {KGKIT_HOME} — run `PYTHONPATH=\"{KGKIT_HOME}\" python -m kgkit <cmd>` "
             f"(PowerShell: `$env:PYTHONPATH=\"{KGKIT_HOME}\"; python -m kgkit <cmd>`); templates in {KGKIT_HOME}/templates."]
    root = C.workspace(payload)
    if root is not None:
        b = brief(root)
        if b:
            parts.append(b)
            st = C.S.load(root)
            frozen = "fixed holdout" if st is not None and st.holdout else "frozen folds"
            parts.append(f"Follow the workspace CLAUDE.md rules ({frozen}, log every run, validate before submitting).")
        les = lessons()
        if les:
            parts.append(les)
    else:
        parts.append(no_workspace(payload.get("cwd") or os.getcwd()))
    C.emit("SessionStart", additionalContext="\n".join(parts))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
