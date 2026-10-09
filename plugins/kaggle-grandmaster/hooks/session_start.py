"""SessionStart: expose kgkit to the session and, inside a competition workspace, inject a compact
status brief (facts, deadline countdown, best CV/LB, accepted baseline, CV-LB agreement, recent
experiments, top of the idea backlog, submissions today, lessons from past competitions)."""

from __future__ import annotations

import datetime as dt
import math
import os
import sys
from pathlib import Path

try:
    import _common as C
except Exception:  # pragma: no cover - fail open
    sys.exit(0)

KGKIT_HOME = C.PLUGIN_ROOT.as_posix()


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


def _deadline(st) -> str:
    if not st.deadline:
        return "deadline: unknown (set it in .kaggle-gm/competition.json)"
    try:
        d = dt.date.fromisoformat(st.deadline[:10])
    except ValueError:
        return f"deadline: {st.deadline}"
    days = (d - dt.datetime.now(dt.timezone.utc).date()).days
    phase = ("ENDED — run /kg-postmortem" if days < 0 else "FINAL DAYS — robustness + /kg-final" if days <= 3
             else "endgame — freeze ensemble design" if days <= 14 else "exploration")
    return f"deadline {d.isoformat()} ({days} days left; phase: {phase})"


def gpu_line(root: Path) -> str:
    """Cached Kaggle GPU quota (no network at session start) and remote runs still in flight."""
    q = C.G.cached_quota(root)
    active = [r for r in C.G.runs(root) if r.get("status") in C.G.ACTIVE]
    if q is None and not active:
        return ""
    s = ""
    if q is not None:
        avail = C.G.available_hours(root, q)
        s = f"Kaggle GPU quota (cached {q.get('fetched_at', '?')}): {avail:.1f}h plannable of {q.get('total_h', 0):.0f}h"
        refresh = C.G._parse_time(q.get("refresh_at"))
        if refresh is not None:
            h = (refresh - dt.datetime.now(dt.timezone.utc)).total_seconds() / 3600
            s += f", resets in {h:.0f}h"
            if 0 < h <= 36 and avail >= 1:
                s += " - USE IT OR LOSE IT: queue long useful runs before the reset"
        s += "."
    if active:
        s += f" Remote GPU runs in flight: {', '.join(r['kernel'] for r in active)} (`python -m kgkit gpu status`)."
    return s.strip()


def brief(root: Path) -> str:
    st = C.S.load(root)
    lines = []
    if st is None:
        return ""
    direction = "higher" if st.greater_is_better else "lower"
    lines.append(f"Competition workspace: **{st.slug}** — metric {st.metric or '?'} ({direction} is better), "
                 f"task {st.task or '?'}, target `{st.target or '?'}`; {_deadline(st)}.")
    if st.code_competition:
        lines.append(f"CODE COMPETITION: runtime limit {st.runtime_limit_hours or '?'}h, internet "
                     f"{'allowed' if st.internet_allowed else 'OFF'} — budget inference early.")
    recs = C.ledger_records(root)
    if recs:
        scored = [r for r in recs if isinstance(r.get("cv"), (int, float))]
        best = sorted(scored, key=lambda r: r["cv"], reverse=st.greater_is_better)[:1]
        with_lb = [r for r in scored if isinstance(r.get("lb_public"), (int, float))]
        s = f"Ledger: {len(recs)} experiments"
        if best:
            s += f"; best CV {best[0]['cv']:.5f} ({best[0]['id']})"
        if with_lb:
            bl = sorted(with_lb, key=lambda r: r["lb_public"], reverse=st.greater_is_better)[0]
            s += f"; best public LB {bl['lb_public']:.5f} ({bl['id']})"
            corr = _pearson([r["cv"] for r in with_lb], [r["lb_public"] for r in with_lb])
            if corr is not None:
                s += f"; CV-LB pearson {corr:.2f} over {len(with_lb)} subs"
        lines.append(s + ".")
        accepted = next((r for r in reversed(scored) if r.get("decision") in ("baseline", "keep")), None)
        if accepted is not None:
            lines.append(f"Accepted baseline (compare new ideas against this): {accepted['id']} CV {accepted['cv']:.5f}.")
        elif len(scored) >= 2:
            lines.append("No accepted baseline yet: decide one (`kgkit ledger decide <id> baseline`) so experiments "
                         "are compared against a deliberate choice, not the luckiest CV.")
        for r in recs[-3:]:
            note = (r.get("notes") or "").replace("\n", " ")[:80]
            cv = f"{r['cv']:.5f}" if isinstance(r.get("cv"), (int, float)) else "?"
            lines.append(f"  - {r['id']}: CV {cv} {('— ' + note) if note else ''}")
    else:
        lines.append("Ledger is empty — next step is usually /kg-cv then /kg-baseline.")
    top = C.B.Backlog(root).ranked()[:3]
    if top:
        lines.append("Backlog top: " + "; ".join(f"#{i['id']} {i['idea'][:60]}" for i in top) + ".")
    used = C.submissions_today(root)
    lines.append(f"Submissions logged today (UTC): {used}/{st.daily_submissions}.")
    gq = gpu_line(root)
    if gq:
        lines.append(gq)
    if not (root / "data" / "folds.csv").exists():
        lines.append("No data/folds.csv yet — freeze the CV scheme before comparing experiments (/kg-cv).")
    return "\n".join(lines)


def lessons() -> str:
    p = Path.home() / ".kaggle-gm" / "lessons.md"
    if not p.is_file():
        return ""
    text = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not text:
        return ""
    return "Lessons from your past competitions (~/.kaggle-gm/lessons.md, latest):\n" + "\n".join(text[-12:])


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
            parts.append("Follow the workspace CLAUDE.md rules (frozen folds, log every run, validate before submitting).")
        les = lessons()
        if les:
            parts.append(les)
    else:
        parts.append("Not in a competition workspace. Start one with /kaggle-grandmaster:kg-start <competition-slug>.")
    C.emit("SessionStart", additionalContext="\n".join(parts))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
