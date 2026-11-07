"""Is the honest track still moving? Plateau and crowd lines for the session brief and ``kgkit status``.

* **Plateau**: the accepted baseline (the latest ``keep``/``baseline`` decision) has not improved by more than the
  measured noise for a while, although experiments kept being logged. In UAT (a Playground) the honest stack stayed
  flat for the last 7 days while most of the remaining GPU quota went unused, and the sessions drifted to fitting the
  public LB. More members of the families already in the stack add ~0 then; the next gain is likelier from a model
  family, a data view or a signal the stack (and the public notebooks) lack.
* **Crowd**: many teams on the public leaderboard sit within noise of the best honest public score. Rank inside
  such a crowd is a coin flip on the private split; only something the crowd lacks moves a team out of it. The
  board comes from the cache ``kgkit standing`` writes (``finals.save_board``), never from the network here.

Standard library only: the SessionStart hook imports it.
"""

from __future__ import annotations

import datetime as dt
import statistics

from . import evalset as E
from . import state as S

ACCEPTED = ("baseline", "keep")
PLATEAU_DAYS = 3        # at least this many days without a gain beyond noise ...
PLATEAU_SHARE = 0.1     # ... and at least this share of the competition's span (first ledger record to deadline)
PLATEAU_TRIES = 3       # honest experiments logged since the last gain: fewer is a pause, not a plateau
CROWD_MIN, CROWD_MAX, CROWD_SHARE = 10, 50, 0.05  # teams within noise that make a crowd: 5% of the board, 10..50
BOARD_STALE_DAYS = 7
GPU_WORTH_H = 5         # GPU hours before the deadline worth mentioning


def _time(rec: dict, key: str) -> dt.datetime | None:
    """A ledger timestamp (local time, ``YYYY-MM-DDTHH:MM:SS``) as a naive datetime."""
    try:
        return dt.datetime.fromisoformat(str(rec.get(key) or "")[:19])
    except ValueError:
        return None


def _better(a: float, b: float, gib: bool, by: float = 0.0) -> bool:
    return a - b > by if gib else b - a > by


def threshold(root, st) -> tuple[float, str]:
    """The smallest gain that counts: twice the measured paired SE, else the competition's rope, else any gain."""
    noise = E.recent_noise(root, E.current_key(root, st))
    if noise:
        return 2 * noise["se"], f"2x the paired SE ~{E.fmt_se(noise['se'])}"
    if st.rope:
        return float(st.rope), f"the rope {E.fmt_se(st.rope)}"
    return 0.0, "noise not measured yet (`kgkit ledger compare`)"


def plateau(root, st, recs: list[dict], now: dt.datetime | None = None) -> dict | None:
    """When the accepted baseline last gained more than ``threshold``, and what was tried since. None when there is
    no accepted baseline with a score of its own, or no timestamps to tell."""
    holdout = st.holdout
    honest = [r for r in recs if E.own_score(r, holdout) and not r.get("recheck")]
    accepted = sorted((r for r in honest if r.get("decision") in ACCEPTED),
                      key=lambda r: _time(r, "decided_at") or _time(r, "time") or dt.datetime.min)
    if not accepted:
        return None
    thr, why = threshold(root, st)
    best, gain_at, gain_rec = None, None, None
    for r in accepted:
        if best is None or _better(r["cv"], best["cv"], st.greater_is_better, thr):
            best, gain_rec = r, r
            gain_at = _time(r, "decided_at") or _time(r, "time")
        elif _better(r["cv"], best["cv"], st.greater_is_better):
            best = r  # a gain within noise: the reference moves, the clock does not
    if gain_at is None:
        return None
    now = now or dt.datetime.now()
    since = [r for r in honest if (_time(r, "time") or dt.datetime.min) > gain_at and r is not gain_rec]
    first = min((t for t in (_time(r, "time") for r in recs) if t), default=None)
    end = S.deadline_date(st)
    span = (end - first.date()).days if end and first else None
    return {"baseline": best, "gain": gain_rec, "gain_at": gain_at, "days": (now - gain_at).total_seconds() / 86400,
            "tries": len(since), "threshold": thr, "threshold_why": why, "span_days": span,
            "days_left": S.deadline_days(st)}


def plateau_line(root, st, recs: list[dict], now: dt.datetime | None = None) -> str:
    """'PLATEAU: ...' when the accepted baseline has been flat for PLATEAU_DAYS (and PLATEAU_SHARE of the
    competition) despite PLATEAU_TRIES experiments, and more than FINAL_DAYS are left; else ''."""
    p = plateau(root, st, recs, now)
    if p is None or p["tries"] < PLATEAU_TRIES:
        return ""
    need = max(PLATEAU_DAYS, PLATEAU_SHARE * (p["span_days"] or 0))
    left = p["days_left"]
    if p["days"] < need or (left is not None and left <= S.FINAL_DAYS):
        return ""
    base = p["baseline"]
    s = (f"PLATEAU: the accepted baseline ({base['id']}, {'holdout' if st.holdout else 'CV'} {base['cv']:.6g}) has "
         f"not gained more than {E.fmt_se(p['threshold'])} ({p['threshold_why']}) for {p['days']:.0f} days, over "
         f"{p['tries']} experiment(s) since {p['gain_at']:%Y-%m-%d}"
         + (f", with {left} days left" if left is not None else "")
         + ". More of the families already in it adds ~0 now: run /kg-ideas and put first a model family, data view "
           "or signal the stack and the public notebooks lack (rank it above more members of an existing family)")
    gpu = _gpu_hours(root)
    if gpu is not None and gpu >= GPU_WORTH_H:
        s += f"; ~{gpu:.0f}h of Kaggle GPU quota are usable before the deadline (unused hours expire weekly)"
    s += ". Fitting the public LB is no way out: it cannot raise the private score"
    try:
        from . import finals as FN

        if FN.load_board(root, st.slug) is None:
            s += ". `python -m kgkit standing` reads the public LB, to see how many teams sit within noise of you"
    except Exception:
        pass
    return s + "."


def _gpu_hours(root) -> float | None:
    try:
        from . import backlog as B

        return B.gpu_budget(root)
    except Exception:
        return None


# ---------------------------------------------------------------------------- the crowd
def honest_public(recs: list[dict], st) -> dict | None:
    """The record with the best public score among those not fitted to the public LB."""
    rows = [r for r in recs if isinstance(r.get("lb_public"), (int, float)) and not E.lb_fitted(r)
            and not r.get("recheck")]
    return sorted(rows, key=lambda r: r["lb_public"], reverse=st.greater_is_better)[0] if rows else None


def public_noise(root, st) -> tuple[float | None, str]:
    """Noise of a public-LB gap between two files: twice the median ``se_public`` of recent ``ledger compare --public-n``
    runs. (None, '') when none was measured."""
    key = E.current_key(root, st)
    rows = [c["se_public"] for c in E.comparisons(root) if c.get("eval") == key
            and isinstance(c.get("se_public"), (int, float)) and c["se_public"] > 0][-5:]
    if not rows:
        return None, ""
    se = float(statistics.median(rows))
    return 2 * se, f"2x the public-LB SE ~{E.fmt_se(se)} measured by `ledger compare --public-n`"


def _unit(scores: list[float]) -> float:
    """The resolution the board shows scores at (1e-5 for five decimals)."""
    digits = 0
    for s in scores[:500]:
        text = repr(float(s))
        if "e" in text:
            continue
        digits = max(digits, len(text.split(".")[1].rstrip("0")) if "." in text else 0)
    return 10.0 ** -min(digits, 8) if digits else 1.0


def crowd(root, st, recs: list[dict]) -> dict | None:
    """Teams on the cached public board within noise of the best honest public score, or None (no board, a private
    one, or no honest public score)."""
    from . import finals as FN

    board = FN.load_board(root, st.slug)
    if not board or board.get("private"):
        return None
    scores = [float(s) for s in board.get("scores") or [] if isinstance(s, (int, float))]
    ours = honest_public(recs, st)
    if not scores or ours is None:
        return None
    band, why = public_noise(root, st)
    unit = _unit(scores)
    if band is None or band < unit:
        lead = (f"{why}, is below" if why else
                "no public-LB noise measured (`ledger compare --public-n`), so")
        band, why = unit, f"{lead} one unit of the scores Kaggle shows"
    eps = unit * 1e-6
    close = [i for i, s in enumerate(scores, 1) if abs(s - ours["lb_public"]) <= band + eps]
    need = max(CROWD_MIN, min(CROWD_MAX, round(CROWD_SHARE * len(scores))))
    return {"record": ours, "n": len(close), "ranks": (close[0], close[-1]) if close else None, "band": band,
            "why": why, "need": need, "teams": len(scores), "at": str(board.get("at") or "")[:10],
            "complete": bool(board.get("complete"))}


def crowd_line(root, st, recs: list[dict]) -> str:
    c = crowd(root, st, recs)
    if c is None or c["n"] < c["need"]:
        return ""
    r = c["record"]
    s = (f"CROWD: {c['n']:,} of {c['teams']:,}{'' if c['complete'] else '+'} teams on the public LB are within "
         f"+/-{E.fmt_se(c['band'])} ({c['why']}) of your best honest public score {r['lb_public']:.6g} ({r['id']}), "
         f"ranks {c['ranks'][0]:,}-{c['ranks'][1]:,}. Rank inside a crowd is noise on the private split; only what "
         "the crowd lacks (a model family or signal missing from the public notebooks and datasets everyone blends) "
         "moves you out of it")
    try:
        age = (dt.date.today() - dt.date.fromisoformat(c["at"])).days
    except ValueError:
        age = None
    when = f"`kgkit standing` of {c['at']}" + (f", {age} days old: run it again" if age and age > BOARD_STALE_DAYS
                                                else "")
    return s + f" ({when})."


def lines(root, st, recs: list[dict]) -> list[str]:
    """The brief's and ``kgkit status``'s progress lines while the competition runs. Never raises."""
    out = []
    for f in (plateau_line, crowd_line):
        try:
            line = f(root, st, recs)
        except Exception:
            line = ""
        if line:
            out.append(line)
    return out
