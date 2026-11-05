"""Which submissions Kaggle would take as the finals, whether that is safe, and how they did once the private
leaderboard is out.

When a team selects no final submissions, Kaggle auto-selects the best public scores. Late in a competition those are
often *public-LB-fitted* files (a ranking tilted toward an estimate of the public labels): fine as display entries,
never as finals, since their public score says nothing about the private one. A ledger record tagged
``public-lb-fitted`` (``kgkit.evalset.lb_fitted``) is such a file. ``kgkit status`` and the session brief print
``lines(...)``: a warning when a fitted record is among the submissions Kaggle would auto-select, and in the last
``S.FINAL_DAYS`` days a warning whenever no selection is confirmed (an unticked selection leaves the pick to Kaggle,
among files that often tie at the cut); a one-line note once Kaggle's selection has been confirmed. Picks typed by
hand (``kgkit config final_picks=<id>,<id>``) are only *intended*: the warning stays, and lists them as not
confirmed, because nothing has asked Kaggle. ``kgkit final`` asks Kaggle what is selected (``live``, through
``kgkit.kaggle_submissions`` in a child process), records the answer and stamps ``final_picks_confirmed``: that alone
silences the warning.

After the deadline ``kgkit final`` reports the outcome instead (``outcome``): the finals or the pool Kaggle picked
from, with their private scores, the best private score of any submission and the team's rank, and caches those
facts in ``.kaggle-gm/standing.json`` (``standing``) for the session brief's after-the-deadline checklist
(``ended_lines``). ``last_day_line`` is the prompt router's reminder on the deadline's day. ``link_scores`` matches
Kaggle's scored submissions to ledger records for ``kgkit ledger sync-lb``.

Kaggle has an API to list the selected submissions, none to select them: only the website does.

This module must stay standard-library only: hooks import it.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

from . import adopt, gpu, safety
from . import evalset as E
from . import state as S

URL = "https://www.kaggle.com/competitions/{slug}/submissions"
FITTED_WORDS = re.compile(r"public[\s_-]*lb[\s_-]*fitted", re.IGNORECASE)  # a submission description that says so
LEADING_NUMBER = re.compile(r"^\W*(\d{4})\b")  # "0054 [public-LB fitted] tilt ..." names a ledger record by its number
LIVE_TIMEOUT = 90
LEADERBOARD_TIMEOUT = 300  # the whole private leaderboard: ~0.6 s per 200 teams


# ---------------------------------------------------------------------------- records and submissions
def _num(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        return None
    return float(v)


def _cv_text(r: dict) -> str:
    cv = _num(r.get("cv"))
    return f"CV {cv:.5f}" if cv is not None else "LB-only"


def _label(text, width: int = 50) -> str:
    return " ".join(safety.clean(str(text or "")).split())[:width]


def _index(recs: list[dict]) -> tuple[dict, dict]:
    """Records by id and by number (``0054`` for ``0054_tilt-v1``)."""
    by_id = {r["id"]: r for r in recs if isinstance(r, dict) and isinstance(r.get("id"), str)}
    by_num: dict[int, list[dict]] = {}
    for i, r in by_id.items():
        head = i.split("_")[0]
        if head.isdigit():
            by_num.setdefault(int(head), []).append(r)
    return by_id, by_num


def _link(index: tuple[dict, dict], description=None, file=None, exp_id=None) -> dict | None:
    """The ledger record a submission belongs to: the id the hook logged, an id in its description or file name, or
    the number its description starts with when exactly one record has it."""
    by_id, by_num = index
    if isinstance(exp_id, str) and exp_id in by_id:
        return by_id[exp_id]
    m = adopt.EXP_ID_RE.search(f"{description or ''} {file or ''}")
    if m and m.group(1) in by_id:
        return by_id[m.group(1)]
    m = LEADING_NUMBER.match(str(description or ""))
    hits = by_num.get(int(m.group(1)), []) if m else []
    return hits[0] if len(hits) == 1 else None


def _find(index: tuple[dict, dict], ref: str) -> dict | None:
    """A record by id, name or number (a ``final_picks`` entry)."""
    by_id, by_num = index
    ref = str(ref)
    if ref in by_id:
        return by_id[ref]
    named = [r for r in by_id.values() if r.get("name") == ref]
    if named:
        return named[0]
    hits = by_num.get(int(ref), []) if ref.isdigit() else []
    return hits[0] if len(hits) == 1 else None


def submission_files(recs: list[dict], subs, rec_id: str) -> list[str]:
    """File names of the submissions in ``submissions.jsonl`` that belong to ledger record ``rec_id`` (by ``exp_id``
    or the id/number in the message or file name), newest first. Tolerates malformed rows."""
    index = _index([r for r in recs if isinstance(r, dict)])
    rows = []
    for row in subs:
        if not isinstance(row, dict) or row.get("status") == "failed" or not row.get("file"):
            continue
        rec = _link(index, row.get("message"), row.get("file"), row.get("exp_id"))
        if rec is not None and rec["id"] == rec_id:
            rows.append((str(row.get("time_utc") or ""), str(row["file"])))
    return [f for _, f in sorted(rows, key=lambda t: t[0], reverse=True)]


def _top(entries: list[dict], n: int, greater: bool) -> list[dict]:
    """The ``n`` best of entries sorted best first, and whoever ties with the last of them."""
    if not entries:
        return []
    cut = entries[min(n, len(entries)) - 1]["score"]
    return [e for e in entries if (e["score"] >= cut if greater else e["score"] <= cut)]


def _picks(st: S.CompetitionState) -> int:
    return max(1, int(st.final_submissions or 2))


def selection(st: S.CompetitionState, recs: list[dict], subs=()) -> dict:
    """What Kaggle would auto-select, from the public scores kgkit knows: ledger ``lb_public`` and the scored rows of
    ``submissions.jsonl`` (``kgkit adopt`` imports them). A row belongs to a record by ``exp_id`` or its message
    (``_link``); one that belongs to none is an unlinked entry. A record is fitted when it is tagged or a submission
    of it says so in its message. Tolerates malformed records: a hook calls this."""
    greater = st.greater_is_better
    better = max if greater else min
    index = _index([r for r in recs if isinstance(r, dict)])
    by_id = index[0]
    score = {}
    for i, r in by_id.items():
        s = _num(r.get("lb_public"))
        if s is not None and not r.get("recheck"):
            score[i] = s
    says: set[str] = set()
    loose = []
    for row in subs:
        if not isinstance(row, dict) or row.get("status") == "failed":
            continue
        s = _num(row.get("public_score"))
        if s is None:
            continue
        message = row.get("message") or row.get("file") or ""
        rec = _link(index, row.get("message"), row.get("file"), row.get("exp_id"))
        said = bool(FITTED_WORDS.search(str(message)))
        if rec is None:
            loose.append({"id": None, "label": _label(message), "score": s, "fitted": said, "rec": None})
        elif not rec.get("recheck"):
            score[rec["id"]] = better(score.get(rec["id"], s), s)
            if said:
                says.add(rec["id"])
    entries = [{"id": i, "label": i, "score": s, "fitted": E.lb_fitted(by_id[i]) or i in says, "rec": by_id[i]}
               for i, s in score.items()] + loose
    entries.sort(key=lambda e: e["score"], reverse=greater)
    n = _picks(st)
    auto = _top(entries, n, greater)
    picks = []
    for ref in st.final_picks or ():
        rec = _find(index, ref)
        picks.append({"ref": str(ref), "id": rec["id"] if rec else None,
                      "fitted": rec is not None and E.lb_fitted(rec)})
    confirmed = st.final_picks_confirmed if isinstance(st.final_picks_confirmed, str) and picks else ""
    return {"n": n, "auto": auto, "fitted_auto": [e for e in auto if e["fitted"]], "picks": picks,
            "confirmed": confirmed, "honest": _honest(st, by_id, score), "unlinked": len(loose),
            "days": S.deadline_days(st)}


def _honest(st: S.CompetitionState, by_id: dict, score: dict) -> list[tuple[str, list[str]]]:
    """Candidates that are safe to select, as ``(id, reasons)``: the accepted baseline, the best CV of its own, the
    best public score that is not fitted. Fitted, rechecked and discarded records are not candidates."""
    ok = [r for r in by_id.values() if not E.lb_fitted(r) and not r.get("recheck") and r.get("decision") != "discard"]
    sign = 1 if st.greater_is_better else -1
    found: dict[str, list[str]] = {}
    base = next((r for r in reversed(list(by_id.values())) if r.get("decision") in ("baseline", "keep")
                 and not E.lb_fitted(r) and not r.get("recheck")), None)
    if base is not None:
        found.setdefault(base["id"], []).append("accepted baseline")
    own = [r for r in ok if E.own_score(r, st.holdout)]
    if own:
        found.setdefault(max(own, key=lambda r: sign * r["cv"])["id"], []).append("best CV")
    public = [r for r in ok if r["id"] in score]
    if public:
        top = max(public, key=lambda r: sign * score[r["id"]])
        found.setdefault(top["id"], []).append("best public score not fitted")
    return list(found.items())


# ---------------------------------------------------------------------------- the text
def utc_deadline(st: S.CompetitionState) -> str:
    d = S.deadline_date(st)
    return f"{d.isoformat()} 23:59 UTC" if d else "the deadline (not set: `python -m kgkit config deadline=YYYY-MM-DD`)"


def deadline_text(st: S.CompetitionState) -> str:
    """The deadline in UTC and in this machine's time zone. Kaggle ends a competition at 23:59 UTC of its last day
    unless the Overview page says otherwise."""
    d = S.deadline_date(st)
    if d is None:
        return utc_deadline(st)
    local = dt.datetime.combine(d, dt.time(23, 59), tzinfo=dt.timezone.utc).astimezone()
    return f"{d.isoformat()} 23:59 UTC (= {local:%Y-%m-%d %H:%M} local, UTC{local:%z})"


def steps(st: S.CompetitionState) -> list[str]:
    """How to select finals: Kaggle has no API for it, so this is the website."""
    n = _picks(st)
    return [
        "Selecting finals (only the website can; Kaggle has no API for it):",
        f"  1. Open {URL.format(slug=st.slug)} while signed in as the team.",
        f"  2. Mark {n} submission{'s' if n > 1 else ''} as the final score: match each pick by file name or "
        "description and date, and never a public-LB fitted file.",
        f"  3. Check that exactly {n} {'are' if n > 1 else 'is'} marked, then run `python -m kgkit final`: it reads "
        "Kaggle's selection back and records it.",
        f"  Deadline: {deadline_text(st)} (if the competition's Overview page gives another time, that one counts)",
    ]


def _show(e: dict) -> str:
    return f"{e['id'] or 'unlinked: ' + e['label']} {e['score']:.5f}" + (" [public-LB fitted]" if e["fitted"] else "")


def _candidates(sel: dict, recs: dict) -> str:
    out = []
    for i, why in sel["honest"]:
        r = recs.get(i, {})
        lb = _num(r.get("lb_public"))
        out.append(f"{i} ({', '.join(why)}; {_cv_text(r)}" + (f", LB {lb:.5f}" if lb is not None else "") + ")")
    return "; ".join(out) or "none logged yet"


def _unconfirmed(sel: dict, st: S.CompetitionState, recs: dict, left: str, url: str, intended: str) -> str:
    """The last days, no selection confirmed on Kaggle and no fitted file in the way: nobody has picked the finals,
    so Kaggle will, among the best public scores, which this late often tie."""
    auto, n = sel["auto"], sel["n"]
    if auto:
        ties = f"; {len(auto)} tie at the cut and Kaggle picks among them" if len(auto) > n else ""
        takes = (f"With nothing selected, Kaggle takes the {n} best public score{'s' if n > 1 else ''}: "
                 f"{', '.join(_show(e) for e in auto)}{ties}.")
    else:
        takes = "With nothing selected, Kaggle takes the best public scores."
    return (f"WARNING: no finals confirmed on Kaggle{left}. {takes} Select finals on the website by "
            f"{utc_deadline(st)}: {url}, then run `python -m kgkit final`: it reads Kaggle's selection back, and this "
            f"warning stays until it confirms the picks. Honest candidates: {_candidates(sel, recs)}.{intended}")


def _warning(sel: dict, st: S.CompetitionState, recs: dict) -> list[str]:
    if sel["days"] is not None and sel["days"] < 0:  # the competition has ended
        return []
    left = f" ({sel['days']} day{'s' if sel['days'] != 1 else ''} left)" if sel["days"] is not None else ""
    url = URL.format(slug=st.slug)
    picks = sel["picks"]
    names = ", ".join(p["id"] or p["ref"] for p in picks)
    bad = ", ".join(p["id"] for p in picks if p["fitted"])
    if picks and sel["confirmed"]:  # `kgkit final` read these from Kaggle
        when = f"checked {sel['confirmed']}"
        if bad:
            return [f"WARNING: final(s) selected on Kaggle ({when}) {bad} are public-LB fitted (tilted toward the public "
                    f"labels: the public score says nothing about the private one). Replace them on the website by "
                    f"{utc_deadline(st)}{left}: {url}, then run `python -m kgkit final`. "
                    f"Honest candidates: {_candidates(sel, recs)}."]
        if len(picks) < sel["n"]:
            return [f"WARNING: only {len(picks)} of {sel['n']} finals are selected on Kaggle ({when}): {names}. Select the "
                    f"other {sel['n'] - len(picks)} on the website by {utc_deadline(st)}{left}: {url}, then run "
                    f"`python -m kgkit final`. Honest candidates: {_candidates(sel, recs)}."]
        return [f"finals selected on Kaggle ({when}): {names}; after any change on the website run "
                "`python -m kgkit final` again"]
    # nothing confirmed by Kaggle: what is recorded is only intended, and Kaggle would still auto-select
    intended = f" Intended picks (NOT confirmed on Kaggle): {names}." if picks else ""
    if bad:
        return [f"WARNING: intended final pick(s) {bad} are public-LB fitted (tilted toward the public labels: the "
                f"public score says nothing about the private one); choose others and select them on the website by "
                f"{utc_deadline(st)}{left}: {url}. Honest candidates: {_candidates(sel, recs)}. Then run "
                f"`python -m kgkit final`.{intended}"]
    fitted = sel["fitted_auto"]
    if not fitted:
        if sel["days"] is not None and sel["days"] <= S.FINAL_DAYS:
            return [_unconfirmed(sel, st, recs, left, url, intended)]
        if picks:
            return [f"final picks (intended, NOT confirmed on Kaggle): {names}; select them on the website by "
                    f"{utc_deadline(st)}{left}: {url}, then run `python -m kgkit final`"]
        return []
    auto = sel["auto"]
    ties = " (tied at the cut: Kaggle takes some of these)" if len(auto) > sel["n"] else ""
    which = "all of them" if len(fitted) == len(auto) else f"{len(fitted)} of them"
    unlinked = (f" {sel['unlinked']} scored submission(s) in the log are not linked to a ledger record."
                if sel["unlinked"] else "")
    return [f"WARNING: with no finals selected, Kaggle takes the {sel['n']} best public scores: "
            f"{', '.join(_show(e) for e in auto)}{ties}; {which} public-LB fitted, so their public score says nothing "
            f"about the private one. Select finals by hand on the website by {utc_deadline(st)}{left}: {url}. "
            f"Honest candidates: {_candidates(sel, recs)}. Then run `python -m kgkit final`: it reads Kaggle's "
            f"selection back, and this warning stays until it confirms the picks.{intended}{unlinked}"]


def lines(root, st: S.CompetitionState, recs: list[dict], subs=None) -> list[str]:
    """The lines ``kgkit status`` and the session brief print about the final picks: the warning, the recorded picks,
    or nothing. Never raises: a hook must not fail closed."""
    try:
        subs = S.submission_records(root) if subs is None else subs
        by_id = _index([r for r in recs if isinstance(r, dict)])[0]
        return _warning(selection(st, recs, subs), st, by_id)
    except Exception:
        return []


def not_fitted_note(recs: list[dict], st: S.CompetitionState) -> str:
    """For the ``best public LB`` line: empty unless the best public score is a fitted file, else what the best one
    that is not fitted is."""
    ranked = sorted((r for r in recs if isinstance(r, dict) and _num(r.get("lb_public")) is not None),
                    key=lambda r: r["lb_public"], reverse=st.greater_is_better)
    if not ranked or not E.lb_fitted(ranked[0]):
        return ""
    other = next((r for r in ranked if not E.lb_fitted(r)), None)
    return "public-LB fitted" + (f"; best not fitted: {other['lb_public']:.5f} ({other['id']}, {_cv_text(other)})"
                                 if other else "; no other public score yet")


# ---------------------------------------------------------------------------- what Kaggle says
def live(slug: str, timeout: float | None = None, leaderboard: bool = False) -> tuple[dict | None, str | None]:
    """The team's scored submissions and the ones selected as finals, from Kaggle, in a child process
    (``kgkit.kaggle_submissions``). Returns ``({"selected": [...], "scored": [...], "leaderboard": ...,
    "leaderboard_error": ...}, None)`` or ``(None, why)``. ``leaderboard``: also where the team stands
    (``kaggle_submissions.standing``); the whole listing is read, so it takes longer. Never raises."""
    req = {"competition": slug, "max_pages": 5, "leaderboard": bool(leaderboard)}
    timeout = timeout or (LEADERBOARD_TIMEOUT if leaderboard else LIVE_TIMEOUT)
    try:
        r = subprocess.run([gpu.kaggle_python() or sys.executable, "-m", "kgkit.kaggle_submissions"],
                           input=json.dumps(req), capture_output=True, encoding="utf-8", errors="replace",
                           timeout=timeout, cwd=str(Path(__file__).resolve().parents[1]), env=safety.kaggle_env())
    except subprocess.TimeoutExpired:
        return None, "the request to Kaggle timed out"
    except OSError as e:
        return None, f"the request to Kaggle did not start: {e}"
    try:
        res = json.loads((r.stdout or "").strip().splitlines()[-1])
        error = res.get("error")
        data = {k: [x for x in res.get(k) or [] if isinstance(x, dict)] for k in ("selected", "scored")}
        board = res.get("leaderboard")
        data["leaderboard"] = board if isinstance(board, dict) else None
        data["leaderboard_error"] = safety.clean(str(res["leaderboard_error"]))[:300] if res.get(
            "leaderboard_error") else None
    except (ValueError, IndexError, TypeError, AttributeError):
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        return None, "the request to Kaggle failed" + (f": {safety.clean(tail[-1])[:300]}" if tail
                                                       else f" (exit {r.returncode})")
    if error:
        return None, safety.clean(str(error))[:300]
    return data, None


def _view(index: tuple[dict, dict], row: dict) -> dict:
    text = f"{row.get('description') or ''} {row.get('file') or ''}"
    rec = _link(index, row.get("description"), row.get("file"))
    return {"ref": row.get("ref"), "date": str(row.get("date") or "")[:16].replace("T", " "),
            "score": _num(row.get("public_score")), "private": _num(row.get("private_score")),
            "id": rec["id"] if rec else None,
            "label": _label(row.get("description") or row.get("file")),
            "fitted": (rec is not None and E.lb_fitted(rec)) or bool(FITTED_WORDS.search(text))}


def _row_text(v: dict) -> str:
    score = f"{v['score']:.5f}" if v["score"] is not None else "   -   "
    return (f"  {score}  {v['id'] or '(no ledger record)':<34} {v['date']:<16}  {v['label']}"
            + ("  [public-LB fitted]" if v["fitted"] else ""))


def report(st: S.CompetitionState, recs: list[dict], data: dict | None, error: str | None,
           local=()) -> tuple[list[str], list[str] | None]:
    """The text of ``kgkit final`` and the ids to record as ``final_picks`` (None when Kaggle was not asked or has
    nothing selected). ``local``: ``lines(...)``, shown when Kaggle could not be asked."""
    if data is None:
        return ([f"Could not ask Kaggle which submissions are selected ({error}). What the ledger and the submission "
                 "log say:", *(local or ["no public-LB fitted file is among the best public scores in the ledger."])],
                None)
    index = _index([r for r in recs if isinstance(r, dict)])
    n, greater = _picks(st), st.greater_is_better
    scored = sorted((v for v in (_view(index, r) for r in data["scored"]) if v["score"] is not None),
                    key=lambda v: v["score"], reverse=greater)
    selected = [_view(index, r) for r in data["selected"]]
    out = [f"Kaggle lists {len(scored)} scored submission(s) for {st.slug}; {len(selected)} selected as finals."]
    if not selected:
        auto = _top(scored, n, greater)
        out += [f"Nothing is selected. If it stays so, Kaggle takes the {n} best public score(s):",
                *(_row_text(v) for v in auto)]
        if any(v["fitted"] for v in auto):
            out.append("WARNING: that includes public-LB fitted file(s): their public score says nothing about the "
                       f"private one. Select finals by hand before {utc_deadline(st)}.")
        return out, None
    out += [f"Selected as finals ({len(selected)} of {n}):", *(_row_text(v) for v in selected)]
    if any(v["fitted"] for v in selected):
        out.append("WARNING: a selected final is public-LB fitted: replace it on the website "
                   f"before {utc_deadline(st)}.")
    if len(selected) < n:
        out.append(f"Only {len(selected)} of {n} are selected: select the other {n - len(selected)} as well; what "
                   "Kaggle does with an empty slot is not something to rely on.")
    return out, [v["id"] or f"sub:{v['ref']}" for v in selected]


# ---------------------------------------------------------------------------- after the deadline
def past_deadline(st: S.CompetitionState) -> bool:
    days = S.deadline_days(st)
    return days is not None and days < 0


def revealed(data: dict | None) -> bool:
    """Whether Kaggle shows private scores for the team's submissions: the private leaderboard is out."""
    return bool(data) and any(isinstance(r, dict) and _num(r.get("private_score")) is not None
                              for r in data.get("scored") or ())


def same_score(old, new, digits: int | None = None) -> bool:
    """Whether ``old`` (perhaps typed by hand with more digits) agrees with ``new`` as Kaggle shows it, rounded to the
    ``digits`` decimals Kaggle gave (``kaggle_submissions.row``; else read off ``new``, which misses trailing
    zeros)."""
    o, n = _num(old), _num(new)
    if o is None or n is None:
        return False
    if not isinstance(digits, int) or isinstance(digits, bool) or digits < 0:
        text = repr(n)
        digits = len(text.split(".")[1]) if "." in text and "e" not in text.lower() else 12
    return abs(o - n) <= 0.5 * 10 ** -digits + 1e-12


def link_scores(recs: list[dict], rows: list[dict]) -> dict:
    """Kaggle's scored submissions (``kaggle_submissions.row``) matched to ledger records by ``_link``: the ledger id
    in the description or file name, or the number the description starts with. Returns ``{"records": {id: [rows,
    newest first]}, "unlinked": [rows]}``; rows with neither score are left out."""
    index = _index([r for r in recs if isinstance(r, dict)])
    linked: dict[str, list[dict]] = {}
    unlinked = []
    for row in rows:
        if not isinstance(row, dict) or (_num(row.get("public_score")) is None
                                         and _num(row.get("private_score")) is None):
            continue
        rec = _link(index, row.get("description"), row.get("file"))
        if rec is None:
            unlinked.append(row)
        else:
            linked.setdefault(rec["id"], []).append(row)
    for subs in linked.values():
        subs.sort(key=lambda r: str(r.get("date") or ""), reverse=True)
    return {"records": linked, "unlinked": unlinked}


def best_of_n(entries, n: int, greater: bool) -> tuple[float, float] | None:
    """Kaggle scores a team by the best private score among its finals. ``entries``: ``(selection score, private
    score)`` of the candidates; the ``n`` best by selection score are taken, any of those tied at the cut. Returns the
    lowest and highest team score over the ways the tie can be broken, None without private scores."""
    rows = [(s, p) for s, p in entries if _num(s) is not None and _num(p) is not None]
    if not rows or n < 1:
        return None
    rows.sort(key=lambda t: t[0], reverse=greater)
    cut = rows[min(n, len(rows)) - 1][0]
    above = [p for s, p in rows if (s > cut if greater else s < cut)]
    tied = sorted((p for s, p in rows if s == cut), reverse=not greater)  # the worst private first
    best = max if greater else min
    return best(above + tied[:max(1, n - len(above))]), best(above + tied)


def medal_cutoffs(teams: int) -> dict[str, int]:
    """The last rank of each medal by Kaggle's progression table for competitions, computed here: the API has no
    medal field, and Kaggle does not say how it rounds. Gold: top 10% under 100 teams, top 10 under 250, then top 10
    + 0.2% (one more per 500 teams). Silver: top 20% under 250 teams, top 50 under 1,000, then top 5%. Bronze: top
    40% under 250 teams, top 100 under 1,000, then top 10%."""
    n = max(0, int(teams))
    if n < 100:
        cut = (0.10 * n, 0.20 * n, 0.40 * n)
    elif n < 250:
        cut = (10, 0.20 * n, 0.40 * n)
    elif n < 1000:
        cut = (10 + 0.002 * n, 50, 100)
    else:
        cut = (10 + 0.002 * n, 0.05 * n, 0.10 * n)
    return {m: int(c + 1e-9) for m, c in zip(("gold", "silver", "bronze"), cut)}


def _fmt(v) -> str:
    return f"{v:.5f}" if isinstance(v, float) else "-"


def _board(data: dict | None) -> tuple[dict, dict | None]:
    """The leaderboard listing the website shows, and the public one when it was read separately."""
    board = (data or {}).get("leaderboard")
    if not isinstance(board, dict):
        return {}, None
    shown = board.get("shown") if isinstance(board.get("shown"), dict) else {}
    public = board.get("public") if isinstance(board.get("public"), dict) else None
    return shown, public


def _rank(listing: dict) -> tuple[int | None, list[float | None]]:
    scores = [_num(s) for s in listing.get("scores") or []]
    ranks = [r for r in listing.get("ranks") or [] if isinstance(r, int) and 1 <= r <= len(scores)]
    return (ranks[0] if ranks else None), scores


def tie_block(scores: list[float | None], rank: int) -> tuple[int, int]:
    """The first and last rank of the run of equal scores around ``rank`` (1-based): the teams that share the score
    Kaggle shows. An unscored row ties with nothing."""
    first = last = rank
    s = scores[rank - 1]
    if s is None:
        return rank, rank
    while first > 1 and scores[first - 2] == s:
        first -= 1
    while last < len(scores) and scores[last] == s:
        last += 1
    return first, last


def _places(n: int) -> str:
    return f"{n} place" if n == 1 else f"{n} places"


def _cuts_inside(tie: tuple[int, int], cut: dict[str, int]) -> list[str]:
    """The medals whose last rank falls inside the tie, so that the tie straddles the line."""
    return [m for m in ("gold", "silver", "bronze") if tie[0] <= cut[m] < tie[1]]


def _tie_line(scores: list[float | None], rank: int, complete: bool, cut: dict[str, int] | None) -> str | None:
    """How many teams share the team's shown score, and where a medal cut-off falls inside that tie."""
    first, last = tie_block(scores, rank)
    if last == first:
        return None
    n = f"{last - first + 1:,}" + (" or more" if not complete and last == len(scores) else "")
    line = f"  Tie: {n} teams share the team's shown score {_fmt(scores[rank - 1])} (ranks {first:,}-{last:,})"
    if cut is None:
        return line + "."
    inside = _cuts_inside((first, last), cut)
    if not inside:
        return line + "; no medal cut-off falls inside it."
    where = []
    for m in inside:
        c = cut[m]
        if rank > c:
            where.append(f"the {m} cut-off (rank {c:,}) falls inside it: the team, at rank {rank:,}, is "
                         f"{_places(rank - c)} short of it")
        else:
            spare = "no place" if c == rank else _places(c - rank)
            where.append(f"the {m} cut-off (rank {c:,}) falls inside it: the team, at rank {rank:,}, is inside it "
                         f"with {spare} to spare")
    return (line + ", and " + "; ".join(where) + ". Kaggle's order within the tie decides; the API shows scores "
            "rounded, so that order cannot be checked here.")


def team_score(data: dict | None) -> float | None:
    """The team's score on the leaderboard the website shows (the private one once revealed)."""
    rank, scores = _rank(_board(data)[0])
    return scores[rank - 1] if rank else None


def standing_lines(data: dict | None) -> list[str]:
    """The team's rank, the number of teams, the winning score and, on the private leaderboard, the medal zone by
    Kaggle's table (computed) with the scores at its cut-offs."""
    shown, public = _board(data)
    if not shown:
        return [f"Leaderboard not read ({(data or {}).get('leaderboard_error') or 'no answer'})."]
    rank, scores = _rank(shown)
    teams = len(scores)
    if not teams:
        return ["The leaderboard lists no teams yet."]
    complete = bool(shown.get("complete"))
    count = f"{teams:,}" if complete else f"at least {teams:,} (the listing was cut short)"
    board = data["leaderboard"]
    team = safety.clean(str(board.get("team") or ""))
    private = revealed(data)
    which = "Private leaderboard" if private else "Leaderboard (public until the private one is out)"
    if rank is None:
        who = f"team {team!r}" if team else "the team (its submissions carry no team name)"
        return [f"{which}: {count} teams, winning score {_fmt(scores[0])}; {who} is not on it."]
    named = [r for r in shown.get("ranks") or [] if isinstance(r, int)]
    same = f"; {len(named)} teams are named {team!r} (ranks {', '.join(map(str, named))}), the first is shown" if len(
        named) > 1 else ""
    pub_rank, pub_scores = _rank(public or {})
    pub = f"; public leaderboard rank {pub_rank:,} ({_fmt(pub_scores[pub_rank - 1])})" if pub_rank else ""
    out = [f"{which}: rank {rank:,} of {count} teams (top {100 * rank / teams:.1f}%), team score "
           f"{_fmt(scores[rank - 1])}, winning score {_fmt(scores[0])}{pub}{same}."]
    cut = medal_cutoffs(teams) if private and complete else None
    if cut:
        medal = next((m for m in ("gold", "silver", "bronze") if rank <= cut[m]), "no medal")
        zones = ", ".join(f"{m} to rank {cut[m]:,}" + (f" ({_fmt(scores[cut[m] - 1])})" if cut[m] >= 1 else "")
                          for m in ("gold", "silver", "bronze"))
        out.append(f"  Medal zone by Kaggle's progression table, computed here: {medal} ({zones}). The API has no "
                   "medal field; this holds in competitions that award medals, and ranks can still move while Kaggle "
                   "removes teams.")
    tie = _tie_line(scores, rank, complete, cut)
    if tie:
        out.append(tie)
    return out


def _out_row(v: dict) -> str:
    return (f"  {_fmt(v['score'])} / {_fmt(v['private'])}  {v['id'] or '(no ledger record)':<34} {v['date']:<16}  "
            f"{v['label']}" + ("  [public-LB fitted]" if v["fitted"] else ""))


def _short(v: dict) -> str:
    """A submission's name in a sentence: its ledger id, else the id its description gives, else the description's
    start and Kaggle's ref."""
    if v["id"]:
        return v["id"]
    m = adopt.EXP_ID_RE.search(v["label"] or "")
    if m:
        return m.group(1)
    return f"{v['label'][:30] or '?'} (ref {v['ref']})"


def _names(views: list[dict], most: int = 6) -> str:
    names = [_short(v) for v in views]
    return ", ".join(names[:most]) + (f" and {len(names) - most} more" if len(names) > most else "")


def _auto_heading(n: int, auto: list) -> str:
    """What Kaggle did with no finals selected: it took the best public scores, and among files tied at the cut its
    API does not say which."""
    if len(auto) > n:
        return (f"Competition ended; no finals were selected, so Kaggle auto-selected from the {n} best public "
                f"score(s): {len(auto)} tie at the cut, and Kaggle's API can't tell which of the tied ones it took")
    return f"Competition ended; no finals were selected, so Kaggle auto-selected the {n} best public score(s)"


def outcome(st: S.CompetitionState, recs: list[dict], data: dict) -> tuple[list[str], list[str] | None]:
    """The text of ``kgkit final`` once the competition has ended, and the ids to record as ``final_picks`` (None
    when nothing was selected): the finals, or the pool Kaggle picked from when nobody selected, with their private
    scores; the team's standing; the best private score of any submission and how far the finals were from it."""
    index = _index([r for r in recs if isinstance(r, dict)])
    n, greater = _picks(st), st.greater_is_better
    better = max if greater else min
    views = [_view(index, r) for r in data["scored"]]
    scored = sorted((v for v in views if v["score"] is not None), key=lambda v: v["score"], reverse=greater)
    selected = [_view(index, r) for r in data["selected"]]
    picks = [v["id"] or f"sub:{v['ref']}" for v in selected] or None
    end = S.deadline_date(st)
    out = [f"The competition has ended{f' ({end.isoformat()})' if end else ''}. Kaggle lists {len(scored)} scored "
           f"submission(s) for {st.slug}; {len(selected)} selected as finals."]
    auto = [] if selected else _top(scored, n, greater)
    privates = [v for v in views if v["private"] is not None]
    if not privates:
        out.append("No private scores yet: Kaggle shows them once the private leaderboard is final (a code "
                   "competition first re-runs the selected notebooks on the private test). Run `python -m kgkit final` "
                   "again then.")
        if selected:
            out += [f"Selected as finals ({len(selected)} of {n}):", *(_row_text(v) for v in selected)]
        else:
            out += [_auto_heading(n, auto) + ":", *(_row_text(v) for v in auto)]
        return out, picks
    out += standing_lines(data)
    on_board = team_score(data)
    if selected:
        out += [f"Selected as finals ({len(selected)} of {n}), public / private:", *(_out_row(v) for v in selected)]
        got = [v["private"] for v in selected if v["private"] is not None]
        final = better(got) if got else None
        out.append(f"Kaggle scores a team by the best private score among its finals: {_fmt(final)}.")
    else:
        out += [_auto_heading(n, auto) + " (public / private):", *(_out_row(v) for v in auto)]
        final = on_board
        if on_board is not None:
            hit = [v for v in auto if v["private"] is not None and abs(v["private"] - on_board) < 1e-9]
            over = [v for v in auto if v["private"] is not None and better(v["private"], on_board) != on_board]
            if hit:
                out.append(f"The team's private score {_fmt(on_board)} matches {_names(hit)}: Kaggle's picks included "
                           + ("it" if len(hit) == 1 else "at least one of them")
                           + (f", and not {_names(over)}, which scored higher" if over else "") + ".")
            else:
                out.append(f"The team's private score {_fmt(on_board)} matches none of them: Kaggle's automatic choice "
                           "differed from the best public scores listed here.")
        else:
            span = best_of_n([(v["score"], v["private"]) for v in scored], n, greater)
            if span:
                final = span[0] if span[0] == span[1] else None
                out.append("Depending on how Kaggle broke the tie, the team scored "
                           + (_fmt(span[0]) if span[0] == span[1] else f"between {_fmt(span[0])} and {_fmt(span[1])}")
                           + " privately (the leaderboard would say which).")
    top = better(v["private"] for v in privates)
    tops = [v for v in privates if abs(v["private"] - top) < 1e-9]
    line = f"Best private score of any submission: {_fmt(top)} ({_names(tops)})."
    if final is not None:
        gap = (top - final) if greater else (final - top)
        line += (" The finals reached it." if gap <= 1e-9 else
                 f" The finals scored {_fmt(final)}, {gap:.5f} short of it.")
    out.append(line)
    return out, picks


# ---------------------------------------------------------------------------- what the session brief says after the end
STANDING = "standing.json"  # the outcome `kgkit final` last read after the deadline, for the session brief
POSTMORTEM = Path("reports") / "postmortem.md"


def standing_path(root) -> Path:
    return Path(root) / S.STATE_DIR / STANDING


def standing(st: S.CompetitionState, recs: list[dict], data: dict) -> dict:
    """The facts of ``outcome`` the session brief repeats, without asking Kaggle again: what was selected (or the
    pool Kaggle auto-selected from), the team's rank and score, the ranks that share its shown score (``tie``), and,
    once the private leaderboard is out, the medal zone and each cut-off's last rank and score (``cutoffs``)."""
    index = _index([r for r in recs if isinstance(r, dict)])
    n, greater = _picks(st), st.greater_is_better
    views = [_view(index, r) for r in data.get("scored") or ()]
    scored = sorted((v for v in views if v["score"] is not None), key=lambda v: v["score"], reverse=greater)
    selected = [_short(_view(index, r)) for r in data.get("selected") or ()]
    auto = [] if selected else [_short(v) for v in _top(scored, n, greater)]
    shown = _board(data)[0]
    rank, scores = _rank(shown)
    private, complete = revealed(data), bool(shown.get("complete"))
    medal = cutoffs = None
    if private and complete and scores:
        cut = medal_cutoffs(len(scores))
        cutoffs = {m: [cut[m], scores[cut[m] - 1] if cut[m] >= 1 else None] for m in ("gold", "silver", "bronze")}
        if rank:
            medal = next((m for m in ("gold", "silver", "bronze") if rank <= cut[m]), "no medal")
    tie = list(tie_block(scores, rank)) if rank else None
    return {"slug": st.slug, "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "n": n,
            "selected": selected, "auto": auto, "tied": len(auto) > n, "private": private, "board_read": bool(shown),
            "rank": rank, "teams": len(scores), "complete": complete, "score": scores[rank - 1] if rank else None,
            "medal": medal, "cutoffs": cutoffs, "tie": tie if tie and tie[1] > tie[0] else None}


def save_standing(root, st: S.CompetitionState, recs: list[dict], data: dict, by: str = "final") -> bool:
    """Cache ``standing`` in ``.kaggle-gm/standing.json``, with the command that read it (``by``). Never raises."""
    try:
        p = standing_path(root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({**standing(st, recs, data), "by": by}, indent=1), encoding="utf-8")
        return True
    except Exception:
        return False


def load_standing(root, slug: str) -> dict | None:
    try:
        d = json.loads(standing_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get("slug") == slug else None


def _standing_line(c: dict | None) -> str:
    if c is None:
        return "Final standing: not read yet: `python -m kgkit final` reads it once the private leaderboard is out."
    when = str(c.get("at") or "")[:10]
    if not c.get("private"):
        return (f"Final standing: the private leaderboard was not out at the last check ({when}): run "
                "`python -m kgkit final` again.")
    rank, teams = c.get("rank"), c.get("teams")
    if not c.get("board_read"):
        return (f"Final standing: private scores are out, but the leaderboard was not read ({when}): run "
                "`python -m kgkit final` again.")
    if not isinstance(rank, int) or not isinstance(teams, int) or not teams:
        return f"Final standing: the team is not on the private leaderboard (`kgkit final` of {when})."
    medal = c.get("medal")
    zone = ("" if not medal else ", outside the medal zones by Kaggle's table (computed)" if medal == "no medal"
            else f", {medal} zone by Kaggle's table (computed)")
    count = f"{teams:,}" if c.get("complete") else f"at least {teams:,}"
    by = c.get("by") if c.get("by") in ("final", "standing") else "final"
    return (f"Final standing: private rank {rank:,} of {count} teams (top {100 * rank / teams:.1f}%), score "
            f"{_fmt(_num(c.get('score')))}{zone}{_cached_tie(c)} (`kgkit {by}` of {when}).")


def _cached_tie(c: dict) -> str:
    """The tie ``standing.json`` records, and the medal cut-offs that fall inside it."""
    tie, cutoffs = c.get("tie"), c.get("cutoffs")
    if not (isinstance(tie, list) and len(tie) == 2 and all(isinstance(r, int) for r in tie) and tie[1] > tie[0]):
        return ""
    out = f", {tie[1] - tie[0] + 1:,} teams tied on that score at ranks {tie[0]:,}-{tie[1]:,}"
    cut = {m: v[0] for m, v in (cutoffs.items() if isinstance(cutoffs, dict) else ())
           if isinstance(v, list) and v and isinstance(v[0], int)}
    inside = _cuts_inside((tie[0], tie[1]), cut) if len(cut) == 3 else []
    if inside:
        many = len(inside) > 1
        out += (f", across the {' and '.join(inside)} cut-off{'s' if many else ''} at rank{'s' if many else ''} "
                + ", ".join(f"{cut[m]:,}" for m in inside))
    return out


def _finals_line(st: S.CompetitionState, c: dict | None) -> str:
    if c is not None:
        sel = [str(x) for x in c.get("selected") or ()]
        if sel:
            return f"Finals: {len(sel)} of {c.get('n') or _picks(st)} selected on Kaggle: {', '.join(sel)}."
        auto = [str(x) for x in c.get("auto") or ()]
        if not auto:
            return "Finals: none were selected, and Kaggle listed no public score to auto-select from."
        return (f"Finals: none were selected, so Kaggle auto-selected from {', '.join(auto)}"
                + (" (tied at the cut: the API can't tell which it took)." if c.get("tied") else "."))
    picks = [str(p) for p in st.final_picks or ()]
    if picks and st.final_picks_confirmed:
        return (f"Finals: {', '.join(picks)} selected on Kaggle (checked {st.final_picks_confirmed}): "
                "`python -m kgkit final` reports how they did.")
    return ("Finals: not checked since the deadline: `python -m kgkit final` says what was selected, or what Kaggle "
            "auto-selected.")


def _repo_line(root: Path) -> str:
    from . import audit  # stdlib-only too; imported here so the tool hooks, which import this module, never load it

    c = audit.load_cache(root)
    if c is None:
        return ("Repo: not audited for competition data: `python -m kgkit audit-repo` before anything goes public "
                "(/kg-showcase runs it).")
    vis = {True: "the repo is PUBLIC", False: "the repo is not public"}.get(c.get("public"), "visibility unknown")
    leaks, checks = c.get("leaks") or [], c.get("checks") or []
    found = (f"found competition data in git ({len(leaks)} file(s), see the warning below)" if leaks else
             f"found no copies, and {len(checks)} judgment call(s) to review (`python -m kgkit audit-repo`)" if checks
             else "found no competition data")
    return f"Repo: {vis}; audit of {str(c.get('at') or '')[:10]} {found}."


def _lessons_line(slug: str) -> str:
    p = Path.home() / ".kaggle-gm" / "lessons.md"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    if slug and slug.lower() in text.lower():
        return "Lessons: appended to ~/.kaggle-gm/lessons.md."
    return f"Lessons: none for {slug} in ~/.kaggle-gm/lessons.md yet (/kg-postmortem appends them)."


def ended_lines(root, st: S.CompetitionState) -> list[str]:
    """The session brief after the deadline: what is left to do now that the CV, submission and GPU checks no longer
    matter. The standing and finals come from the last ``kgkit final`` (``standing.json``); the post-mortem, the repo
    audit and the lessons from the files those steps write. No network. Never raises."""
    root = Path(root)
    out = ["After the competition (the CV, submission and GPU checks are off):"]
    c = load_standing(root, st.slug)
    for build in (lambda: _standing_line(c), lambda: _finals_line(st, c),
                  lambda: "Post-mortem: reports/postmortem.md " + ("written." if (root / POSTMORTEM).is_file() else
                                                                   "not written yet: run /kg-postmortem."),
                  lambda: _repo_line(root), lambda: _lessons_line(st.slug)):
        try:  # one malformed cache costs its own line, not the brief
            out.append("  - " + build())
        except Exception:
            continue
    return out


def last_day_line(st: S.CompetitionState) -> str:
    """On the deadline's own day (UTC), how many finals are selected on Kaggle as kgkit last read it, while fewer than
    the allowed number are: the prompt router repeats it once per session, so a session that began days earlier still
    hears it. Empty otherwise. Never raises."""
    try:
        if S.deadline_days(st) != 0:
            return ""
        n = _picks(st)
        picks = [str(p) for p in st.final_picks or ()]
        confirmed = st.final_picks_confirmed if picks and isinstance(st.final_picks_confirmed, str) else ""
        k = len(picks) if confirmed else 0
        if k >= n:
            return ""
        how = f"checked {confirmed}, `python -m kgkit final` checks again" if confirmed else "`python -m kgkit final`"
        intended = f" Intended picks (NOT confirmed on Kaggle): {', '.join(picks)}." if picks and not confirmed else ""
        return (f"Last day (deadline {utc_deadline(st)}): {k} of {n} finals selected on Kaggle ({how}). Unselected "
                f"slots go to Kaggle's auto-selection by public score. Kaggle has no API to select them: tell the user "
                f"to tick them on the website, {URL.format(slug=st.slug)}, then run `python -m kgkit final` to read the "
                f"selection back.{intended}")
    except Exception:
        return ""
