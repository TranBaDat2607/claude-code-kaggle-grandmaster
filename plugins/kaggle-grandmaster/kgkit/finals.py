"""Which submissions Kaggle would take as the finals, and whether that is safe.

When a team selects no final submissions, Kaggle auto-selects the best public scores. Late in a competition those are
often *public-LB-fitted* files (a ranking tilted toward an estimate of the public labels): fine as display entries,
never as finals, since their public score says nothing about the private one. A ledger record tagged
``public-lb-fitted`` (``kgkit.evalset.lb_fitted``) is such a file. ``kgkit status`` and the session brief print
``lines(...)``: a warning when a fitted record is among the submissions Kaggle would auto-select, a one-line note once
the picks are recorded (``kgkit config final_picks=<id>,<id>``). ``kgkit final`` asks Kaggle what is selected
(``live``, through ``kgkit.kaggle_submissions`` in a child process) and records the answer.

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
    return {"n": n, "auto": auto, "fitted_auto": [e for e in auto if e["fitted"]], "picks": picks,
            "honest": _honest(st, by_id, score), "unlinked": len(loose), "days": S.deadline_days(st)}


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


def _warning(sel: dict, st: S.CompetitionState, recs: dict) -> list[str]:
    if sel["days"] is not None and sel["days"] < 0:  # the competition has ended
        return []
    left = f" ({sel['days']} day{'s' if sel['days'] != 1 else ''} left)" if sel["days"] is not None else ""
    url = URL.format(slug=st.slug)
    if sel["picks"]:
        bad = [p for p in sel["picks"] if p["fitted"]]
        if not bad:
            names = ", ".join(p["id"] or p["ref"] for p in sel["picks"])
            return [f"final picks (recorded): {names}; `python -m kgkit final` checks what Kaggle has selected"]
        names = ", ".join(p["id"] for p in bad)
        return [f"WARNING: recorded final pick(s) {names} are public-LB fitted (tilted toward the public labels: the "
                f"public score says nothing about the private one). Replace them on the website by "
                f"{utc_deadline(st)}{left}: {url}. Honest candidates: {_candidates(sel, recs)}."]
    fitted = sel["fitted_auto"]
    if not fitted:
        return []
    auto = sel["auto"]
    ties = " (tied at the cut: Kaggle takes some of these)" if len(auto) > sel["n"] else ""
    which = "all of them" if len(fitted) == len(auto) else f"{len(fitted)} of them"
    unlinked = (f" {sel['unlinked']} scored submission(s) in the log are not linked to a ledger record."
                if sel["unlinked"] else "")
    return [f"WARNING: with no finals selected, Kaggle takes the {sel['n']} best public scores: "
            f"{', '.join(_show(e) for e in auto)}{ties}; {which} public-LB fitted, so their public score says nothing "
            f"about the private one. Select finals by hand on the website by {utc_deadline(st)}{left}: {url}. "
            f"Honest candidates: {_candidates(sel, recs)}. Then record the picks: "
            f"`python -m kgkit config final_picks=<id>,<id>` (`python -m kgkit final` checks what Kaggle has "
            f"selected).{unlinked}"]


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
def live(slug: str, timeout: float = LIVE_TIMEOUT) -> tuple[dict | None, str | None]:
    """The team's scored submissions and the ones selected as finals, from Kaggle, in a child process
    (``kgkit.kaggle_submissions``). Returns ``({"selected": [...], "scored": [...]}, None)`` or ``(None, why)``.
    Never raises."""
    req = {"competition": slug, "max_pages": 5}
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
            "score": _num(row.get("public_score")), "id": rec["id"] if rec else None,
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
