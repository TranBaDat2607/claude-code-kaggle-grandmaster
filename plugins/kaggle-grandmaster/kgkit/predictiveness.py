"""Which signal predicted the private leaderboard: ``kgkit ledger predictiveness``.

After the deadline the post-mortem question is not whether the CV and the public score tracked the private score over
the whole competition: early, large gains make any signal look good (the run that prompted this showed a CV-LB
Pearson of 0.98). It is whether they ranked the *strongest* submissions, the ones the finals were picked from; there
the same run's 58-study holdout had a Spearman of 0.03 with private, against 0.70 for the public score.

This reports the Spearman rank correlation of each signal with the private score over every scored submission, over
the ledger records with a validation score of their own, and over the top k of those by public score and by CV, each
with a 95% bootstrap interval; then what each way of picking the finals would have scored, since Kaggle scores a team
by the best private score among its finals; then how far private fell from public. Public-LB-fitted files are left
out: their public score is tilted toward the public labels by construction.

Inputs: the ledger (CV, public and private after ``kgkit ledger sync-lb``) and Kaggle's list of submissions that
``sync-lb`` saves (``S.kaggle_list``), which also holds the submissions no ledger record names.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from . import evalset as E
from . import finals as FN
from . import safety
from . import state as S
from .experiment import has_own_cv

N_BOOT = 2000
MIN_N = 3  # fewer points: no correlation
MIN_CI = 5  # fewer points: no interval
SEED = 0


def _num(v) -> float | None:
    return FN._num(v)


def saved_list(root) -> dict:
    """``{"scored": [...], "selected": [...]}`` as ``sync-lb`` saved them; empty lists when it has not run."""
    try:
        d = json.loads(S.kaggle_list(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    d = d if isinstance(d, dict) else {}
    return {k: [r for r in d.get(k) or [] if isinstance(r, dict)] for k in ("scored", "selected")}


def gather(recs: list[dict], rows: list[dict], holdout: bool) -> dict:
    """The points: ``subs``, every scored submission with both scores (from Kaggle's list, else one per ledger record);
    ``own``, the ledger records with a validation score of their own and both scores. ``fitted`` counts the
    public-LB-fitted submissions left out."""
    recs = [r for r in recs if isinstance(r, dict) and isinstance(r.get("id"), str)]
    index = FN._index(recs)
    subs, fitted = [], 0
    for row in rows:
        v = FN._view(index, row)  # linked to its record; fitted when the record is tagged or the description says so
        if v["score"] is None or v["private"] is None:
            continue
        if v["fitted"]:
            fitted += 1
            continue
        subs.append({"public": v["score"], "private": v["private"], "date": str(row.get("date") or ""),
                     "name": FN._short(v)})
    if not rows:  # no list saved: one point per ledger record
        for r in recs:
            pub, prv = _num(r.get("lb_public")), _num(r.get("lb_private"))
            if pub is None or prv is None or r.get("recheck"):
                continue
            if E.lb_fitted(r):
                fitted += 1
                continue
            subs.append({"public": pub, "private": prv, "date": str(r.get("time") or ""), "name": r["id"]})
    own = []
    for r in recs:
        pub, prv = _num(r.get("lb_public")), _num(r.get("lb_private"))
        if pub is None or prv is None or r.get("recheck") or not has_own_cv(r, holdout):
            continue
        own.append({"name": r["id"], "cv": float(r["cv"]), "public": pub, "private": prv})
    return {"subs": subs, "own": own, "fitted": fitted}


def spearman(x, y, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Spearman's rho (Pearson on average ranks, so ties are handled) and a 95% percentile bootstrap interval over
    the points. ``rho`` is None under ``MIN_N`` points or when a signal is constant; no interval under ``MIN_CI``."""
    from scipy.stats import rankdata

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    out = {"n": n, "rho": None, "lo": None, "hi": None}
    if n < MIN_N or np.ptp(x) == 0 or np.ptp(y) == 0:
        return out

    def rho(ix) -> float:
        a, b = rankdata(x[ix]), rankdata(y[ix])
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    out["rho"] = rho(np.arange(n))
    if n >= MIN_CI:
        rng = np.random.default_rng(seed)
        draws = np.array([rho(rng.integers(0, n, n)) for _ in range(n_boot)])
        draws = draws[np.isfinite(draws)]
        if len(draws) >= n_boot // 2:
            out["lo"], out["hi"] = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    return out


def _fmt_rho(r: dict) -> str:
    if r["rho"] is None:
        return "-"
    ci = f" [{r['lo']:+.2f}, {r['hi']:+.2f}]" if r["lo"] is not None else ""
    return f"{r['rho']:+.2f}{ci}"


def _best_first(points: list[dict], key: str, greater: bool) -> list[dict]:
    return sorted(points, key=lambda p: p[key], reverse=greater)


def correlations(g: dict, top: int, greater: bool) -> list[tuple[str, int, dict | None, dict]]:
    """``(set, n, CV->private or None, public->private)`` for each set of points."""
    out = [("every scored submission", len(g["subs"]), None,
            spearman([p["public"] for p in g["subs"]], [p["private"] for p in g["subs"]]))]
    own = g["own"]
    sets = [("records with their own validation score", own)] if own else []
    if len(own) > top:
        sets += [(f"top {top} of those by public score", _best_first(own, "public", greater)[:top]),
                 (f"top {top} of those by validation score", _best_first(own, "cv", greater)[:top])]
    for name, pts in sets:
        priv = [p["private"] for p in pts]
        out.append((name, len(pts), spearman([p["cv"] for p in pts], priv), spearman([p["public"] for p in pts], priv)))
    return out


def _span(r: tuple[float, float] | None, digits: int = 5) -> str:
    if r is None:
        return "-"
    lo, hi = sorted(r)
    return f"{lo:.{digits}f}" if lo == hi else f"{lo:.{digits}f} to {hi:.{digits}f}, depending on a tie at the cut"


def finals_private(st: S.CompetitionState, recs: list[dict], saved: dict) -> dict:
    """How many finals were selected on Kaggle and the team's private score from them (the best among them).
    Kaggle's list saved by ``sync-lb`` is the truth; without it, a selection ``kgkit final`` confirmed."""
    best = max if st.greater_is_better else min
    if saved["scored"]:
        got = [p for p in (_num(r.get("private_score")) for r in saved["selected"]) if p is not None]
        return {"known": True, "n": len(saved["selected"]), "private": best(got) if got else None}
    if st.final_picks and st.final_picks_confirmed:
        by_id = {r.get("id"): r for r in recs if isinstance(r, dict)}
        got = [p for p in (_num((by_id.get(i) or {}).get("lb_private")) for i in st.final_picks) if p is not None]
        return {"known": True, "n": len(st.final_picks), "private": best(got) if got else None}
    return {"known": False, "n": 0, "private": None}


def report(st: S.CompetitionState, g: dict, finals: dict, top: int) -> list[str]:
    greater = st.greater_is_better
    best = max if greater else min
    val = "holdout" if st.holdout else "CV"
    rows = correlations(g, top, greater)
    out = [f"Which signal predicted the private leaderboard ({st.slug}): Spearman rank correlation with the private "
           "score, 95% bootstrap interval in brackets",
           f"  {'set':44s} {'n':>4s}   {val + ' -> private':22s}   public -> private"]
    for name, n, cv, pub in rows:
        out.append(f"  {name:44s} {n:4d}   {(_fmt_rho(cv) if cv else '-'):22s}   {_fmt_rho(pub)}")
    notes = []
    if not g["own"]:
        notes.append(f"no ledger record has a {val} of its own and both scores: `kgkit ledger sync-lb` links a "
                     "submission to its record by the ledger id its description starts with")
    elif len(g["own"]) <= top:
        notes.append(f"the top-{top} sets need more than {top} records with their own {val} and both scores "
                     "(`--top` sets k)")
    if g["fitted"]:
        notes.append(f"{g['fitted']} public-LB-fitted submission(s) left out")
    pts = g["own"] or g["subs"]
    distinct = len({p["private"] for p in pts})
    if pts and distinct < len(pts):
        notes.append(f"private scores tie: {distinct} distinct values among {len(pts)}, as Kaggle rounds them")
    notes.append("neighbouring submissions are variants of one another, so n overstates the evidence: trust a gap "
                 "between two signals only when it is large next to these intervals")
    out += [f"  note: {x}" for x in notes]

    n = max(1, int(st.final_submissions or 2))
    out += ["", f"Picking {n} final(s) by each signal (Kaggle scores a team by the best private score among its "
                "finals):"]
    if g["own"]:
        by_cv = FN.best_of_n([(p["cv"], p["private"]) for p in g["own"]], n, greater)
        out.append(f"  best {val + ':':24s} private {_span(by_cv)}")
    none = finals["known"] and not finals["n"]
    if g["subs"]:
        out.append(f"  {'best public score:':29s} private "
                   f"{_span(FN.best_of_n([(p['public'], p['private']) for p in g['subs']], n, greater))}"
                   + ("; Kaggle's rule when nothing is selected" if none else ""))
    if not finals["known"]:
        out.append(f"  {'your finals:':29s} unknown (`python -m kgkit ledger sync-lb` saves Kaggle's selection)")
    elif none:
        out.append(f"  {'your finals:':29s} none selected on Kaggle: the public-score rule above applied")
    elif finals["private"] is None:
        out.append(f"  {'your finals:':29s} {finals['n']} selected, no private score on record")
    else:
        out.append(f"  {'your finals:':29s} private {finals['private']:.5f} ({finals['n']} selected)")
    every = g["subs"] or g["own"]
    if every:
        top_p = best(p["private"] for p in every)
        names = [p["name"] for p in every if abs(p["private"] - top_p) < 1e-9]
        out.append(f"  {'best possible:':29s} private {top_p:.5f} ({', '.join(names[:5])}"
                   + (f" and {len(names) - 5} more" if len(names) > 5 else "") + ")")

    if g["subs"]:
        sign = 1 if greater else -1
        d = np.array([sign * (p["public"] - p["private"]) for p in g["subs"]])
        out += ["", f"Public minus private (positive: private was worse): mean {d.mean():+.4f} (SD {d.std():.4f}), "
                    f"from {d.min():+.4f} to {d.max():+.4f} over {len(d)} submissions."]
    return out


def _when(text: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(text)[:19])
    except ValueError:
        return None


def figures(outdir, g: dict, st: S.CompetitionState) -> tuple[list[Path], str | None]:
    """public_vs_private.png, private_over_time.png and cv_vs_private.png, or why none was written."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return [], "matplotlib is not installed (pip install matplotlib)"
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    greater = st.greater_is_better
    val = "holdout" if st.holdout else "CV"

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        written.append(out / name)

    subs = g["subs"]
    if subs:
        x, y = [p["public"] for p in subs], [p["private"] for p in subs]
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(x, y, s=18, alpha=0.7, label="submission")
        lo, hi = min(x + y), max(x + y)
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="grey", label="private = public")
        ax.set(xlabel="public score", ylabel="private score", title=f"{st.slug}: public vs private")
        ax.legend(fontsize=8)
        save(fig, "public_vs_private.png")
        dated = sorted(((_when(p["date"]), p) for p in subs if _when(p["date"])), key=lambda t: t[0])
        if dated:
            when = [t for t, _ in dated]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(when, [p["public"] for _, p in dated], s=16, facecolors="none", edgecolors="tab:blue",
                       label="public")
            ax.scatter(when, [p["private"] for _, p in dated], s=16, color="tab:orange", label="private")
            running, cur = [], None
            for _, p in dated:
                cur = p["private"] if cur is None else (max if greater else min)(cur, p["private"])
                running.append(cur)
            ax.step(when, running, where="post", color="tab:orange", lw=1, alpha=0.6, label="best private so far")
            ax.set(xlabel="submitted", ylabel="score", title=f"{st.slug}: scores over time")
            ax.legend(fontsize=8)
            fig.autofmt_xdate()
            save(fig, "private_over_time.png")
    own = g["own"]
    if len(own) >= 2:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter([p["cv"] for p in own], [p["private"] for p in own], s=18, alpha=0.7)
        ax.set(xlabel=f"{val} score", ylabel="private score", title=f"{st.slug}: {val} vs private")
        save(fig, "cv_vs_private.png")
    return written, None


def main(led, a) -> None:
    """``kgkit ledger predictiveness [--top K] [--plot DIR]``."""
    st = S.load(led.root)
    if st is None:
        raise SystemExit("not inside a kaggle-grandmaster workspace (no .kaggle-gm/competition.json)")
    saved = saved_list(led.root)
    g = gather(led.records(), saved["scored"], st.holdout)
    if not g["subs"] and not g["own"]:
        raise SystemExit("no submission has both a public and a private score on record: once the private leaderboard "
                         "is out, run `python -m kgkit ledger sync-lb` first")
    if not saved["scored"]:
        print("note: Kaggle's list is not saved (`python -m kgkit ledger sync-lb` saves it): every scored submission "
              "below is a ledger record, and submissions no record names are missing")
    top = max(2, int(a.top or 10))
    finals = finals_private(st, led.records(), saved)
    print("\n".join(safety.clean(line) for line in report(st, g, finals, top)))
    if a.plot:
        written, why = figures(a.plot, g, st)
        if why:
            print(f"\nfigures not written: {why}")
        else:
            print("\nwrote " + ", ".join(str(p) for p in written))
