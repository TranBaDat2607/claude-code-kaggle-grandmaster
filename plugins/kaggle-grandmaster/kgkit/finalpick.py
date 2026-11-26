"""Which k finals to pick (#142): the set whose *best* private score is highest on average, by simulating the
private leaderboard on the evaluation set.

Kaggle scores a team by the best of its k final submissions, so the second pick is worth what it adds when the
first one does badly. Two near-identical blends rise and fall together and the second adds almost nothing; a weaker
candidate whose errors differ can add more. This module measures that instead of guessing:

1. **Candidates** (``pool``): honest records (``finals._honest``) and the accepted baseline of every model line
   (#133), or the records named, each with predictions on the current evaluation set (OOF in a K-fold workspace, the
   holdout predictions in a ``holdout+lb`` one) and through the vault gate (#141). Fitted, public-tuned, rechecked,
   discarded, seed-replica, screen and pre-vault records are left out, each with the reason.
2. **Simulate** (``simulate``): the candidates' scores on many private-sized samples of the evaluation rows, drawn the
   way the private set differs from them: whole groups (``eval_groups``), contiguous time blocks for a time series
   (``time_blocks``). With an influence function and enough units, the scores are drawn from their joint normal
   (``clt``: covariance from the per-unit contributions); otherwise every candidate is scored with the exact metric
   on the same bootstrap resamples. Never on the vault: its labels stay sealed.
3. **Rank sets** (``rank_sets``): every set of k candidates by E[best of the set], in the metric's own direction,
   with the gain over "the top k by score".

The submitted files are fixed, so seed noise is not part of a private score: only which rows land in the private
set is. The simulation cannot see a shift between the evaluation rows and the private ones (a later period, new
data): it picks the set that does best across sampling luck, nothing more.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import math
from pathlib import Path

import numpy as np

from . import compare as C
from . import evalset as E
from . import finals as FN
from . import metrics as M
from . import state as S
from . import vault as V

SKIP_TASKS = {
    "rl": "agents are rated from episodes against other agents: there are no rows to resample",
    "optimization": "the score is computed on the whole solution, the same on any split: pick the best score",
}
MAX_CANDIDATES = 12  # sets of k out of 12 stay a few thousand at most
MIN_COVERAGE = 0.9   # a candidate scored on fewer rows than this share of the widest one is a subset
TIME_BLOCKS = 50     # contiguous blocks a time series is resampled in
NORMAL_DRAWS = 20_000
ASSUMED_PUBLIC = 0.25  # the public share assumed when the competition does not say (Kaggle's are 20-30% mostly)


# ---------------------------------------------------------------------------- candidates
def _why_not(rec: dict, gate: dict | None, holdout: bool, key: str | None) -> str | None:
    """Why ``rec`` cannot be a candidate, or None."""
    if E.lb_fitted(rec):
        return "public-LB fitted: display only, never a final"
    if E.public_tuned(rec):
        return "public-tuned: chosen on the public LB, not a hedge"
    if rec.get("recheck"):
        return "a recheck run on re-drawn folds"
    if rec.get("decision") == "discard":
        return "discarded"
    if rec.get("replica_of"):
        return f"a seed re-run of {rec['replica_of']}"
    if E.partial(rec):
        return "a screen (reduced fidelity)"
    if rec.get("pre_vault"):
        return "logged before the vault was sealed"
    if not V.cleared(rec, gate):
        return "has not passed the vault (`kgkit vault check`)"
    if rec.get("units"):
        return "scored on a unit table: not simulated yet"
    pred = "holdout_path" if holdout else "oof_path"
    if not rec.get(pred):
        return "no holdout predictions on file" if holdout else "no OOF predictions on file (LB-only, or a stacker " \
                                                                "without nested OOF)"
    if key and E.record_key(rec, holdout) not in (None, key):
        return "scored on another holdout" if holdout else "scored on another fold split"
    return None


def pool(st: S.CompetitionState, recs: list[dict], gate: dict | None, key: str | None,
         ids: list[str] | None = None) -> tuple[list[dict], list[tuple[str, str]]]:
    """``(candidates, excluded)``: the records to simulate and ``(id, reason)`` for each one left out. Without
    ``ids``: the honest candidates and every model line's accepted baseline."""
    by_id = {r["id"]: r for r in recs if isinstance(r, dict) and r.get("id")}
    if ids:
        chosen = list(dict.fromkeys(ids))
    else:
        score = {i: r["lb_public"] for i, r in by_id.items() if isinstance(r.get("lb_public"), (int, float))}
        chosen = [i for i, _ in FN._honest(st, by_id, score, gate)]
        chosen += [r["id"] for r in E.track_heads(list(by_id.values())).values()]
        chosen = list(dict.fromkeys(chosen))
    out, excluded = [], []
    for i in chosen:
        why = _why_not(by_id[i], gate, st.holdout, key)
        (excluded.append((i, why)) if why else out.append(by_id[i]))
    if len(out) > MAX_CANDIDATES:  # keep the strongest by their own score; sets of k out of more add little
        sign = 1 if st.greater_is_better else -1
        ranked = sorted(out, key=lambda r: -sign * r["cv"] if isinstance(r.get("cv"), (int, float)) else math.inf)
        out = ranked[:MAX_CANDIDATES]
        excluded += [(r["id"], f"beyond the {MAX_CANDIDATES} best scores") for r in ranked[MAX_CANDIDATES:]]
    return out, excluded


def shared_rows(preds: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """The rows every candidate predicted, and the candidates dropped first because they cover a subset (fewer
    than ``MIN_COVERAGE`` of the widest one's rows: a model trained on filtered rows, a one-fold OOF)."""
    def covered(p):
        p = np.asarray(p)
        if p.dtype == object:
            return np.array([x is not None for x in p])
        return ~np.isnan(p.reshape(len(p), -1).astype(float)).any(1)

    cover = {i: covered(p) for i, p in preds.items()}
    widest = max(c.sum() for c in cover.values())
    dropped = [i for i, c in cover.items() if c.sum() < MIN_COVERAGE * widest]
    mask = np.ones(len(next(iter(preds.values()))), dtype=bool)
    for i, c in cover.items():
        if i not in dropped:
            mask &= c
    return mask, dropped


def time_blocks(order, n_blocks: int = TIME_BLOCKS) -> np.ndarray:
    """A block id per row: rows sorted by ``order`` (a time column, or the row index) cut into ``n_blocks``
    contiguous blocks, so a resample keeps runs of consecutive time together."""
    order = np.asarray(order)
    n = len(order)
    rank = np.empty(n, dtype=int)
    rank[np.argsort(order, kind="stable")] = np.arange(n)
    return rank * max(1, min(n_blocks, n)) // max(1, n)


# ---------------------------------------------------------------------------- private size
def private_sizes(n_test: int | None, public_n: int | None, n_eval: int,
                  hidden_test: bool = False) -> tuple[int, list[int], str]:
    """``(size, others, note)``: the private LB's rows to simulate at, other plausible sizes to check the pick
    against when it is not known, and what was assumed."""
    if n_test and public_n and 0 < public_n < n_test and not hidden_test:
        return n_test - public_n, [], f"private LB = {n_test:,} test rows - {public_n:,} public"
    if n_test and not hidden_test:
        size = int(round(n_test * (1 - ASSUMED_PUBLIC)))
        return size, [int(round(n_test * 0.5)), n_test], (
            f"public share not set (`kgkit config public_lb_n=<rows>`): assumed {ASSUMED_PUBLIC:.0%} of {n_test:,} "
            "test rows; checked at half and at all of them too")
    what = "the hidden test's size is unknown (code competition)" if hidden_test else "no test size known"
    return n_eval, [max(2, n_eval // 2), 2 * n_eval], (
        f"{what}: simulated at the evaluation set's {n_eval:,} rows, checked at half and at twice that")


# ---------------------------------------------------------------------------- the simulation
def metric_problem(m: M.Metric, p: np.ndarray) -> str | None:
    if m.kind == "label" and p.dtype.kind == "f" and not np.allclose(p, np.round(p), equal_nan=True):
        return f"{m.name} scores labels and these predictions are raw outputs: decode them first"
    return None


def simulate(y, preds: dict[str, np.ndarray], metric: str, *, groups=None, size: int | None = None,
             method: str = "auto", n_draws: int | None = None, seed: int = 0) -> dict:
    """The candidates' scores on private-sized samples of the evaluation rows: ``draws`` (draws x candidates, in
    the metric's units), ``value`` (each one's score on all the rows), ``corr`` (correlation of the drawn scores),
    ``method`` (``clt`` or ``bootstrap``) and ``caveats``. Rows are resampled whole per ``groups``; ``size`` is the
    private LB's row count (the evaluation set's own when None)."""
    m = M.get(metric)
    ids = list(preds)
    num = None if m.kind == "ranking" else float
    y = C._arr(y)
    ps = [C._arr(preds[i], num) for i in ids]
    rows = np.arange(len(y))
    value = np.array([m(y, p) for p in ps], dtype=float)
    members, inv = C._units(rows, groups)
    n_units = len(rows) if members is None else len(members)
    caveats = []
    if n_units < C.SMALL_N:
        caveats.append(f"small evaluation set ({n_units} {'rows' if members is None else 'groups'}): every "
                       "candidate's private score is spread wide, and sets are hard to tell apart")
    rng = np.random.default_rng(seed)
    if method == "auto":
        method = "clt" if m.influence is not None and n_units >= C.CLT_MIN_UNITS else "bootstrap"
    if method == "clt":
        try:
            contrib = np.column_stack([np.asarray(m.influence(y, p)[1], dtype=float) for p in ps])
            per_unit = contrib if inv is None else np.stack([np.bincount(inv, weights=contrib[:, j])
                                                             for j in range(len(ids))], 1)
            per_unit = per_unit - per_unit.mean(0)
            g = len(per_unit)
            cov = g / (g - 1) * per_unit.T @ per_unit
            if size:
                cov *= len(rows) / size
            draws = rng.multivariate_normal(value, cov, size=n_draws or NORMAL_DRAWS, method="eigh",
                                            check_valid="ignore")
            return _out(ids, value, draws, "clt", caveats, n_units, members)
        except (NotImplementedError, ValueError, TypeError, IndexError, np.linalg.LinAlgError) as e:
            caveats.append(f"influence function failed ({e}): simulated by the bootstrap instead")
    n_draws = n_draws or (2000 if len(rows) <= 20_000 else 500)
    m_units = n_units if not size else max(2, int(round(size * n_units / len(rows))))
    req = C._required(y)
    out, failed = [], 0
    for _ in range(n_draws):
        for _try in range(C.MAX_REDRAWS + 1):
            pick = rng.integers(0, n_units, m_units)
            idx = pick if members is None else np.concatenate([members[i] for i in pick])
            if req is None or C._keeps(y[idx], req):
                break
        try:
            s = [m(y[idx], p[idx]) for p in ps]
        except (ValueError, ZeroDivisionError, IndexError):
            failed += 1
            continue
        if all(np.isfinite(s)):
            out.append(s)
        else:
            failed += 1
    if failed > 0.05 * n_draws:
        caveats.append(f"{failed} of {n_draws} resamples could not be scored")
    if not out:
        raise ValueError("no resample could be scored")
    return _out(ids, value, np.asarray(out, dtype=float), "bootstrap", caveats, n_units, members)


def _out(ids, value, draws, method, caveats, n_units, members) -> dict:
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(draws.T) if len(ids) > 1 else np.ones((1, 1))
    return {"ids": ids, "value": value, "draws": draws, "corr": np.atleast_2d(corr), "method": method,
            "caveats": caveats, "n_units": int(n_units), "unit": "rows" if members is None else "groups",
            "se": draws.std(0, ddof=1)}


def rank_sets(sim: dict, k: int, greater: bool) -> dict:
    """Every set of ``k`` candidates by E[best of the set] over the draws, best first, and the set of the top k
    by score on all the rows (``top``, what "the best k by CV" gives) with the gain of each set over it."""
    ids, draws, value = sim["ids"], sim["draws"], sim["value"]
    sign = 1.0 if greater else -1.0
    signed = sign * draws
    k = max(1, min(k, len(ids)))
    top = tuple(sorted(np.argsort(-sign * value, kind="stable")[:k]))
    top_best = signed[:, list(top)].max(1)
    sets = []
    for combo in itertools.combinations(range(len(ids)), k):
        best = signed[:, list(combo)].max(1)
        diff = best - top_best
        sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0  # the private split's noise of the gain
        sets.append({"set": [ids[j] for j in combo], "idx": list(combo), "e_best": float(sign * best.mean()),
                     "gain": float(diff.mean()), "gain_sd": sd, "gain_mc": sd / math.sqrt(len(diff)) if sd else 0.0,
                     "p_top": float((diff > 0).mean()), "corr": _set_corr(sim["corr"], combo)})
    sets.sort(key=lambda s: -s["gain"])
    return {"k": k, "sets": sets, "top": [ids[j] for j in top],
            "top_e_best": float(sign * top_best.mean())}


def _set_corr(corr: np.ndarray, combo) -> float | None:
    pairs = [corr[a, b] for a, b in itertools.combinations(combo, 2)]
    return float(np.nanmax(pairs)) if pairs and np.isfinite(pairs).any() else None


TIE_SHARE = 0.3  # a runner-up that scores higher than the top set in this share of the draws or more is tied with it


def tie_with_top(ranked: dict) -> dict | None:
    """The runner-up when the top k by score rank first but are not clear of it: its gap is within twice the
    simulation's own error (the sign of the expected gain is not known), or it scores higher on the private split in
    ``TIE_SHARE`` of the draws or more (which set wins there is close to a coin flip). A runner-up that never wins
    (it differs from the top set only in a pick that never decides the best) is not tied, however small its gap.
    None when the top set is not first, or is clear of the next."""
    sets = ranked["sets"]
    if len(sets) < 2 or sets[0]["set"] != ranked["top"]:
        return None
    runner = sets[1]
    return runner if -runner["gain"] <= 2 * runner["gain_mc"] or runner["p_top"] >= TIE_SHARE else None


def settled(ranked: dict) -> bool:
    """True when the best set's gain over the top k by score is clear of the simulation's own error."""
    best = ranked["sets"][0]
    return best["set"] == ranked["top"] or best["gain"] > 2 * best["gain_mc"]


# ---------------------------------------------------------------------------- output
def _digits(values) -> int:
    """Decimals that tell ``values`` (scores) apart: 5, or more when the closest two differ by less than 1e-5."""
    xs = sorted({float(v) for v in values if np.isfinite(v)})
    gaps = [b - a for a, b in zip(xs, xs[1:]) if b > a]
    return 5 if not gaps else max(5, min(9, math.ceil(-math.log10(min(gaps))) + 1))


def _f(v: float, digits: int = 5) -> str:
    return f"{v:.{digits}f}"


def lines(sim: dict, ranked: dict, recs: dict, *, note: str, others: dict | None = None, top_n: int = 5,
          excluded=(), groups_note: str = "") -> list[str]:
    """The report of ``kgkit final pick``."""
    k, ids = ranked["k"], sim["ids"]
    out = [f"Final pick: {len(ids)} candidate(s), sets of {k} ranked by the expected best private score "
           f"({sim['method']}, {len(sim['draws']):,} draws over {sim['n_units']:,} {sim['unit']})."]
    out.append(f"  {note}")
    if groups_note:
        out.append(f"  {groups_note}")
    shown = ranked["sets"][:top_n]
    d = _digits([*sim["value"], *(s["e_best"] for s in shown)])
    out.append("Candidates (score on the evaluation rows, spread of the private score):")
    for j, i in enumerate(ids):
        sub = "" if isinstance(recs[i].get("lb_public"), (int, float)) else "  [not submitted]"
        out.append(f"  {i:<34} {_f(sim['value'][j], d)} +/- {E.fmt_se(sim['se'][j])}{sub}")
    for i, why in excluded:
        out.append(f"  left out: {i}: {why}")
    out.append(f"Sets by E[best of {k}]:" if k > 1 else "By expected private score:")
    for s in shown:
        corr = f"  corr {s['corr']:.3f}" if s["corr"] is not None else ""
        mark = "  <- top by score" if s["set"] == ranked["top"] else ""
        out.append(f"  {_f(s['e_best'], d)}  {' + '.join(s['set']):<60} gain {E.fmt_gain(s['gain'])} "
                   f"(+/- {E.fmt_se(s['gain_mc'])}){corr}{mark}")
    best, tie = ranked["sets"][0], tie_with_top(ranked)
    if tie is not None:
        how = ("it scores the same in every draw" if tie["gain_sd"] == 0 else
               f"gap {E.fmt_gain(tie['gain'])}, simulation error {E.fmt_se(tie['gain_mc'])}, private split's noise "
               f"{E.fmt_se(tie['gain_sd'])}; it scores higher in {tie['p_top']:.0%} of the draws")
        out.append(f"Recommended: {' + '.join(best['set'])}, the top {k} by score, tied within error with "
                   f"{' + '.join(tie['set'])} ({how}): a swap gains nothing that can be measured, so keep the "
                   "current selection unless another reason (a different validation, diversity of the models) "
                   "favours the other set.")
    elif best["set"] == ranked["top"]:
        out.append(f"Recommended: {' + '.join(best['set'])}, the top {k} by score: no other set does better here.")
    elif settled(ranked):
        out.append(f"Recommended: {' + '.join(best['set'])}: E[best] {_f(best['e_best'], d)} against "
                   f"{_f(ranked['top_e_best'], d)} for the top {k} by score ({' + '.join(ranked['top'])}); better "
                   f"in {best['p_top']:.0%} of the draws.")
    else:
        out.append(f"Recommended: {' + '.join(best['set'])}, but within the simulation's error of the top {k} by "
                   f"score ({' + '.join(ranked['top'])}): either is fine.")
    for size, alt in (others or {}).items():
        same = alt["sets"][0]["set"] == best["set"]
        out.append(f"  at {size:,} private rows: " + ("the same set" if same else
                                                      f"{' + '.join(alt['sets'][0]['set'])} instead"))
    for c in sim["caveats"]:
        out.append(f"  note: {c}")
    unsub = [i for i in best["set"] if not isinstance(recs[i].get("lb_public"), (int, float))]
    if unsub:
        out.append(f"  not submitted yet: {', '.join(unsub)}; a final must be submitted before it can be selected.")
    out.append("It cannot see a shift between the evaluation rows and the private test (a later period, new data). "
               "Select the set on the website, then run `kgkit final` to confirm.")
    return out


def save(root, st: S.CompetitionState, ranked: dict, sim: dict, size: int | None) -> Path:
    """Record the recommendation for the outcome report after the deadline (``finals.load_pick``)."""
    best = ranked["sets"][0]
    p = Path(root) / S.STATE_DIR / FN.PICK_FILE
    p.write_text(json.dumps({
        "slug": st.slug, "when": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "k": ranked["k"], "ids": best["set"], "e_best": best["e_best"], "top": ranked["top"],
        "top_e_best": ranked["top_e_best"], "method": sim["method"], "size": size, "candidates": sim["ids"],
    }, indent=1), encoding="utf-8")
    return p
