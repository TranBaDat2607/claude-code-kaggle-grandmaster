"""Screens: cheap runs of a pipeline that are still comparable with each other (#134).

A screen is a run at reduced *fidelity*: on a frozen subset of the rows or units (``kgkit screen new``), on some
folds, or at cheaper settings (fewer epochs, lower resolution, a shorter context, fewer games). It tells variants apart
for a fraction of a full run, but its score is no estimate of the full pipeline's, so:

* it is compared only with a **control**: the baseline's pipeline run at the same fidelity (``KG_CONTROL_OF``), never
  with a full run (``kgkit ledger compare`` refuses);
* it is never the best CV, the accepted baseline or a final pick (``kgkit.evalset.partial``); a screen that passes is
  *promoted*: run again at the next fidelity, up to the full run that is decided as usual.

The contract a pipeline honours, whatever it trains (the plugin's templates do; a custom one reads the same)::

    KG_SCREEN=<name>        train and evaluate only on the screen's rows / units (``active().mask(ids)``), and only
                            on its folds (``active().folds``) when it names some
    KG_FIDELITY='{"epochs": 3, "img_size": 256}'   the cheaper settings it runs at, recorded on the record as given
    KG_CONTROL_OF=<id>      this run is the control of baseline <id> at this fidelity

``Ledger.log`` reads the three variables and records ``fidelity`` (the screen's name, fingerprint, share and folds,
plus ``KG_FIDELITY``) and ``control_of``. Two runs compare only when their ``fidelity`` is identical.

A subset keeps whole groups (patients, sessions, sites: ``group``), takes the latest window of a time series
(``time``: a screen of the past says little about forecasting the future) or is stratified by the target.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import state as S

DIR = "screens"
ENV, FIDELITY_ENV, CONTROL_ENV = "KG_SCREEN", "KG_FIDELITY", "KG_CONTROL_OF"


def select(ids, frac: float, seed: int = 0, group=None, time=None, strata=None) -> np.ndarray:
    """Which of ``ids`` a screen of share ``frac`` keeps (a boolean mask): the latest window when ``time`` is given
    (whole groups, by their latest row, with ``group``), else whole groups drawn at random, else a share of each
    ``strata`` class, else rows at random. Deterministic for a seed."""
    n = len(ids)
    if not 0 < frac <= 1:
        raise ValueError(f"the share of a screen is in (0, 1], not {frac}")
    if frac >= 1:
        return np.ones(n, bool)
    want = frac * n
    rng = np.random.default_rng(seed)
    if time is not None:
        t = np.asarray(time)
        if group is not None:
            g = np.asarray(group)
            names, inv = np.unique(g, return_inverse=True)
            latest = np.full(len(names), -1)
            np.maximum.at(latest, inv[np.argsort(t, kind="stable")], np.arange(n))  # rank of each group's latest row
            order = np.argsort(latest, kind="stable")[::-1]
            sizes = np.bincount(inv)
            keep_groups = order[: int(np.searchsorted(np.cumsum(sizes[order]), want) + 1)]
            return np.isin(inv, keep_groups)
        cut = np.sort(t, kind="stable")[n - max(1, int(math.ceil(want)))]
        return t >= cut
    if group is not None:
        names, inv = np.unique(np.asarray(group), return_inverse=True)
        order = rng.permutation(len(names))
        sizes = np.bincount(inv)
        cum = np.cumsum(sizes[order])
        keep_groups = order[: int(np.searchsorted(cum, want, side="left") + 1)]
        return np.isin(inv, keep_groups)
    mask = np.zeros(n, bool)
    if strata is not None:
        s = np.asarray(strata)
        for c in np.unique(s):
            rows = np.flatnonzero(s == c)
            mask[rng.choice(rows, max(1, int(round(frac * len(rows)))), replace=False)] = True
        return mask
    mask[rng.choice(n, max(1, int(round(want))), replace=False)] = True
    return mask


def path(root, name: str) -> Path:
    return Path(root) / S.STATE_DIR / DIR / f"{name}.json"


def _digest(ids, folds) -> str:
    blob = "\n".join(sorted(str(i) for i in ids)) + "|folds:" + ",".join(str(f) for f in (folds or ()))
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


@dataclass
class Screen:
    name: str
    frac: float
    folds: list[int] | None
    seed: int
    strategy: str
    n: int
    n_total: int
    hash: str
    ids_file: str
    root: Path = field(default_factory=Path.cwd)
    _ids: set | None = field(default=None, repr=False)

    @property
    def ids(self) -> set:
        if self._ids is None:
            with (self.root / self.ids_file).open(encoding="utf-8", newline="") as f:
                self._ids = {row["id"] for row in csv.DictReader(f)}
        return self._ids

    def mask(self, ids) -> np.ndarray:
        """Which of ``ids`` (the rows' or units' ids, any type) are in the screen."""
        keep = self.ids
        return np.asarray([str(i) in keep for i in np.asarray(ids).tolist()], dtype=bool)

    def fidelity(self) -> dict:
        out = {"screen": self.name, "screen_hash": self.hash, "frac": self.frac}
        if self.folds:
            out["folds"] = list(self.folds)
        return out

    def describe(self) -> str:
        folds = f", folds {' '.join(map(str, self.folds))}" if self.folds else ""
        return (f"screen {self.name}: {self.n} of {self.n_total} ({100 * self.n / max(1, self.n_total):.0f}%), "
                f"{self.strategy}{folds}")


def create(root, name: str, ids, frac: float, folds=None, seed: int = 0, group=None, time=None, strata=None,
           strategy: str | None = None) -> Screen:
    """Draw the screen's subset of ``ids`` (``select``) and freeze it under ``.kaggle-gm/screens/``."""
    if not name or any(c in name for c in "/\\ :"):
        raise ValueError(f"a screen name is one word, not {name!r}")
    ids = np.asarray([str(i) for i in np.asarray(ids).tolist()])
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("the ids of a screen's source are not unique")
    m = select(ids, frac, seed, group=group, time=time, strata=strata)
    chosen = ids[m]
    folds = sorted(int(f) for f in folds) if folds else None
    p = path(root, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    ids_file = p.with_name(f"{name}.ids.csv")
    with ids_file.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id"])
        w.writerows([i] for i in chosen.tolist())
    s = Screen(name, float(frac), folds, int(seed), strategy or "random", int(m.sum()), len(ids),
               _digest(chosen, folds), ids_file.relative_to(Path(root)).as_posix(), Path(root))
    spec = {k: getattr(s, k) for k in ("name", "frac", "folds", "seed", "strategy", "n", "n_total", "hash", "ids_file")}
    p.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    return s


def load(root, name: str) -> Screen:
    """A screen by name (or the path of its .json)."""
    root = Path(root)
    p = Path(name) if name.endswith(".json") else path(root, name)
    if not p.is_file():
        known = sorted(q.stem for q in (root / S.STATE_DIR / DIR).glob("*.json")) if (root / S.STATE_DIR / DIR).is_dir() else []
        raise ValueError(f"no screen {name!r} ({', '.join(known) or 'none yet'}: `kgkit screen new <name> --frac 0.3`)")
    spec = json.loads(p.read_text(encoding="utf-8"))
    return Screen(**{k: spec[k] for k in ("name", "frac", "folds", "seed", "strategy", "n", "n_total", "hash",
                                          "ids_file")}, root=root)


def listed(root) -> list[Screen]:
    d = Path(root) / S.STATE_DIR / DIR
    return [load(root, p.stem) for p in sorted(d.glob("*.json"))] if d.is_dir() else []


def active(root=None) -> Screen | None:
    """The screen this process runs on (``KG_SCREEN``), None for a full run. ValueError for an unknown screen: a run
    told to screen must not silently train on everything."""
    name = os.environ.get(ENV, "").strip()
    return load(Path(root or Path.cwd()), name) if name else None


def settings() -> dict:
    """The cheaper settings of ``KG_FIDELITY`` (a JSON object), {} without it."""
    raw = os.environ.get(FIDELITY_ENV, "").strip()
    if not raw:
        return {}
    try:
        val = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"KG_FIDELITY is not a JSON object ({e}): e.g. KG_FIDELITY='{{\"epochs\": 3}}'")
    if not isinstance(val, dict):
        raise ValueError("KG_FIDELITY is a JSON object of settings, e.g. {\"epochs\": 3}")
    return val


def current_fidelity(root=None) -> dict | None:
    """The fidelity a run of this process is recorded with: the active screen's plus ``KG_FIDELITY``; None for a
    full run."""
    s = active(root)
    out = {**(s.fidelity() if s is not None else {}), **settings()}
    return out or None


def run_tag(root=None) -> str:
    """'' for a full run; else a suffix for the run's files (``artifacts/<name><suffix>``) that keeps a screened run of
    one name apart from its full run and from its runs at other fidelities, which would otherwise resume each
    other's folds."""
    fid = current_fidelity(root)
    if not fid:
        return ""
    digest = hashlib.sha1(json.dumps(fid, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"__{fid.get('screen') or 'fidelity'}-{digest}"


def control_of() -> str | None:
    return os.environ.get(CONTROL_ENV, "").strip() or None


def describe(fidelity: dict | None) -> str:
    """'screen s30 (30%, folds 0), epochs=3' for a record's fidelity; 'full run' without one."""
    if not fidelity:
        return "full run"
    f = dict(fidelity)
    parts = []
    if f.get("screen"):
        inner = [f"{100 * float(f['frac']):.0f}%"] if f.get("frac") is not None else []
        if f.get("folds"):
            inner.append("folds " + " ".join(map(str, f["folds"])))
        parts.append(f"screen {f['screen']}" + (f" ({', '.join(inner)})" if inner else ""))
    parts += [f"{k}={v}" for k, v in sorted(f.items()) if k not in ("screen", "screen_hash", "frac", "folds")]
    return ", ".join(parts)
