"""A sealed vault split (#141): held-out rows whose labels the agent never sees, checked pass/fail only.

Every keep/discard decision, screen ranking and final pick is taken on the same frozen folds, so after many decisions
the accepted baseline's CV is optimistic: part of its gain is selection luck on *those* folds. ``kgkit recheck``
re-draws the folds but reuses the rows, and the agent sees every score it prints. The vault is about 10% of the
training rows, drawn the way the folds are (whole groups, the latest rows by time, else stratified), whose labels
leave the workspace:

* ``data/train.csv`` loses the vault rows (the labels and the rows as they were go to ``<home>/<key>/rows.csv``,
  outside the workspace and git; ``<home>`` is ``~/.kaggle-gm/vault``, the PreToolUse hook denies reading it);
* ``data/vault.csv`` holds their feature columns without the target: the training templates predict it like a second
  test set and log ``vault_pred`` (``artifacts/<id>/vault.npy``);
* ``data/folds.csv`` lists the vault ids too, with ``fold = VAULT`` (-2): a remote Kaggle run sees the original
  ``train.csv`` (labels included), and the templates drop those rows before anything else.

``kgkit vault check <id>`` answers only PASS or FAIL and how many checks are left (Thresholdout-style, Dwork et al.
2015): never a score, so the agent cannot climb a number it never sees. A check tests the *total* gain of the
pipeline (from the root of its lineage, or ``--against``) and that it is not worse than the accepted baseline: a vault
of 10% of the rows has ~3x the SE of the full OOF, too noisy for one +0.001 step. Each workspace has a budget of
checks (default 20); asking the same question again is free. Once the finals are picked, ``kgkit vault open`` puts
the rows back (as ``fold = -1``: trained on by every fold, scored by none), so the shipped model loses no data.

The status helpers (``meta``, ``status_line``, ``passed``, ``sealed``) are standard library only: the session hook
and ``kgkit.finals`` import this module. numpy, pandas and the metrics are imported inside the functions that need
them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from . import state as S

VAULT = -2                 # the fold of a vault row in the split file; -1 stays "trained on, never validated"
META = "vault.json"        # .kaggle-gm/vault.json: what was sealed (no labels)
FEATURES = "data/vault.csv"
ROWS = "rows.csv"          # <home>/<key>/rows.csv: the vault rows as they were, labels included
CHECKS = "checks.jsonl"    # <home>/<key>/checks.jsonl: one line per check spent
POS = "__kg_pos"           # rows.csv: the row's position in the original file, so `open` restores the order
GROUP = "__kg_group"       # rows.csv (holdout+lb): the row's group, from eval_groups
DEFAULT_FRAC = 0.10
DEFAULT_BUDGET = 20
DEFAULT_SEED = 141
SMALL_UNITS = 2000         # fewer units (rows, or groups) than this: measure the vault's noise before sealing
MIN_RARE = 30              # a vault with fewer rows of the rarest class than this is measured first too
FRACS = (0.10, 0.15, 0.20, 0.25, 0.30)
HOME_ENV = "KG_VAULT_HOME"  # where sealed labels live (tests); default ~/.kaggle-gm/vault


# The folder of sealed labels, as a command or a tool call names it: ~/.kaggle-gm/vault, C:\Users\x\.kaggle-gm\vault\..
# Not the workspace's .kaggle-gm/vault.json, which holds no label.
VAULT_PATH_RE = re.compile(r"\.kaggle-gm[\\/]+vault(?![\w.-])", re.IGNORECASE)


def names_vault(text: str | None) -> bool:
    """True when ``text`` (a path, a glob, a shell command) names the folder of sealed labels: the PreToolUse hooks
    deny reading it. Never raises."""
    try:
        if not text:
            return False
        if VAULT_PATH_RE.search(text):
            return True
        h = os.environ.get(HOME_ENV, "").strip()
        if h:
            def norm(s: str) -> str:
                return s.replace("\\", "/").rstrip("/").lower()
            return norm(h) in norm(text)
    except Exception:  # noqa: BLE001 - a hook calls this
        return False
    return False


DENY_REASON = ("the sealed vault (~/.kaggle-gm/vault/) holds labels the agent must never see: reading them would let "
               "the final picks climb the one score that is still honest. `kgkit vault check <id>` is the only use of "
               "them (it answers PASS or FAIL); `kgkit vault status` shows the checks left.")


# ---------------------------------------------------------------------------- where things are (stdlib)
def home() -> Path:
    v = os.environ.get(HOME_ENV, "").strip()
    return Path(v).expanduser() if v else Path.home() / ".kaggle-gm" / "vault"


def meta_path(root: str | Path) -> Path:
    return Path(root) / S.STATE_DIR / META


def meta(root: str | Path | None) -> dict | None:
    """``.kaggle-gm/vault.json``, or None. Never raises."""
    if root is None:
        return None
    try:
        m = json.loads(meta_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return m if isinstance(m, dict) else None


def _save_meta(root: str | Path, m: dict) -> None:
    p = meta_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=1), encoding="utf-8")


def sealed(root: str | Path | None) -> bool:
    m = meta(root)
    return bool(m) and m.get("state") == "sealed"


def store(m: dict) -> Path:
    """The folder outside the workspace that holds the vault's labels."""
    return home() / m["key"]


def _key(root: str | Path, slug: str) -> str:
    """``<slug>-<8 hex of the workspace path>``: two workspaces of one competition never share a vault."""
    h = hashlib.sha1(str(Path(root).resolve()).lower().encode()).hexdigest()[:8]
    return f"{re.sub(r'[^A-Za-z0-9_.-]+', '-', slug or 'competition')}-{h}"


def checks(m: dict | None) -> list[dict]:
    if not m:
        return []
    try:
        lines = (store(m) / CHECKS).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            c = json.loads(line)
        except ValueError:
            continue
        if isinstance(c, dict):
            out.append(c)
    return out


def left(m: dict | None) -> int:
    if not m:
        return 0
    return max(0, int(m.get("budget") or DEFAULT_BUDGET) - len(checks(m)))


def passed(rec: dict) -> bool:
    """True for a record that passed a vault check."""
    return isinstance(rec, dict) and rec.get("vault") == "pass"


def overrides(root: str | Path | None) -> dict:
    """{record id: reason} for final picks accepted without a vault pass (`kgkit final --accept-unvaulted`)."""
    m = meta(root) or {}
    o = m.get("overrides")
    return o if isinstance(o, dict) else {}


def gate(m: dict | None) -> dict | None:
    """What the final-pick gate needs (``kgkit.finals``): None without a sealed vault, else ``{"overrides"}``."""
    if not m or m.get("state") != "sealed":
        return None
    o = m.get("overrides")
    return {"overrides": o if isinstance(o, dict) else {}}


def cleared(rec: dict, g: dict | None) -> bool:
    """True when ``rec`` may be a final pick under the gate ``g``: no vault, a pass, or an accepted exception."""
    return g is None or passed(rec) or (isinstance(rec, dict) and rec.get("id") in g["overrides"])


def accept_unvaulted(root: str | Path, exp_id: str, why: str) -> dict:
    """Record a final pick taken without a vault pass (the checks ran out, a deadline): in ``vault.json`` and in
    ``reports/final-selection.md``, so the exception is on file. Returns the metadata."""
    m = meta(root)
    if not m or m.get("state") != "sealed":
        raise SystemExit("no sealed vault here: nothing to override")
    if not why.strip():
        raise SystemExit("--why is required: say why this pick ships without a vault pass")
    m.setdefault("overrides", {})[exp_id] = why.strip()
    _save_meta(root, m)
    rep = Path(root) / "reports" / "final-selection.md"
    rep.parent.mkdir(parents=True, exist_ok=True)
    with rep.open("a", encoding="utf-8") as f:
        f.write(f"\n- {time.strftime('%Y-%m-%d %H:%M')}: {exp_id} accepted as a final pick WITHOUT a vault pass: "
                f"{why.strip()}\n")
    return m


def status_line(root: str | Path | None, recs: list[dict] | None = None) -> str | None:
    """The vault line of `kgkit status` and the session brief, or None. Never raises."""
    try:
        m = meta(root)
        if not m:
            return None
        if m.get("state") == "opened":
            return (f"vault: opened {m.get('opened_at', '')} (its rows train every fold now; no more checks). Retrain "
                    "the final picks so they ship on all rows")
        if m.get("state") != "sealed":
            return None
        unit = m.get("unit") or "rows"
        ok = [r["id"] for r in recs or () if isinstance(r, dict) and passed(r)]
        n_left = left(m)
        return (f"vault: sealed {float(m.get('frac') or 0):.0%} ({m.get('n_units')} {unit}), {n_left} of "
                f"{m.get('budget')} checks left; " + (f"passed: {', '.join(ok)}" if ok else "nothing passed yet")
                + ". A final candidate goes CV gate -> recheck -> `kgkit vault check <id>`"
                + (" (NO CHECKS LEFT: final picks need `kgkit final --accept-unvaulted`)" if n_left == 0 else ""))
    except Exception:  # noqa: BLE001 - a hook calls this
        return None


def vault_hash(ids) -> str:
    """Fingerprint of the vault: its ids, sorted (no labels)."""
    h = hashlib.sha1()
    for i in sorted(str(x) for x in ids):
        h.update(i.encode() + b"\x1e")
    return h.hexdigest()[:12]


def current_hash(root: str | Path | None) -> str | None:
    m = meta(root)
    return m.get("vault_hash") if m and m.get("state") == "sealed" else None


# ---------------------------------------------------------------------------- drawing (pandas)
def _read(path: Path):
    import pandas as pd

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".feather":
        return pd.read_feather(path)
    return pd.read_csv(path)


def _write(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".parquet":
        df.to_parquet(tmp, index=False)
    elif path.suffix == ".feather":
        df.reset_index(drop=True).to_feather(tmp)
    else:
        df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _targets(target) -> list[str]:
    if not target:
        return []
    return [t.strip() for t in target.split(",") if t.strip()] if isinstance(target, str) else list(target)


def draw(df, frac: float = DEFAULT_FRAC, target=None, group: str | None = None, time_col: str | None = None,
         seed: int = DEFAULT_SEED, strategy: str = "auto"):
    """Boolean mask of the vault rows of ``df``: about ``frac`` of them, drawn the way the folds are. With
    ``time_col``: the rows of the latest ``frac`` of the time values. Else the first of ``round(1/frac)`` folds of
    ``kgkit.cv.assign_folds``: whole groups, stratified, multilabel alike."""
    import numpy as np

    from .cv import assign_folds

    if not 0 < frac < 0.5:
        raise ValueError(f"the vault fraction must be between 0 and 0.5, got {frac}")
    if time_col:
        t = df[time_col]
        uniq = np.sort(t.dropna().unique())
        k = max(1, int(round(frac * len(uniq))))
        return (t >= uniq[-k]).to_numpy()
    tg = _targets(target)
    tg = tg if len(tg) > 1 else (tg[0] if tg else None)
    if strategy in ("group", "stratified_group") and not group:
        strategy = "auto"
    folds = assign_folds(df, max(2, int(round(1 / frac))), strategy, target=tg, group=group, seed=seed)
    return folds == 0


def units(df, mask, group: str | None) -> int:
    return int(df.loc[mask, group].nunique()) if group else int(mask.sum())


def small_reason(df, frac: float, target=None, group: str | None = None) -> str | None:
    """Why a vault of ``frac`` may be too noisy to judge anything, or None."""
    n = int(df[group].nunique()) if group else len(df)
    what = f"{n} {'groups' if group else 'rows'}"
    if n < SMALL_UNITS:
        return f"{what} (under {SMALL_UNITS}): a {frac:.0%} vault has about {int(n * frac)}"
    tg = _targets(target)
    if len(tg) == 1 and tg[0] in df.columns:
        from .cv import _is_classification_target

        y = df[tg[0]]
        if _is_classification_target(y):
            rare = int(y.value_counts().min())
            if rare * frac < MIN_RARE:
                return (f"the rarest class of {tg[0]!r} has {rare} rows: a {frac:.0%} vault has about "
                        f"{int(rare * frac)} (under {MIN_RARE})")
    return None


# ---------------------------------------------------------------------------- sealing
def _rel(root: Path, p: str | Path) -> str:
    p = Path(p)
    try:
        return (p if p.is_absolute() else (Path.cwd() / p)).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(p)


def _records_in_play(root: Path) -> list[dict]:
    from .experiment import Ledger

    return [r for r in Ledger(root).records() if not r.get("pre_vault")]


def _mark_pre_vault(root: Path) -> int:
    """Records logged before the seal scored other rows: never a baseline, the best CV or a final pick again."""
    from .experiment import Ledger

    led = Ledger(root)
    return led.mark([r["id"] for r in led.records() if not r.get("pre_vault")], pre_vault=True)


def _zip_note(root: Path) -> str:
    zips = sorted(p.name for p in (root / "data").glob("*.zip")) if (root / "data").is_dir() else []
    return (f"note: {', '.join(zips)} in data/ still hold every label; delete the archive (it can be downloaded "
            "again) or never extract it over data/train.csv" if zips else "")


def write_folds(root: Path, df, spec: dict, out: str | Path, vault_ids=None):
    """Assign folds to ``df`` with the recipe ``spec`` and write ``out``: the id column and ``fold``, then the vault
    ids with ``fold = VAULT``. Returns the folds of ``df``'s rows."""
    import numpy as np
    import pandas as pd

    from .cv import assign_folds

    tg = _targets(spec.get("target"))
    tg = tg if len(tg) > 1 else (tg[0] if tg else None)
    folds = assign_folds(df, int(spec.get("n_splits") or 5), spec.get("strategy") or "auto", target=tg,
                         group=spec.get("group"), seed=int(spec.get("seed") or 42))
    id_col = spec.get("id_col")
    if id_col and id_col in df.columns:
        frame = pd.DataFrame({id_col: df[id_col].to_numpy(), "fold": folds})
        if vault_ids is not None and len(vault_ids):
            frame = pd.concat([frame, pd.DataFrame({id_col: np.asarray(vault_ids), "fold": VAULT})],
                              ignore_index=True)
    else:
        frame = pd.DataFrame({"fold": folds})
    out = Path(out)
    out = out if out.is_absolute() else root / out
    _write(frame, out)
    return folds


def seal(root: str | Path, frac: float = DEFAULT_FRAC, *, train: str = "data/train.csv", target=None,
         group: str | None = None, time_col: str | None = None, id_col: str | None = None, seed: int = DEFAULT_SEED,
         budget: int = DEFAULT_BUDGET, reset: bool = False, spec: dict | None = None,
         folds_out: str = "data/folds.csv") -> dict:
    """Seal a vault in a K-fold workspace: move about ``frac`` of the rows of ``train`` out (see the module
    docstring), re-draw the folds on the rest with the recipe ``spec`` (default ``.kaggle-gm/folds_spec.json``)
    and write ``.kaggle-gm/vault.json``. Refuses when records are already logged unless ``reset``: their scores are
    on other rows. Returns the metadata."""
    root = Path(root)
    st = S.load(root)
    if sealed(root):
        raise SystemExit("a vault is already sealed here (`kgkit vault status`); `kgkit vault open` ends it")
    if st is not None and st.holdout:
        return seal_holdout(root, st, frac, seed=seed, budget=budget, reset=reset)
    if _records_in_play(root) and not reset:
        raise SystemExit("records are already logged on the current folds: sealing re-draws the split, so none of "
                         "them can be compared with what comes after. Pass --reset-comparisons to seal anyway (they "
                         "stay in the ledger, marked pre_vault: never a baseline, the best CV or a final pick)")
    if spec is None:
        p = root / S.STATE_DIR / "folds_spec.json"
        spec = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    spec = dict(spec)
    id_col = id_col or spec.get("id_col") or (st.id_col if st else "") or None
    target = target or spec.get("target") or (st.target if st else "") or None
    group = group if group is not None else spec.get("group")
    train_path = Path(train) if Path(train).is_absolute() else root / _rel(root, train)
    df = _read(train_path)
    if not id_col or id_col not in df.columns:
        raise SystemExit(f"the vault needs an id column ({id_col or 'none set'} is not in {train_path.name}): the "
                         "split file names the vault rows by id, so a remote run on the original data can drop them. "
                         "Pass --id-col")
    if df[id_col].duplicated().any():
        raise SystemExit(f"{id_col!r} repeats in {train_path.name}: the vault needs one row per id")
    tg = _targets(target)
    missing = [t for t in tg if t not in df.columns]
    if not tg or missing:
        raise SystemExit(f"target column(s) {missing or '(none)'} not in {train_path.name}: pass --target")
    mask = draw(df, frac, target, group, time_col, seed, spec.get("strategy") or "auto")
    if mask.all() or not mask.any():
        raise SystemExit("the vault draw took no rows or every row: check --vault and the data")
    key = _key(root, st.slug if st else root.name)
    m = {"state": "sealed", "mode": "kfold", "key": key, "frac": frac, "seed": seed, "budget": int(budget),
         "train": _rel(root, train_path), "target": ",".join(tg), "group": group, "time_col": time_col,
         "id_col": id_col, "n_rows": int(mask.sum()), "n_units": units(df, mask, group),
         "unit": "groups" if group else ("rows by time" if time_col else "rows"),
         "vault_hash": vault_hash(df.loc[mask, id_col]), "features": FEATURES, "folds": _rel(root, folds_out),
         "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "overrides": {}}
    d = home() / key
    rows = df.loc[mask].copy()
    rows.insert(0, POS, rows.index.to_numpy())
    _write(rows, d / ROWS)  # the labels leave first: nothing below may run with them only in the workspace
    (d / CHECKS).touch()
    _write(df.loc[mask].drop(columns=tg), root / FEATURES)
    rest = df.loc[~mask].reset_index(drop=True)
    _write(rest, train_path)
    spec.update({k: v for k, v in (("train", m["train"]), ("target", m["target"]), ("id_col", id_col),
                                   ("out", m["folds"])) if v})
    spec.setdefault("n_splits", 5)
    spec.setdefault("strategy", "auto")
    spec.setdefault("seed", 42)
    spec["group"] = group
    spec["vault"] = {"frac": frac, "seed": seed, "time_col": time_col}
    write_folds(root, rest, spec, folds_out, vault_ids=df.loc[mask, id_col].to_numpy())
    (root / S.STATE_DIR).mkdir(parents=True, exist_ok=True)
    (root / S.STATE_DIR / "folds_spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    n_pre = _mark_pre_vault(root) if reset else 0
    m["pre_vault_records"] = n_pre
    _save_meta(root, m)
    return m


def seal_holdout(root: Path, st, frac: float, *, seed: int = DEFAULT_SEED, budget: int = DEFAULT_BUDGET,
                 reset: bool = False) -> dict:
    """holdout+lb: carve the vault from the labelled holdout (``eval_truth``), grouped by ``eval_groups``. The truth
    file keeps the rest (its fingerprint changes: earlier holdout scores are not comparable), ``data/vault.csv`` lists
    the vault ids, and pipelines log their predictions for them with ``kgkit ledger add ... --vault-pred``."""
    from . import evalset as E

    if not st.eval_truth:
        raise SystemExit("holdout+lb: set the holdout labels first (`kgkit config eval_truth=FILE:COL`)")
    if _records_in_play(root) and not reset:
        raise SystemExit("records are already scored on this holdout: carving the vault out of it changes it, so "
                         "none of them can be compared with what comes after. Pass --reset-comparisons to seal "
                         "anyway (they stay in the ledger, marked pre_vault)")
    path, cols = E.split_spec(st.eval_truth)
    truth_path = root / path if not Path(path).is_absolute() else Path(path)
    df = _read(truth_path)
    id_col = st.id_col
    if not id_col or id_col not in df.columns:
        raise SystemExit(f"the vault needs an id column in {truth_path.name} (`kgkit config id_col=...`): pipelines "
                         "predict the vault rows by id")
    gcol = None
    if st.eval_groups:
        gpath, gcols = E.split_spec(st.eval_groups)
        gdf = _read(root / gpath if not Path(gpath).is_absolute() else Path(gpath))
        if len(gdf) != len(df):
            raise SystemExit(f"eval_groups ({st.eval_groups}) has {len(gdf)} rows, the holdout {len(df)}")
        df[GROUP] = gdf[gcols[0]].to_numpy()
        gcol = GROUP
    mask = draw(df, frac, cols if len(cols) > 1 else cols[0], gcol, None, seed)
    if mask.all() or not mask.any():
        raise SystemExit("the vault draw took no rows or every row of the holdout: check --frac")
    key = _key(root, st.slug)
    d = home() / key
    rows = df.loc[mask].copy()
    rows.insert(0, POS, rows.index.to_numpy())
    _write(rows, d / ROWS)
    (d / CHECKS).touch()
    keep = [c for c in df.columns if c not in cols and c != GROUP]
    _write(df.loc[mask, keep], root / FEATURES)
    _write(df.loc[~mask].drop(columns=[GROUP], errors="ignore").reset_index(drop=True), truth_path)
    if gcol is not None:  # the groups file keeps the holdout's rows, in its order
        gpath, gcols = E.split_spec(st.eval_groups)
        gp = root / gpath if not Path(gpath).is_absolute() else Path(gpath)
        if gp.resolve() != truth_path.resolve():
            _write(_read(gp).loc[~mask].reset_index(drop=True), gp)
    m = {"state": "sealed", "mode": "holdout", "key": key, "frac": frac, "seed": seed, "budget": int(budget),
         "truth": st.eval_truth, "groups": st.eval_groups or None, "target": ",".join(cols), "id_col": id_col,
         "group": gcol, "n_rows": int(mask.sum()), "n_units": units(df, mask, gcol),
         "unit": "groups" if gcol else "rows", "vault_hash": vault_hash(df.loc[mask, id_col]), "features": FEATURES,
         "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "overrides": {},
         "pre_vault_records": _mark_pre_vault(root) if reset else 0}
    _save_meta(root, m)
    return m


# ---------------------------------------------------------------------------- measuring before sealing
def size(root: str | Path, exp: str, vs: str, fracs=FRACS, seed: int = DEFAULT_SEED, draws: int = 5) -> dict:
    """How noisy a vault of each fraction would be, measured on two quick records' OOF over the whole (unsealed)
    training set: the paired SE of ``exp - vs`` on vault-sized draws (median of ``draws``), and the smallest total
    gain such a vault confirms 80% of the time (``(Z_KEEP + 0.84) x SE``). A fraction is OK when it confirms half
    the gap between the two quick models. Marks both records ``pre_vault``."""
    import numpy as np

    from .compare import Z_KEEP, estimate
    from .experiment import Ledger

    root = Path(root)
    if sealed(root):
        raise SystemExit("a vault is already sealed: `kgkit vault size` measures before sealing")
    st = S.load(root)
    if st is None:
        raise SystemExit("not inside a competition workspace")
    p = root / S.STATE_DIR / "folds_spec.json"
    spec = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    led = Ledger(root)
    a, b = led.get(exp), led.get(vs)
    if not a.get("oof_path") or not b.get("oof_path"):
        raise SystemExit("both records need OOF predictions on file (the training templates store them)")
    train = _read(root / (spec.get("train") or "data/train.csv"))
    target = spec.get("target") or st.target
    tg = _targets(target)
    from .cli import _truth

    y = _truth(f"{root / (spec.get('train') or 'data/train.csv')}:{','.join(tg)}")
    pa, pb = led.load_oof(a["id"]), led.load_oof(b["id"])
    if len(pa) != len(y) or len(pb) != len(y):
        raise SystemExit(f"the OOF has {len(pa)} rows and the target {len(y)}: measure on the whole training set, "
                         "before any vault is sealed")
    group = spec.get("group")
    groups = train[group].to_numpy() if group else None
    full = estimate(y, pa, pb, st.metric, greater_is_better=st.greater_is_better, groups=groups)
    gap = abs(full["gain"])
    rows = []
    for f in fracs:
        ses, n_units = [], []
        for k in range(draws):
            mask = draw(train, f, target, group, (spec.get("vault") or {}).get("time_col"), seed + k,
                        spec.get("strategy") or "auto")
            r = estimate(y, pa, pb, st.metric, greater_is_better=st.greater_is_better, groups=groups, mask=mask)
            if r["se"] == r["se"]:
                ses.append(r["se"])
            n_units.append(units(train, mask, group))
        se = float(np.median(ses)) if ses else float("nan")
        mde = (Z_KEEP + 0.84) * se
        rows.append({"frac": f, "units": int(np.median(n_units)), "se": se, "mde": mde,
                     "ok": bool(se == se and mde <= gap / 2)})
    led.mark([a["id"], b["id"]], pre_vault=True)
    return {"exp": a["id"], "vs": b["id"], "gap": full["gain"], "gap_se": full["se"], "unit": "groups" if group
            else "rows", "rows": rows, "smallest_ok": next((r["frac"] for r in rows if r["ok"]), None)}


def format_size(res: dict) -> str:
    from .evalset import fmt_gain, fmt_se

    lines = [f"# Vault size: noise measured on {res['exp']} vs {res['vs']}",
             f"the two quick models differ by {fmt_gain(res['gap'])} on the whole OOF (SE {fmt_se(res['gap_se'])})",
             "", f"| vault | {res['unit']} | paired SE | smallest total gain it confirms | verdict |",
             "|---|---|---|---|---|"]
    for r in res["rows"]:
        lines.append(f"| {r['frac']:.0%} | {r['units']} | {fmt_se(r['se'])} | {fmt_se(r['mde'])} | "
                     f"{'OK' if r['ok'] else 'TOO NOISY'} |")
    ok = res["smallest_ok"]
    lines += ["", "OK = it confirms half the gap between the two quick models (80% of the time)."]
    if ok is None:
        lines.append(f"VERDICT: every fraction up to {res['rows'][-1]['frac']:.0%} is TOO NOISY. Ask the user: seal a "
                     "larger vault anyway, or none (then trust the CV gate and `kgkit recheck` with several seeds).")
    elif ok > res["rows"][0]["frac"]:
        lines.append(f"VERDICT: {res['rows'][0]['frac']:.0%} is TOO NOISY; {ok:.0%} is the smallest that is OK. Ask "
                     f"the user: `kgkit vault seal --frac {ok:g}`, or no vault.")
    else:
        lines.append(f"VERDICT: {ok:.0%} is OK: `kgkit vault seal --frac {ok:g}`.")
    lines.append("Both records are marked pre_vault: sealing re-draws the split, and they scored the vault's rows.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------- checking
def _labels(root: Path, m: dict):
    """(labels as the templates encode them, groups or None) of the vault rows, in data/vault.csv's order."""
    import numpy as np

    rows = _read(store(m) / ROWS)
    tg = _targets(m["target"])
    if len(tg) > 1:
        y = rows[tg].to_numpy()
    else:
        import pandas as pd

        raw = rows[tg[0]]
        if pd.api.types.is_numeric_dtype(raw) and not pd.api.types.is_bool_dtype(raw):
            y = raw.to_numpy()
        else:  # string labels: the codes of the classes the visible rows have, as the templates encode them
            src = root / (m.get("train") or "data/train.csv") if m.get("mode") == "kfold" else None
            seen = _read(src)[tg[0]] if src is not None and src.is_file() else raw
            classes = np.unique(np.concatenate([seen.astype(str).to_numpy(), raw.astype(str).to_numpy()]))
            y = np.searchsorted(classes, raw.astype(str).to_numpy())
    g = m.get("group")
    groups = rows[g].to_numpy() if g and g in rows.columns else None
    return y, groups


def _decode(rec: dict, p, metric):
    """Predictions as the metric reads them: a label metric gets the record's threshold(s), or the argmax."""
    import numpy as np

    p = np.asarray(p)
    if metric.kind != "label":
        return p
    params = rec.get("params") or {}
    if p.ndim == 2 and p.shape[1] > 1:
        return p.argmax(1)
    p = p.reshape(len(p))
    if isinstance(params.get("threshold"), (int, float)):
        return (p >= params["threshold"]).astype(int)
    if isinstance(params.get("thresholds"), list) and params["thresholds"]:
        return np.digitize(p, np.asarray(params["thresholds"], dtype=float))
    return p


def _vault_pred(led, rec: dict, m: dict, n: int):
    import numpy as np

    if not rec.get("vault_path"):
        raise SystemExit(f"{rec['id']} has no vault predictions: run it with a template of this kgkit (it predicts "
                         f"data/vault.csv), or log them with `kgkit ledger add ... --vault-pred FILE`")
    if rec.get("vault_hash") != m["vault_hash"]:
        raise SystemExit(f"{rec['id']} predicted another vault (vault_hash {rec.get('vault_hash')}, now "
                         f"{m['vault_hash']}): re-run it")
    p = np.load(led.root / rec["vault_path"])
    if len(p) != n:
        raise SystemExit(f"{rec['id']}'s vault predictions have {len(p)} rows, the vault {n}")
    return p


def default_against(led, rec: dict, m: dict) -> dict | None:
    """The oldest ancestor of ``rec`` that predicted this vault: the start of the gain the check tests."""
    for r in led.lineage(rec["id"]):
        if r["id"] != rec["id"] and r.get("vault_path") and r.get("vault_hash") == m["vault_hash"]:
            return r
    return None


def judge(y, cand, against, baseline=None, metric: str = "auc", greater_is_better: bool | None = None,
          groups=None, seed: int = 0) -> bool:
    """The vault's rule on predictions of its rows: ``cand`` beats ``against`` with z >= Z_KEEP (paired, whole groups
    resampled) and, when ``baseline`` is given, is not worse than it with z <= -Z_KEEP. No spread (identical
    predictions on every resample) leaves the sign of the gain to decide."""
    from .compare import Z_KEEP, estimate

    def z(pa, pb) -> float:
        r = estimate(y, pa, pb, metric, greater_is_better=greater_is_better, groups=groups, seed=seed)
        v = r.get("z")
        if v != v:
            return float("inf") if r["gain"] > 0 else (float("-inf") if r["gain"] < 0 else 0.0)
        return float(v)

    if z(cand, against) < Z_KEEP:
        return False
    return baseline is None or z(cand, baseline) > -Z_KEEP


def check(root: str | Path, exp: str, against: str | None = None) -> dict:
    """Spend one check: PASS when ``exp`` (1) beats ``against`` (default: the root of its lineage) by a significant
    margin on the vault (paired, whole groups resampled, z >= Z_KEEP) and (2) is not significantly worse than the
    accepted baseline there. Returns ``{"result", "left", "cached", "against", "baseline"}``: no score."""
    from . import metrics as M
    from .experiment import Ledger

    root = Path(root)
    m = meta(root)
    if not m or m.get("state") != "sealed":
        raise SystemExit("no sealed vault here (`kgkit vault status`)")
    st = S.load(root)
    led = Ledger(root)
    rec = led.get(exp)
    ref = led.get(against) if against else default_against(led, rec, m)
    if ref is None:
        raise SystemExit(f"{rec['id']} has no ancestor that predicted the vault: name the start of its gain with "
                         "--against <id> (the first baseline run after the seal)")
    if ref["id"] == rec["id"]:
        raise SystemExit("--against must be another record")
    base = led.baseline(fallback=False)
    base_id = base["id"] if base is not None and base["id"] not in (rec["id"], ref["id"]) else None
    question = {"exp": rec["id"], "against": ref["id"], "baseline": base_id}
    for c in checks(m):
        if all(c.get(k) == v for k, v in question.items()):
            return {**question, "result": c["result"], "left": left(m), "cached": True}
    n_left = left(m)
    if n_left <= 0:
        raise SystemExit(f"no vault checks left (budget {m.get('budget')}): the vault cannot judge more candidates. "
                         "A final pick without a pass needs `kgkit final --accept-unvaulted <id> --why \"...\"`")
    y, groups = _labels(root, m)
    metric = M.get(st.metric)
    pc = _decode(rec, _vault_pred(led, rec, m, len(y)), metric)
    pr = _decode(ref, _vault_pred(led, ref, m, len(y)), metric)
    pb = _decode(base, _vault_pred(led, base, m, len(y)), metric) if base_id is not None else None
    ok = judge(y, pc, pr, pb, metric.name, st.greater_is_better, groups)
    result = "PASS" if ok else "FAIL"
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **question, "result": result}
    d = store(m)
    d.mkdir(parents=True, exist_ok=True)
    with (d / CHECKS).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    led.update(rec["id"], vault="pass" if ok else "fail", vault_against=ref["id"], vault_checked=entry["time"])
    return {**question, "result": result, "left": n_left - 1, "cached": False}


def format_check(res: dict) -> str:
    base = f", not worse than the baseline {res['baseline']}" if res.get("baseline") else ""
    again = " (asked before: no check spent)" if res["cached"] else ""
    what = (f"{res['exp']} vs {res['against']}{base}: {res['result']}{again}. {res['left']} checks left.")
    if res["result"] == "PASS":
        return f"vault check {what} It can be a final candidate."
    return (f"vault check {what} The total gain is not confirmed on rows the pipeline never saw: do not pick it. "
            "Re-test the marginal KEEPs of its lineage (`kgkit ledger lineage`) before asking again.")


# ---------------------------------------------------------------------------- opening
def open_vault(root: str | Path) -> dict:
    """End the vault: put its rows back (K-fold: in ``train.csv`` at their old positions, ``fold = -1`` in the
    split, so every fold trains on them and none scores them; holdout+lb: in the truth file), remove
    ``data/vault.csv`` and stop the checks. The final picks are then retrained on all rows."""
    import numpy as np
    import pandas as pd

    root = Path(root)
    m = meta(root)
    if not m or m.get("state") != "sealed":
        raise SystemExit("no sealed vault here (`kgkit vault status`)")
    rows = _read(store(m) / ROWS)
    if m.get("mode") == "holdout":
        from . import evalset as E

        path, _ = E.split_spec(m["truth"])
        target = root / path if not Path(path).is_absolute() else Path(path)
    else:
        target = root / m["train"]
    rest = _read(target)
    pos = rows[POS].to_numpy()
    n = len(rest) + len(rows)
    others = np.setdiff1d(np.arange(n), pos)
    if len(others) != len(rest):
        raise SystemExit(f"{target.name} has {len(rest)} rows, expected {n - len(rows)}: was it replaced since the "
                         "seal? Restore it before opening the vault")
    back = rows.drop(columns=[POS, GROUP], errors="ignore")
    full = pd.concat([rest.set_axis(others), back.set_axis(pos)]).sort_index().reset_index(drop=True)
    _write(full[list(rest.columns)], target)
    if m.get("mode") != "holdout":
        fp = root / (m.get("folds") or "data/folds.csv")
        if fp.is_file():
            folds = pd.read_csv(fp)
            folds.loc[folds["fold"] == VAULT, "fold"] = -1
            _write(folds, fp)
        spec_p = root / S.STATE_DIR / "folds_spec.json"
        if spec_p.is_file():
            spec = json.loads(spec_p.read_text(encoding="utf-8"))
            spec["vault"] = {**(spec.get("vault") or {}), "opened": True}
            spec_p.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    elif m.get("groups"):
        from . import evalset as E

        gpath, _ = E.split_spec(m["groups"])
        gp = root / gpath if not Path(gpath).is_absolute() else Path(gpath)
        if gp.resolve() != target.resolve() and GROUP in rows.columns:
            g = _read(gp)
            gcol = E.split_spec(m["groups"])[1][0]
            gb = pd.DataFrame({gcol: rows[GROUP].to_numpy()})
            gfull = pd.concat([g.set_axis(others), gb.set_axis(pos)]).sort_index().reset_index(drop=True)
            _write(gfull[list(g.columns)], gp)
    (root / FEATURES).unlink(missing_ok=True)
    m["state"] = "opened"
    m["opened_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _save_meta(root, m)
    return m
