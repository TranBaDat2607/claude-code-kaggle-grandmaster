"""Command line interface: ``python -m kgkit <command>``.

Commands
  init        create a competition workspace (.kaggle-gm/, folders, CLAUDE.md)
  adopt       adopt an existing project mid-competition: .kaggle-gm/ only, past submissions imported
  config      show or set competition.json fields (validation=holdout+lb, holdout_n, deadline, ...)
  status      one-screen summary: competition, best CV, CV-LB correlation, recent experiments
  folds       write folds.csv with a leak-free strategy and print the fold report
  eda         markdown EDA report with red flags
  adv         adversarial validation (train vs test)
  score       score a prediction column against a truth column with a registered metric
  metrics     list registered metrics
  ledger      list | best | show | lb | add | compare | noise | decide | baseline | lineage | import
  backlog     add | list | set | render | fidelity  — the ranked idea backlog (human vs GPU hours)
  features    search  — generate and screen thousands of candidate features on the frozen folds
  kernels     top | pull | review  — study and reproduce public notebooks
  discussions sync | top | search | solutions | read  — competition forum (live, or Meta Kaggle)
  pages       the competition's own pages (evaluation, rules, data description) as text, via the Kaggle CLI
  recheck     re-draw the folds with a new seed, re-run old vs current pipeline: does the gain survive?
  blend       hill / weights / rank / multi-level stack / residual stack over ledger OOFs + submission
  validate    validate a submission against sample_submission
  vendor      copy kgkit into a project or a Kaggle dataset folder (for offline kernels)
  whoami      your Kaggle username (KAGGLE_USERNAME, kaggle.json or `kaggle config view`; never the key)
  datasets    pull  — download a dataset, file by file when the whole-dataset download returns 404
  gpu         quota | plan | build | push | status | wait | collect | release | cancel | log  — Kaggle GPUs/TPUs
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str):
    import pandas as pd

    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".feather":
        return pd.read_feather(p)
    return pd.read_csv(p)


def _col(spec: str):
    """'file.csv:col' -> Series."""
    path, _, col = spec.rpartition(":")
    if not path or (len(path) == 1 and path.isalpha()):  # windows drive letter, no column
        raise SystemExit(f"expected FILE:COLUMN, got {spec!r}")
    return _read(path)[col]


def _cols(spec: str) -> tuple[str, list[str]]:
    """'file.csv:a,b' -> ('file.csv', ['a', 'b'])."""
    from .evalset import split_spec

    try:
        return split_spec(spec)
    except ValueError as e:
        raise SystemExit(str(e))


def _frame(spec: str):
    """'file.csv:a,b' -> the file and its requested columns, checked to exist."""
    path, cols = _cols(spec)
    df = _read(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} has no column(s) {', '.join(missing)}")
    return df, cols


def _truth(spec: str):
    """Training target as a numpy array; string/bool labels become sorted-class codes (the template encoding).
    Several columns (``FILE:a,b,c``, e.g. one per label of a multilabel target) give a 2-D numeric array."""
    import numpy as np
    import pandas as pd

    df, cols = _frame(spec)
    if len(cols) > 1:
        bad = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c]) or df[c].isna().any()]
        if bad:
            raise SystemExit(f"multi-column truth must be numeric without gaps (0/1 per label); not: {', '.join(bad)}")
        return df[cols].to_numpy()
    y_raw = df[cols[0]]
    if pd.api.types.is_numeric_dtype(y_raw) and not pd.api.types.is_bool_dtype(y_raw):
        return y_raw.to_numpy()
    classes, y = np.unique(y_raw.astype(str).to_numpy(), return_inverse=True)
    print("encoded truth labels: " + ", ".join(f"{c}={i}" for i, c in enumerate(classes)))
    return y


def _preds(spec: str, truth_spec: str | None = None, id_col: str = ""):
    """Predictions as a numpy array: a ``.npy`` file, or ``FILE:COL[,COL...]``. When the prediction file and the
    truth file both have ``id_col``, rows are put in the truth's order; otherwise they must already match it."""
    import numpy as np

    if spec.endswith(".npy"):
        return np.load(spec)
    df, cols = _frame(spec)
    if truth_spec and id_col and id_col in df.columns:
        tdf, _ = _frame(truth_spec)
        if id_col in tdf.columns:
            import pandas as pd

            ids = pd.Index(df[id_col])
            if not ids.is_unique:
                raise SystemExit(f"{spec}: {id_col} has duplicate values, cannot align to the truth rows")
            pos = ids.get_indexer(tdf[id_col])
            if (pos < 0).any():
                raise SystemExit(f"{spec}: {(pos < 0).sum()} {id_col} value(s) of the truth file have no prediction")
            df = df.iloc[pos]
    out = df[cols].to_numpy()
    return out[:, 0] if len(cols) == 1 else out


def _load_pred(led, ref: str, n: int | None):
    """OOF of a ledger experiment, or of a per-fold artefact folder (artifacts/<name>/fold<k>_oof.npy +
    fold<k>_idx.npy, e.g. an unassembled 1-fold screen) as a full-length array with NaN elsewhere."""
    import numpy as np

    d = Path(ref)
    if d.is_dir() and list(d.glob("fold*_oof.npy")):
        if n is None:
            raise SystemExit(f"{ref}: --truth is needed to place per-fold predictions")
        out = None
        for f in sorted(d.glob("fold*_oof.npy")):
            idx = np.load(f.with_name(f.name.replace("_oof.npy", "_idx.npy")))
            p = np.load(f)
            if out is None:
                out = np.full((n, *p.shape[1:]), np.nan)
            out[idx] = p
        return out, ref
    try:
        return led.load_eval_pred(ref), led.get(ref)["id"]
    except ValueError as e:
        raise SystemExit(str(e))


# ---------------------------------------------------------------------------- commands
def _state_from_args(a):
    from . import state as S

    st = S.CompetitionState(
        slug=a.slug, metric=a.metric or "", task=a.task or "", target=a.target or "", id_col=a.id_col or "",
        code_competition=a.code_competition, runtime_limit_hours=a.runtime_hours, internet_allowed=not a.no_internet,
        daily_submissions=a.daily_subs, deadline=a.deadline or "",
        validation=a.validation, holdout_n=a.holdout_n, eval_truth=a.eval_truth or "",
        eval_groups=a.eval_groups or "", public_lb_n=a.public_lb_n,
    )
    if a.metric:
        from . import metrics as M

        try:
            st.greater_is_better = M.get(a.metric).greater_is_better
        except KeyError:
            st.greater_is_better = not a.lower_is_better
            print(f"note: metric {a.metric!r} is not in the registry — implement it in src/metric.py; "
                  f"direction set to {'higher' if st.greater_is_better else 'lower'} is better")
    if a.lower_is_better:
        st.greater_is_better = False
    return st


def cmd_init(a):
    from . import state as S

    root = Path(a.dir).resolve()
    st = _state_from_args(a)
    if (root / S.STATE_DIR / S.STATE_FILE).exists() and not a.force:
        raise SystemExit(f"{root / S.STATE_DIR / S.STATE_FILE} exists (use --force to overwrite)")
    st.save(root)
    for d in ["data", "src", "notebooks", "artifacts", "subs", "reports", "kernels"]:
        (root / d).mkdir(exist_ok=True)
    gi = root / ".gitignore"
    if not gi.exists():
        gi.write_text("data/\nartifacts/\nsubs/*.csv\n*.npy\n*.pkl\n*.pt\n*.pth\n*.ckpt\n*.safetensors\nkaggle.json\n"
                      "__pycache__/\n.ipynb_checkpoints/\nwandb/\n", encoding="utf-8")
    tmpl = PLUGIN_ROOT / "templates" / "CLAUDE.competition.md"
    cm = root / "CLAUDE.md"
    if tmpl.exists() and not cm.exists():
        text = tmpl.read_text(encoding="utf-8")
        for k, v in {"{{SLUG}}": st.slug, "{{METRIC}}": st.metric or "TBD",
                     "{{DIRECTION}}": "higher" if st.greater_is_better else "lower",
                     "{{TASK}}": st.task or "TBD", "{{TARGET}}": st.target or "TBD",
                     "{{CODE_COMP}}": "yes" if st.code_competition else "no",
                     "{{DEADLINE}}": st.deadline or "TBD"}.items():
            text = text.replace(k, str(v))
        cm.write_text(text, encoding="utf-8")
    print(f"initialised competition workspace for '{a.slug}' at {root}")


def cmd_adopt(a):
    """Workspace state for an existing project: only .kaggle-gm/ is written, the repo layout is left alone."""
    import datetime as dt
    import subprocess

    from . import adopt as A
    from . import safety as SF
    from . import state as S

    root = Path(a.dir).resolve()
    found = A.detect(root, max_depth=4, max_dirs=20000, max_seconds=10) or {"slugs": [], "kernels": [],
                                                                          "internet_off": False, "evidence": []}
    if found["evidence"]:
        print(f"found: {', '.join(found['evidence'])}" + (f" ({', '.join(found['kernels'][:8])})" if found["kernels"] else ""))
    slug = a.slug
    if not slug:
        if len(found["slugs"]) != 1:
            raise SystemExit("pass the competition slug: " + (f"candidates {', '.join(found['slugs'])}"
                                                              if found["slugs"] else "none detected"))
        slug = found["slugs"][0]
        print(f"competition: {slug} (detected)")
    a.slug = slug
    state_file = root / S.STATE_DIR / S.STATE_FILE
    if state_file.exists() and not a.force:
        print(f"{state_file} exists: keeping it (use --force to rewrite it)")
        slug = S.load(root).slug
    else:
        st = _state_from_args(a)
        if found["internet_off"] and not a.no_internet:
            st.internet_allowed = False
            print("note: the competition kernels run with internet off; recorded internet_allowed=false")
        if found["internet_off"] and not a.code_competition:
            print("note: offline kernels with competition sources usually mean a code competition - "
                  "re-run with --force --code-competition --runtime-hours H if so")
        st.save(root)
        print(f"wrote {state_file.relative_to(root).as_posix()} for '{slug}' (no folders, .gitignore or CLAUDE.md created)")

    if a.no_submissions:
        text = None
    elif a.submissions_file:
        text = Path(a.submissions_file).read_text(encoding="utf-8")
    else:
        cmd = [shutil.which("kaggle") or "kaggle", "competitions", "submissions", slug, "--format", "json"]
        try:
            r = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=120,
                               env=SF.kaggle_env())
            text = r.stdout if r.returncode == 0 else None
            err = (r.stderr or r.stdout).strip().splitlines()[-1:] if r.returncode else []
        except (OSError, subprocess.TimeoutExpired) as e:
            text, err = None, [str(e)]
        if text is None:
            print(f"could not fetch the submission history ({' '.join(err) or 'kaggle CLI failed'}); "
                  f"retry with `kaggle competitions submissions {slug} --format json > subs.json` and "
                  f"`python -m kgkit adopt {slug} --submissions-file subs.json`")
    if text is not None:
        try:
            recs = A.parse_submissions(text, slug)
        except ValueError as e:
            raise SystemExit(f"could not parse the submission history: {e}")
        added, skipped = A.import_submissions(root, recs)
        today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        n_today = sum(r["time_utc"].startswith(today) and r["status"] != "failed" for r in recs)
        scored = [r for r in recs if r["public_score"] is not None]
        print(f"submission history: {len(recs)} on Kaggle, {added} imported, {skipped} already logged; "
              f"{n_today} today (UTC), {len(scored)} with a public score")
    print("next: record past experiments in the ledger (`kgkit ledger import <name> <oof> --truth ... --folds ...` "
          "or `kgkit ledger add <name> <cv> <notes>`), mark the accepted one with `kgkit ledger decide <id> baseline`, "
          "then `kgkit status`")


def cmd_config(a):
    """Show or change fields of .kaggle-gm/competition.json (typed like the field's default)."""
    import dataclasses

    from . import state as S

    root = S.find_root()
    st = S.load()
    if st is None:
        raise SystemExit("not inside a kaggle-grandmaster workspace (no .kaggle-gm/competition.json)")
    fields = {f.name: f for f in dataclasses.fields(st)}
    if not a.set:
        for k in fields:
            print(f"{k} = {json.dumps(getattr(st, k))}")
        return
    for kv in a.set:
        k, sep, v = kv.partition("=")
        k = k.strip().replace("-", "_")
        if not sep or k not in fields:
            raise SystemExit(f"expected KEY=VALUE with KEY one of {', '.join(fields)}; got {kv!r}")
        cur = getattr(st, k)
        if v.strip().lower() in ("", "none", "null"):
            val = None if k in ("runtime_limit_hours", "holdout_n", "holdout_se", "public_lb_n", "rope") else ""
        elif isinstance(cur, bool):
            val = v.strip().lower() in ("1", "true", "yes", "on")
        elif k in ("daily_submissions", "final_submissions", "holdout_n", "public_lb_n"):
            val = int(v)
        elif k in ("runtime_limit_hours", "holdout_se", "rope"):
            val = float(v)
        elif isinstance(cur, list):
            val = [x.strip() for x in v.split(",") if x.strip()]
        else:
            val = v.strip()
        if k == "validation" and val not in S.VALIDATION_MODES:
            raise SystemExit(f"validation must be one of {', '.join(S.VALIDATION_MODES)}")
        setattr(st, k, val)
        print(f"{k} = {json.dumps(val)}")
    st.save(root)


def cmd_status(a):
    from . import state as S
    from .experiment import Ledger, fmt_cv

    st = S.load(a.dir)
    if st is None:
        print("not inside a kaggle-grandmaster workspace (no .kaggle-gm/competition.json). Run: python -m kgkit init <slug>")
        return
    root = S.find_root(a.dir)
    led = Ledger(root)
    recs = led.records()
    direction = "higher" if st.greater_is_better else "lower"
    print(f"# {st.slug}  —  metric: {st.metric or '?'} ({direction} is better), task: {st.task or '?'}")
    bits = []
    if st.code_competition:
        bits.append(f"CODE COMPETITION (runtime {st.runtime_limit_hours or '?'}h, internet {'on' if st.internet_allowed else 'OFF'})")
    if st.deadline:
        bits.append(f"deadline {st.deadline}")
    bits.append(f"{st.daily_submissions} subs/day, {st.final_submissions} final picks")
    if st.holdout:
        bits.append(f"validation: {st.validation}" + (f" (holdout n={st.holdout_n})" if st.holdout_n else ""))
    print("; ".join(bits))
    from . import gpu as G

    q = G.cached_quota(root)
    active = G.active_runs(root)
    if q is not None or active:
        avail = (f"{G.available_hours(root, q):.1f}h plannable (cached {q.get('fetched_at')}, resets "
                 f"{q.get('refresh_at')})" if q is not None else "quota not fetched yet")
        print(f"Kaggle GPU: {avail}" + (f"; in flight (as last seen, `gpu status` refreshes): "
                                         f"{', '.join(G.run_label(r) for r in active)}" if active else ""))
    print(f"experiments logged: {len(recs)}")
    if recs:
        best = led.best(1)
        if best:
            print(f"best CV: {best[0]['cv']:.5f} ({best[0]['id']})")
        base = led.baseline(fallback=False)
        if base is not None:
            print(f"accepted baseline: {fmt_cv(base)} ({base['id']}, {base.get('decision')}) - new ideas are "
                  f"compared against this, not against the best CV")
        else:
            print("no accepted baseline yet: mark one with `kgkit ledger decide <id> baseline`")
        n_dec = led.decisions_on_evalset()
        if st.holdout and n_dec >= 10:
            print(f"{n_dec} keep/discard decisions taken on this fixed holdout: each one spends a little of it, and "
                  "there is no fresh split to re-check on. Settle close calls on the public LB as well, and treat "
                  "marginal KEEPs as provisional (`kgkit ledger lineage`).")
        elif not st.holdout and n_dec >= 20:
            print(f"{n_dec} keep/discard decisions taken on the current folds: the CV is getting optimistic. "
                  "Re-check on a fresh fold seed: `kgkit recheck --seed 7 --run <old> --run <current> --exp <root> <baseline>`")
        with_lb = [r for r in recs if r.get("lb_public") is not None]
        if with_lb:
            bl = sorted(with_lb, key=lambda r: r["lb_public"], reverse=st.greater_is_better)[0]
            print(f"best public LB: {bl['lb_public']:.5f} ({bl['id']}, CV {fmt_cv(bl)})")
        agreement = led.cv_lb_summary() or led.noise_note()  # the noise line rides on the CV-LB line when there is one
        if agreement:
            print(agreement)
        print()
        print(led.table(a.n))
    from .backlog import Backlog

    top = Backlog(root).ranked()[:3]
    if top:
        print("\ntop of the backlog: " + "; ".join(f"#{i['id']} {i['idea'][:60]}" for i in top))


def cmd_folds(a):
    from .cv import assign_folds, fold_report

    df = _read(a.train)
    target = a.target.split(",") if a.target and "," in a.target else a.target
    folds = assign_folds(df, a.n_splits, a.strategy, target=target, group=a.group, seed=a.seed)
    df["fold"] = folds
    keep = [c for c in [a.id_col] if c and c in df.columns] + ["fold"]
    df[keep].to_csv(a.out, index=False)
    from . import state as S

    root = S.find_root()
    if root is not None:  # remember how the split was made, so `kgkit recheck` can re-draw it with another seed
        spec = {"train": a.train, "strategy": a.strategy, "target": a.target, "group": a.group,
                "n_splits": a.n_splits, "seed": a.seed, "id_col": a.id_col, "out": a.out}
        (root / S.STATE_DIR / "folds_spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(fold_report(df, "fold", target=target if isinstance(target, str) else None, group=a.group))
    print(f"\nwrote {a.out} ({len(df)} rows, strategy={a.strategy}, seed={a.seed})")


def cmd_eda(a):
    from .eda import quick_eda

    tr = _read(a.train)
    te = _read(a.test) if a.test else None
    rep = quick_eda(tr, te, a.target, a.id_col)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(rep, encoding="utf-8")
        print(f"wrote {a.out}")
    print(rep)


def cmd_adv(a):
    from .adversarial import adversarial_validation

    tr, te = _read(a.train), _read(a.test)
    drop = set(a.drop.split(",")) if a.drop else set()
    feats = [c for c in tr.columns if c in te.columns and c not in drop]
    res = adversarial_validation(tr, te, feats, n_splits=a.n_splits)
    print(f"adversarial AUC: {res['auc']:.4f}  (folds: {', '.join(f'{x:.3f}' for x in res['fold_aucs'])})")
    verdict = ("no meaningful shift" if res["auc"] < 0.6 else "moderate shift — inspect top features" if res["auc"] < 0.8
               else "STRONG shift — CV may not track LB; investigate before modelling")
    print(f"verdict: {verdict}\n\ntop drifting features (AUC drop when permuted):")
    print(res["importance"].head(15).to_string(index=False))
    if a.save_p:
        import pandas as pd

        pd.DataFrame({"p_test": res["p_test"]}).to_csv(a.save_p, index=False)
        print(f"\nwrote per-train-row p(test) to {a.save_p}")


def cmd_score(a):
    from . import metrics as M
    from . import state as S

    st = S.load()
    tdf, tcols = _frame(a.truth)
    y = tdf[tcols].to_numpy()
    p = _preds(a.pred, a.truth, st.id_col if st else "")
    print(f"{a.metric}: {M.score(a.metric, y[:, 0] if len(tcols) == 1 else y, p):.6f}")


def cmd_metrics(a):
    from . import metrics as M

    for m in M.list_metrics():
        print(f"{m.name:20s} {'max' if m.greater_is_better else 'min'}  {m.kind:8s} {m.description}")


def _cv_arg(v: str) -> float | None:
    """A CV value, or none / lb-only / - for a record that cannot be cross-validated."""
    if v.lower() in ("none", "lb-only", "-", "na", "n/a"):
        return None
    try:
        return float(v)
    except ValueError:
        raise SystemExit(f"CV must be a number or 'lb-only', got {v!r}")


def cmd_ledger(a):
    from .experiment import Ledger, fmt_cv

    led = Ledger()
    if a.action == "list":
        print(led.table(a.n))
    elif a.action == "best":
        print(led.table(a.n, sort="cv"))
    elif a.action == "show":
        print(json.dumps(led.get(a.args[0]), indent=2))
    elif a.action == "lb":
        if len(a.args) < 2:
            raise SystemExit("usage: ledger lb <exp_id> <public_score> [private_score]")
        priv = float(a.args[2]) if len(a.args) > 2 else None
        r = led.attach_lb(a.args[0], float(a.args[1]), priv)
        print(f"{r['id']}: CV {fmt_cv(r)} -> public LB {r['lb_public']}")
        agreement = led.cv_lb_summary()
        if agreement:
            print(agreement)
    elif a.action == "add":
        if len(a.args) < 2:
            raise SystemExit("usage: ledger add <name> <cv|auto|lb-only> [notes] [--pred FILE:COLS|preds.npy] "
                             "[--tag lb-blend] [--variant V]")
        extra = {"variant": a.variant} if a.variant else {}
        if a.pred:
            r = _ledger_add_preds(led, a, extra)
            _auto_compare(led, a, r)
            return
        if a.args[1].lower() == "auto":
            raise SystemExit("'auto' recomputes the score from predictions: pass --pred FILE:COLS (or preds.npy)")
        r = led.log(a.args[0], _cv_arg(a.args[1]), notes=" ".join(a.args[2:]), parent=a.parent, tags=a.tag or (),
                    **extra)
        print(f"logged {r['id']}" + ("" if r["cv"] is not None else " (LB-only: attach its score with "
                                     f"`ledger lb {r['id']} <score>`; it stays out of the CV-LB correlation)"))
    elif a.action == "compare":
        _ledger_compare(led, a)
    elif a.action == "noise":
        _ledger_noise(led, a)
    elif a.action == "decide":
        if len(a.args) < 2:
            raise SystemExit("usage: ledger decide <exp_id> baseline|keep|discard|inconclusive [note] [--parent ID]")
        r = led.decide(a.args[0], a.args[1], " ".join(a.args[2:]), parent=a.parent)
        print(f"{r['id']}: {r['decision']}" + (f" (vs {r['parent']})" if r.get("parent") else ""))
        if r["decision"] in ("keep", "baseline"):
            print(f"accepted baseline is now {r['id']} (CV {fmt_cv(r)})")
    elif a.action == "baseline":
        r = led.baseline()
        if r is None:
            print("(ledger is empty)")
        else:
            src = r.get("decision") if r.get("decision") in ("keep", "baseline") else "highest CV - nothing accepted yet"
            print(f"{r['id']}  CV {fmt_cv(r)}  [{src}]")
    elif a.action == "lineage":
        chain = led.lineage(a.args[0] if a.args else led.baseline()["id"])
        prev = None
        for r in chain:
            d = f"{r['cv'] - prev:+.5f}" if prev is not None and r.get("cv") is not None else "root"
            print(f"{r['id']:40s} CV {fmt_cv(r)} ({d})  {r.get('decision') or '-':12s} "
                  f"{(r.get('decision_note') or r.get('notes') or '')[:60]}")
            prev = r.get("cv")
        if len(chain) > 2:
            print("\nreverse ablation: re-test removing each marginal KEEP above from the current baseline; "
                  "small gains decided on the same folds can be noise that compounded.")
    elif a.action == "import":
        import numpy as np

        if len(a.args) < 2:
            raise SystemExit("usage: ledger import <name> <oof.npy|FILE:COLUMN> [--test test.npy|FILE:COLUMN] "
                             "[--truth FILE:COL --folds FILE:COL | --cv X] [--source teammate|notebook]")

        def arr(spec):
            return np.load(spec) if spec.endswith(".npy") else _col(spec).to_numpy()

        y = _truth(a.truth) if a.truth else None
        folds = _col(a.folds).to_numpy() if a.folds else None
        r = led.import_preds(a.args[0], arr(a.args[1]), arr(a.test) if a.test else None, y_true=y, folds=folds,
                             cv=a.cv, notes=a.notes or "", source=a.source)
        print(f"logged {r['id']}: CV {r['cv']:.5f}" + (f" +/- {r['cv_std']:.5f}" if r.get("cv_std") else "")
              + f" [{a.source}]")


def _ledger_add_preds(led, a, extra: dict) -> dict:
    """`ledger add <name> <cv|auto> --pred ...`: the score is recomputed from predictions on the evaluation set,
    and they are stored (holdout predictions in a holdout+lb workspace, OOF otherwise) so that later records can
    be compared with this one, paired, on the same rows."""
    from . import metrics as M
    from . import state as S
    from .experiment import score_preds

    st = S.load()
    truth_spec = a.truth or (st.eval_truth if st else "")
    if not truth_spec:
        raise SystemExit("--pred needs the evaluation labels: `kgkit config eval_truth=FILE:COL[,COL...]` (one column "
                         "per label of a multilabel target) or --truth FILE:COLS")
    metric = a.metric or (st.metric if st else None)
    if not metric:
        raise SystemExit("--metric required (no competition state found)")
    try:
        m = M.get(metric)
    except KeyError as e:
        raise SystemExit(e.args[0])
    auto = a.args[1].lower() == "auto"
    passed = None if auto else _cv_arg(a.args[1])
    if not auto and passed is None:
        raise SystemExit("an LB-only record has no score to recompute: drop --pred, or pass 'auto'")
    y = _truth(truth_spec)
    p = _preds(a.pred, truth_spec, st.id_col if st else "")
    if len(p) != len(y):
        raise SystemExit(f"{a.pred} has {len(p)} rows but {truth_spec} has {len(y)}")
    holdout = st is not None and st.holdout
    folds = _col(a.folds).to_numpy() if a.folds and not holdout else None
    score, fold_scores = score_preds(m.name, y, p, folds)
    if passed is not None and abs(passed - score) > 1e-6 * max(1.0, abs(score)):
        print(f"note: {m.name} recomputed from the predictions is {score:.6f}, not {passed}; logging the recomputed score")
    kw = dict(notes=" ".join(a.args[2:]), parent=a.parent, tags=a.tag or (), metric=m.name, fold_scores=fold_scores,
              **extra)
    r = led.log(a.args[0], score, holdout_pred=p, **kw) if holdout else led.log(a.args[0], score, oof=p, **kw)
    where = "holdout" if holdout else "OOF"
    blend = "; tagged lb-blend, its own holdout score still counts" if holdout and "lb-blend" in (a.tag or ()) else ""
    print(f"logged {r['id']}: {where} {m.name} {score:.5f} recomputed from {len(y)} rows, predictions stored{blend}")
    return r


def _auto_compare(led, a, r: dict) -> None:
    """After `ledger add --pred`: compare the new record, paired, with the accepted baseline when that baseline
    has predictions on the same evaluation set."""
    import argparse

    base = led.baseline(fallback=False)
    path = "holdout_path" if led._holdout() else "oof_path"
    if base is None or base["id"] == r["id"] or not base.get(path):
        if base is None:
            print("no accepted baseline yet: `kgkit ledger decide <id> baseline`, then later records are compared "
                  "with it automatically")
        return
    if led.eval_record_key(base) != led.eval_record_key(r):
        print(f"not compared with the baseline {base['id']}: its predictions are on a different evaluation set")
        return
    print()
    _ledger_compare(led, argparse.Namespace(**{**vars(a), "args": [r["id"], base["id"]]}))


def _ledger_compare(led, a):
    import numpy as np

    from . import metrics as M
    from . import state as S
    from .compare import compare_folds, describe, estimate, format_comparison, verdict

    if len(a.args) < 1:
        raise SystemExit("usage: ledger compare <new> [baseline] [--truth FILE:COL] [--folds FILE:COL] "
                         "[--fold K] [--groups FILE:COL] [--public-n ROWS]   (baseline defaults to the accepted "
                         "baseline)")
    st = S.load()
    holdout = st is not None and st.holdout
    if len(a.args) > 1:
        base_ref = a.args[1]
    else:
        base = led.baseline()
        if base is None:
            raise SystemExit("the ledger is empty: nothing to compare against")
        base_ref = base["id"]
    for ref in (a.args[0], base_ref):
        if not Path(ref).is_dir():
            led.get(ref)  # fail early on a typo
    if not Path(a.args[0]).is_dir() and not Path(base_ref).is_dir():
        ha, hb = (led.eval_record_key(led.get(r)) for r in (a.args[0], base_ref))
        if led.get(a.args[0])["id"] == led.get(base_ref)["id"]:
            raise SystemExit(f"{base_ref} is both the new experiment and the baseline (no accepted baseline yet? "
                             "pass the baseline explicitly, or `ledger decide <id> baseline`)")
        if ha and hb and ha != hb:
            print("WARNING: the two experiments were scored on different holdout labels (eval_hash differs) - they "
                  "are not comparable" if holdout else
                  "WARNING: the two experiments used different fold splits - the per-fold pairing is not valid; "
                  "only the OOF bootstrap (with --truth) is meaningful")
    if holdout and (a.folds or a.fold is not None):
        raise SystemExit("holdout+lb workspace: there are no folds; the holdout predictions are compared as a whole")
    truth_spec = a.truth
    if not truth_spec and st is not None and st.eval_truth:  # K-fold records without OOF: fold scores only
        if holdout or all(Path(r).is_dir() or led.get(r).get("oof_path") for r in (a.args[0], base_ref)):
            truth_spec = st.eval_truth
    if holdout and not truth_spec:
        raise SystemExit("holdout+lb: set the holdout labels (`kgkit config eval_truth=FILE:COL[,COL...]`) or pass "
                         "--truth; the stored holdout predictions are then compared paired")
    y = _truth(truth_spec) if truth_spec else None
    n = len(y) if y is not None else None
    metric_name = a.metric or (st.metric if st else None)
    gib = st.greater_is_better if st else True
    folds_res = boot = None
    if y is None:
        if Path(a.args[0]).is_dir() or Path(base_ref).is_dir():
            raise SystemExit("artifact folders have no stored fold scores: pass --truth FILE:COL --folds FILE:COL")
        ra, rb = led.get(a.args[0]), led.get(base_ref)
        if not (ra.get("fold_scores") and rb.get("fold_scores")) or len(ra["fold_scores"]) != len(rb["fold_scores"]):
            raise SystemExit("no comparable fold_scores in the ledger: pass --truth (and --folds) to compare OOFs")
        folds_res = compare_folds(ra["fold_scores"], rb["fold_scores"], gib)
        new_id, base_id = ra["id"], rb["id"]
    else:
        if not metric_name:
            raise SystemExit("--metric required (no competition state found)")
        m = M.get(metric_name)
        gib = m.greater_is_better
        pa, new_id = _load_pred(led, a.args[0], n)
        pb, base_id = _load_pred(led, base_ref, n)
        ok = ~np.isnan(pa.reshape(n, -1)).any(1) & ~np.isnan(pb.reshape(n, -1)).any(1)
        folds = _col(a.folds).to_numpy() if a.folds else None
        if a.fold is not None:
            if folds is None:
                raise SystemExit("--fold needs --folds FILE:COL")
            ok &= folds == a.fold
        if folds is not None:
            ks = [k for k in sorted(set(folds[ok & (folds >= 0)].tolist()))]
            if len(ks) >= 2:
                fa = [m(y[ok & (folds == k)], pa[ok & (folds == k)]) for k in ks]
                fb = [m(y[ok & (folds == k)], pb[ok & (folds == k)]) for k in ks]
                folds_res = compare_folds(fa, fb, gib)
        if m.kind == "label":
            print(f"note: {m.name} scores labels; OOFs are compared as stored (decode them first if they are "
                  "probabilities)")
        groups_spec = a.groups or (st.eval_groups if st else "")
        groups = _col(groups_spec).to_numpy() if groups_spec else None
        boot = estimate(y, pa, pb, m.name, method=a.method, n_boot=a.boot, groups=groups, mask=ok)
    v = verdict(folds_res, boot)
    print(format_comparison(new_id, base_id, v))
    lb = None
    if boot is not None:
        print(f"  {describe(boot)}")
        for c in boot["caveats"]:
            print(f"  note: {c}")
        public_n = a.public_n or (st.public_lb_n if st else None)
        if public_n:
            lb = estimate(y, pa, pb, m.name, method=a.method, n_boot=a.boot, groups=groups, mask=ok, size=public_n)
            print(f"  public LB (n={public_n}): {describe(lb)}: LB gaps between these two under ~{2 * lb['se']:.5f} "
                  "are noise (assumes the public rows look like the evaluation rows)")
    _log_comparison(led, new_id, base_id, folds_res, boot, lb, v["decision"])
    print(f"\nrecord it: python -m kgkit ledger decide {new_id} keep|discard|inconclusive \"<why>\" --parent {base_id}")


def _log_comparison(led, new_id: str, base_id: str, folds_res: dict | None, boot: dict | None, lb: dict | None,
                    decision: str) -> None:
    """Append the measured paired noise to .kaggle-gm/comparisons.jsonl: `status` and the session hook read it
    there instead of re-running a bootstrap (or guessing from the size of the evaluation set)."""
    import math

    from . import evalset as EV

    src = boot if boot is not None else folds_res
    if src is None or not (isinstance(src.get("se"), float) and math.isfinite(src["se"]) and src["se"] > 0):
        return
    try:
        key = led.eval_record_key(led.get(new_id))
    except KeyError:  # an artefact folder
        key = None
    entry = {"new": new_id, "base": base_id, "eval": key or led.eval_key(), "question": "paired",
             "gain": src["gain"], "se": src["se"], "decision": decision,
             "method": boot["method"] if boot is not None else "fold-paired t",
             "unit": boot["unit"] if boot is not None else "folds",
             "n_units": boot["n_units"] if boot is not None else folds_res["k"]}
    if lb is not None:
        entry.update(public_n=lb["size"], se_public=lb["se"])
    EV.log_comparison(led.root, entry)


def _ledger_noise(led, a):
    """`ledger noise <id>`: the noise of one absolute score. It bounds that record's CV-LB gap; it is not the
    noise of a difference between two models (that is `ledger compare`, paired)."""
    import math

    from . import metrics as M
    from . import state as S
    from .compare import describe, estimate

    if len(a.args) < 1:
        raise SystemExit("usage: ledger noise <exp_id> [--public-n ROWS] [--truth FILE:COLS] [--groups FILE:COL]")
    st = S.load()
    truth_spec = a.truth or (st.eval_truth if st else "")
    metric_name = a.metric or (st.metric if st else None)
    if not truth_spec or not metric_name:
        raise SystemExit("needs the evaluation labels (`kgkit config eval_truth=FILE:COLS` or --truth) and a metric")
    m = M.get(metric_name)
    y = _truth(truth_spec)
    p, rid = _load_pred(led, a.args[0], len(y))
    groups_spec = a.groups or (st.eval_groups if st else "")
    groups = _col(groups_spec).to_numpy() if groups_spec else None
    res = estimate(y, p, None, m.name, method=a.method, n_boot=a.boot, groups=groups)
    print(f"{rid}: {m.name} {res['value']:.5f}; {describe(res)}")
    for c in res["caveats"]:
        print(f"  note: {c}")
    public_n = a.public_n or (st.public_lb_n if st else None)
    if public_n:
        lb = estimate(y, p, None, m.name, method=a.method, n_boot=a.boot, groups=groups, size=public_n)
        gap = math.hypot(res["se"], lb["se"])
        print(f"public LB (n={public_n}): {describe(lb)}; a CV-LB gap within ~{2 * gap:.5f} (2 x sqrt(SE_eval^2 + "
              "SE_LB^2)) is sampling noise, not a validation problem (assumes the public rows look like the "
              "evaluation rows)")
    print("this is the noise of one absolute score: judge a difference between two models with `kgkit ledger "
          "compare` (paired), not with this")


def cmd_blend(a):
    import numpy as np
    import pandas as pd

    from . import ensemble as E
    from . import metrics as M
    from . import state as S
    from .experiment import Ledger

    led = Ledger()
    st = S.load()
    metric = a.metric or (st.metric if st else None)
    if not metric:
        raise SystemExit("--metric required (no competition state found)")
    y = _truth(a.truth)  # string / bool labels: sorted-class codes, the same encoding the training templates use
    oofs = {e: led.load_oof(e) for e in a.exp}
    if a.prune:
        pr = E.prune_library(oofs, y, metric, max_models=a.prune, corr_threshold=a.corr_threshold)
        for n, why in pr["dropped"].items():
            print(f"pruned {n}: {why}")
        oofs = {e: oofs[e] for e in pr["kept"]}
        a.exp = pr["kept"]
    for e, o in oofs.items():
        if len(o) != len(y):
            raise SystemExit(f"OOF of {e} has {len(o)} rows but truth has {len(y)}")
    hashes = {e: led.get(e).get("folds_hash") for e in a.exp}
    if len({h for h in hashes.values() if h}) > 1:
        print("WARNING: members were trained on DIFFERENT fold splits: "
              + ", ".join(f"{e}={h}" for e, h in hashes.items())
              + "\n  Blend weights fit on mismatched OOFs leak held-out labels into the level-2 fit, so nested/"
                "stacked scores are optimistic. Re-run members on the shared folds, or keep the blend simple.\n")
    if len(oofs) <= 25:
        print("OOF correlation:\n" + E.oof_correlation(oofs).round(4).to_string() + "\n")
    if a.method in ("stack", "residual"):
        if not a.folds:
            raise SystemExit(f"--method {a.method} needs --folds FILE:COLUMN (meta-models are fit out-of-fold)")
        folds = _col(a.folds).to_numpy()
        tests = {e: led.load_test(e) for e in oofs}
        if a.method == "stack":
            res = E.multi_level_stack(oofs, tests, y, folds, metric, n_layers=a.levels)
            for i, ls in enumerate(res["layer_scores"], start=2):
                print(f"level {i}: " + ", ".join(f"{k}={v:.6f}" for k, v in ls.items()))
            print(f"final blend of the last level: " + ", ".join(f"{k}={v:.3f}" for k, v in res["weights"].items()))
            params = {"method": "stack", "levels": a.levels + 1, "members": list(oofs), "final_weights": res["weights"]}
        else:
            base = a.base or max(oofs, key=lambda e: M.score(metric, y, oofs[e]) * (1 if M.get(metric).greater_is_better else -1))
            res = E.residual_stack(oofs, tests, y, folds, base=base)
            print(f"residual stack on base {base}: base OOF {M.score(metric, y, oofs[base]):.6f}")
            params = {"method": "residual", "base": base, "members": list(oofs)}
        score = M.score(metric, y, res["oof"])
        print(f"meta OOF {metric}: {score:.6f} (out-of-fold; stacking on the same folds is still slightly optimistic)")
        if res.get("final_honest") is not None:
            print(f"honest (nested) final-blend {metric}: {res['final_honest']:.6f}")
        rec = led.log(a.name, score, model="blend", params=params, oof=res["oof"], test_pred=res["test"],
                      notes=f"{a.method} of {', '.join(oofs)}", metric=metric)
        print(f"logged {rec['id']}")
        _write_blend_sub(led, rec, res["test"], a)
        return
    if a.method == "hill":
        res = E.hill_climb(oofs, y, metric, allow_negative=a.allow_negative)
    elif a.method == "weights":
        res = E.optimize_weights(oofs, y, metric)
    else:
        res = {"weights": {e: 1 / len(oofs) for e in oofs}}
        res["score"] = M.score(metric, y, E.blend(oofs, res["weights"], "rank"))
    print("weights: " + ", ".join(f"{k}={v:.3f}" for k, v in res["weights"].items()))
    print(f"blended OOF {metric}: {res['score']:.6f}")
    if a.folds:
        honest = E.cv_blend_score(oofs, y, _col(a.folds).to_numpy(), metric,
                                  method="hill_climb" if a.method == "hill" else "weights")
        print(f"honest (nested) blended {metric}: {honest['oof_score']:.6f}")
    method = "rank" if a.method == "rank" else "mean"
    blended_oof = E.blend(oofs, res["weights"], method)
    tests = {e: led.load_test(e) for e in res["weights"]}
    blended_test = E.blend(tests, res["weights"], method)
    rec = led.log(a.name, res["score"], model="blend", params={"weights": res["weights"], "method": a.method},
                  oof=blended_oof, test_pred=blended_test, notes=f"blend of {', '.join(a.exp)}", metric=metric)
    print(f"logged {rec['id']}")
    _write_blend_sub(led, rec, blended_test, a)


def _write_blend_sub(led, rec, blended_test, a):
    import numpy as np
    import pandas as pd

    if a.sample:
        sub = pd.read_csv(a.sample)
        cols = [c for c in sub.columns[1:]] if not a.pred_cols else a.pred_cols.split(",")
        bt = np.asarray(blended_test)
        if bt.ndim == 1:
            sub[cols[0]] = bt
        else:
            sub[cols] = bt
        out = Path(a.out or f"subs/{rec['id']}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out, index=False)
        led.update(rec["id"], submission=str(out))
        print(f"wrote {out}")


def cmd_backlog(a):
    from .backlog import Backlog

    bl = Backlog()
    if a.action == "add":
        if not a.args:
            raise SystemExit('usage: backlog add "<idea>" --gain 1-5 --prob 0-1 --cost HOURS [--gpu-hours H] '
                             '[--evidence ..] [--source ..] [--status ..] [--result ..]')
        i = bl.add(" ".join(a.args), gain=a.gain or 2, prob=a.prob if a.prob is not None else 0.3,
                   cost=a.cost if a.cost is not None else (0 if a.gpu_hours else 1), evidence=a.evidence or "",
                   source=a.source or "", first_experiment=a.first_experiment or "", gpu_hours=a.gpu_hours or 0,
                   status=a.status or "todo", result=a.result or "")
        if a.exp:
            i = bl.set(i["id"], exp=a.exp)
        print(f"#{i['id']} {i['idea']} (status {i['status']}, score {bl.score(i):.2f})"
              + (f": {i['result']}" if i.get("result") else ""))
    elif a.action == "list":
        print(bl.table(include_closed=a.all, n=a.n))
    elif a.action == "set":
        if not a.args:
            raise SystemExit("usage: backlog set <id> [--status ..] [--exp ID] [--result ..] [--screen-gain X] [--full-gain Y]")
        i = bl.set(int(a.args[0]), exp=a.exp, status=a.status, result=a.result, gain=a.gain, prob=a.prob,
                   cost=a.cost, gpu_hours=a.gpu_hours, evidence=a.evidence, screen_gain=a.screen_gain,
                   full_gain=a.full_gain)
        print(f"#{i['id']} {i['idea']} -> {i['status']}" + (f": {i['result']}" if i.get("result") else ""))
    elif a.action == "render":
        print(f"wrote {bl.render(a.out)}")
    elif a.action == "fidelity":
        f = bl.fidelity()
        if f is None:
            print("need >= 3 ideas with both --screen-gain and --full-gain recorded")
        else:
            print(f"screens vs full CV over {f['n']} ideas: sign agreement {f['sign_agreement']:.0%}, "
                  f"spearman {f['spearman']:.2f}")
            if f["sign_agreement"] < 0.7 or f["spearman"] < 0.5:
                print("screens are a poor proxy here: screen on 2 folds, at closer-to-final resolution/epochs, "
                      "or with 2 seeds before promoting")


def cmd_features(a):
    import numpy as np
    import pandas as pd

    from . import featsearch as FS
    from . import state as S

    st = S.load()
    train = _read(a.train)
    target = a.target or (st.target if st else None)
    if not target:
        raise SystemExit("--target required")
    y = _truth(f"{a.train}:{target}") if not a.truth else _truth(a.truth)
    folds = _col(a.folds).to_numpy()
    test = _read(a.test) if a.test else None
    id_col = a.id_col or (st.id_col if st else None)
    drop = {target, id_col, "fold"} | set((a.drop or "").split(","))
    cols = [c for c in train.columns if c not in drop]
    is_num = {c: pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c]) for c in cols}
    # integer columns with few distinct values are usually codes: use them as group keys too (they stay numeric)
    int_codes = [c for c in cols if is_num[c] and train[c].nunique() <= 200
                 and np.allclose(train[c].dropna() % 1, 0)]
    cats = a.cat.split(",") if a.cat else [c for c in cols if not is_num[c]] + int_codes
    nums = a.num.split(",") if a.num else [c for c in cols if is_num[c]]
    print(f"categorical keys: {', '.join(cats) or '-'}\nnumeric: {', '.join(nums) or '-'}  (override with --cat/--num)")
    metric = a.metric or (st.metric if st else None)
    if not metric:
        raise SystemExit("--metric required (no competition state found)")
    specs = FS.generate_specs(cats, nums, kinds=a.kinds.split(","), aggs=a.aggs.split(","),
                              max_candidates=a.max_candidates, seed=a.seed)
    print(f"{len(specs)} candidates from {len(cats)} categorical x {len(nums)} numeric columns")
    recheck = None
    if a.recheck_seed is not None:
        from .cv import assign_folds

        k = len(set(folds[folds >= 0].tolist()))
        df = pd.DataFrame({"__y": y, "__g": train[a.group].to_numpy() if a.group else 0})
        recheck = assign_folds(df, k, "auto", target="__y", group="__g" if a.group else None, seed=a.recheck_seed)
        if (recheck == folds).all():
            print("WARNING: the recheck split equals the screening split (deterministic splitter); recheck skipped")
            recheck = None
    res = FS.feature_search(train, y, folds, cats, nums, metric, base_cols=list(dict.fromkeys(cats + nums)),
                            test=test, specs=specs,
                            batch_size=a.batch, max_batches=a.max_batches,
                            screen_folds=[int(x) for x in a.screen_folds.split(",")] if a.screen_folds else None,
                            model=a.model, time_budget_s=a.minutes * 60 if a.minutes else None,
                            recheck_folds=recheck, seed=a.seed)
    out = FS.save_specs(a.out, res)
    md = Path(a.out).with_suffix(".md")
    md.write_text(FS.report(res), encoding="utf-8")
    print(FS.report(res).split("## Kept features")[0])
    print(f"wrote {out} and {md}")


def cmd_kernels(a):
    from . import kernels as K
    from . import safety as SF
    from . import state as S

    st = S.load()
    slug = a.competition or (st.slug if st else None)
    if a.action == "top":
        if not slug:
            raise SystemExit("--competition required (no competition state found)")
        rows = K.top(slug, a.n, a.sort_by)
        table = "\n".join(f"{r.get('totalVotes', ''):>5}  {r.get('ref', '')}  |  {r.get('title', '')}  "
                          f"({r.get('lastRunTime', '')})" for r in rows)
        print(SF.UNTRUSTED_NOTE)
        print(SF.wrap(table, "kaggle-cli", competition=slug, view="top kernels"))
        print("\nnext: python -m kgkit kernels pull <ref>   (review + local script under ref/)")
    elif a.action in ("pull", "review"):
        if not a.ref:
            raise SystemExit(f"usage: kernels {a.action} <owner/slug>")
        holdout = bool(st and st.holdout)
        if a.action == "pull":
            rev = K.pull(a.ref, a.dest, slug, check_access=a.check_access, holdout=holdout)
        else:
            rev = K.analyse_dir(Path(a.dest) / a.ref.split("/")[-1], a.ref, slug, check_access=a.check_access,
                                holdout=holdout)
        print((Path(rev["dir"]) / "REVIEW.md").read_text(encoding="utf-8"))
        print(f"local script: {rev['dir']}/{a.ref.split('/')[-1]}_local.py")


def cmd_discussions(a):
    import pandas as pd

    from . import discussions as D
    from . import safety as SF
    from . import state as S

    st = S.load()
    slug = a.competition or (st.slug if st else None)
    if not slug:
        raise SystemExit("--competition required (no competition state found)")
    root = S.find_root()
    idx_path = (root / "reports" if root else Path("reports")) / (
        "discussions.csv" if not a.competition or (st and a.competition == st.slug) else f"discussions_{slug}.csv")
    def build_index():
        """(topics, Meta Kaggle competition row or None, where they came from)"""
        if a.source != "meta":
            try:
                return D.live_index(slug, a.max_pages), None, "live, Kaggle CLI"
            except D.LiveUnavailable as e:
                if a.source == "live":
                    raise SystemExit(str(e))
                print(f"{e}; falling back to Meta Kaggle (finished competitions only)")
        d = D.ensure(["Competitions.csv"], max_age_days=a.max_age, force=a.force)
        comp = D.competition_row(slug, d)
        D.forum_id(comp, slug)  # stop before the ~70 MB topic download when the export hides this forum
        D.ensure(["ForumTopics.csv"], max_age_days=a.max_age, force=a.force)
        return D.index(slug, d), comp, f"Meta Kaggle cache {d}"

    if a.action == "sync":
        t, comp, where = build_index()
        if a.messages:
            D.ensure(["ForumMessages.csv"], max_age_days=a.max_age, force=a.force)
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(idx_path, index=False)
        print(f"{slug}: {len(t)} topics -> {idx_path} ({where})")
        if t.attrs.get("truncated"):
            print(f"stopped at --max-pages {a.max_pages} (the {len(t)} most voted topics); raise it for the rest")
        if comp is None and (D.meta_dir() / "Competitions.csv").is_file():
            try:  # rule facts from an already cached export; never downloaded for this
                comp = D.competition_row(slug)
            except SystemExit:
                pass
        f = D.facts(comp) if comp else ""
        print("facts: " + f if f else "facts: `kaggle competitions list -s <slug>` (deadline), the rules page "
              "(`kgkit pages --page rules`) for merger deadline / team size")
        src = D.SOURCES[t["Source"].iloc[0]] if len(t) else "kaggle-cli"
        print("\nmost voted (" + SF.UNTRUSTED_NOTE + "):\n"
              + SF.wrap(D.format_table(D.top(t, min(a.n, 10), "votes")), src, competition=slug))
        return
    if a.action == "read":
        if not a.query:
            raise SystemExit("usage: discussions read <topic_id>")
        tid = int(a.query)
        url = f"https://www.kaggle.com/competitions/{slug}/discussion/{tid}"
        msgs, src = [], "meta-kaggle"
        if a.source != "meta":
            try:
                msgs, src = D.live_thread(slug, tid), "kaggle-cli"
            except D.LiveUnavailable as e:
                if a.source == "live":
                    raise SystemExit(str(e))
                print(f"{e}; trying the Meta Kaggle message cache")
        if not msgs:
            src = "meta-kaggle"
            try:
                msgs = D.read_thread(tid)
            except FileNotFoundError:
                print(f"ForumMessages.csv is not cached (`discussions sync --source meta --messages`, ~1.8 GB once)."
                      f"\nRead it at {url}")
                return
        if not msgs:
            print(f"topic {tid} not found (wrong competition, or newer than the last Meta Kaggle export). "
                  f"Read it at {url}")
            return
        print(SF.UNTRUSTED_NOTE)
        for m in msgs[: a.n if a.n else None]:
            medal = int(m["Medal"]) if pd.notna(m.get("Medal")) and m.get("Medal") else None
            print(SF.wrap(m["text"][:a.max_chars], src, topic=tid, posted=m.get("PostDate"), medal=medal))
        print(url)
        return
    if idx_path.is_file():
        t = pd.read_csv(idx_path, parse_dates=["Created"])
    else:
        t = build_index()[0]
    src = D.SOURCES.get(t["Source"].iloc[0], "meta-kaggle") if "Source" in t and len(t) else "meta-kaggle"
    if a.action == "top":
        out = D.top(t, a.n, a.sort)
    elif a.action == "search":
        if not a.query:
            raise SystemExit('usage: discussions search "<regex>"')
        out = D.search(t, a.query).head(a.n)
    else:  # solutions
        out = D.solutions(t).head(a.n)
        if out.empty:
            print("no write-up threads found (competition still running, or write-ups not in the export yet)")
            return
    print(SF.UNTRUSTED_NOTE)
    print(SF.wrap(D.format_table(out), src, competition=slug, view=a.action))
    print("\nread one: python -m kgkit discussions read <id>   (or WebFetch its URL: "
          f"https://www.kaggle.com/competitions/{slug}/discussion/<id>)")


def cmd_pages(a):
    from . import pages as P
    from . import safety as SF
    from . import state as S

    st = S.load()
    slug = a.competition or (st.slug if st else None)
    if not slug:
        raise SystemExit("--competition required (no competition state found)")
    pages = P.fetch(slug)
    if a.save:
        for f in P.save(pages, a.save):
            print(f"wrote {f}")
        return
    if not a.page:
        print(SF.UNTRUSTED_NOTE)
        print(SF.wrap(", ".join(f"{p['name']} ({len(p['text'])} chars)" for p in pages), "kaggle-cli",
                      competition=slug, view="page names"))
        print("\nread one: python -m kgkit pages --page <name>   (all: --save reports/pages)")
        return
    page = P.pick(pages, a.page)
    if page is None:
        raise SystemExit(f"no page {a.page!r}; pages: " + ", ".join(p["name"] for p in pages))
    text = page["text"]
    cut = len(text) > a.max_chars
    print(SF.UNTRUSTED_NOTE)
    print(SF.wrap(text[: a.max_chars], "kaggle-cli", competition=slug, page=page["name"]))
    if cut:
        print(f"(cut at {a.max_chars} of {len(text)} characters: --max-chars {len(text)} for all of it)")


def cmd_recheck(a):
    from .recheck import main_recheck

    main_recheck(a)


def cmd_validate(a):
    from .submission import format_report, validate_submission

    metric = a.metric
    if metric is None:
        from . import state as S

        st = S.load()
        metric = st.metric if st and st.metric else None
    res = validate_submission(a.sub, a.sample, a.id_col, a.prob_cols.split(",") if a.prob_cols else None, metric)
    print(format_report(res))
    sys.exit(0 if res["ok"] else 1)


def cmd_vendor(a):
    dest = Path(a.dest) / "kgkit"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(PLUGIN_ROOT / "kgkit", dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"copied kgkit to {dest}")


def cmd_datasets(a):
    from . import datasets as DS

    res = DS.pull(a.dataset, a.dest, files=a.files, per_file=a.per_file, force=a.force, unzip=not a.no_unzip)
    if res["mode"] == "whole":
        print(f"downloaded {a.dataset} -> {res['dest']}")
        return
    print(f"downloaded {len(res['files'])} file(s) of {a.dataset} -> {res['dest']} (one request per file)")
    for f in res["failed"]:
        print(f"  FAILED {f}")
    if res["failed"]:
        sys.exit(1)


def cmd_whoami(a):
    from . import gpu as G

    user = G.kaggle_username()
    if not user:
        raise SystemExit("unknown: set KAGGLE_USERNAME, or authenticate the Kaggle CLI (skill kaggle-cli)")
    print(user)


def cmd_gpu(a):
    from . import gpu as G

    if a.gpu_cmd == "quota":
        print(G.quota_report(refresh=not a.cached))
    elif a.gpu_cmd == "plan":
        print(G.plan(refresh=not a.cached, deadline=a.deadline))
    elif a.gpu_cmd == "build":
        kdir = G.build_kernel(
            None, a.name, a.cmd, a.hours, folds=a.folds, accelerator=a.accelerator, user=a.user,
            datasets=a.dataset, kernels=a.kernel, models=a.model, resume_from=a.resume_from, include=a.include,
            internet=not a.no_internet, pip=a.pip, wheel_sources=a.wheels, assemble_cmd=a.assemble_cmd,
            post_cmds=a.post_cmd, prune_against=a.prune_against, prune_margin=a.prune_margin,
            prune_min_frac=a.prune_min_frac, slug=a.slug, public=a.public,
            extra_jobs=[tuple(j) for j in a.extra_job or []])
        tm = json.loads((kdir / "kg-train.json").read_text(encoding="utf-8"))
        print(f"wrote {kdir} (kernel {tm['kernel']}, jobs {tm['jobs']}, {tm['accelerator']}, cap {tm['hours']}h, "
              f"folds {tm['folds']}, code bundle {tm['bundle_kb']} KB)")
        if tm.get("prune"):
            print(f"pruning against {tm['prune']['baseline']} with margin {tm['prune']['margin']:.5f}")
        print(f"next: smoke-test the command locally, then  python -m kgkit gpu push {kdir.as_posix()}")
    elif a.gpu_cmd == "push":
        rec = G.push(None, a.dir, force=a.force, hours=a.hours, check_sources=not a.no_source_check,
                     check_sessions=not a.no_session_check)
        built = rec.get("jobs") != ["notebook"]
        print(f"pushed {rec['kernel']} v{rec['version']} ({rec['accelerator']}, cap {rec['hours']}h). Next: "
              f"python -m kgkit gpu wait {rec['kernel']}  (run it in the background)"
              + (", then gpu collect" if built else "; `kaggle kernels output` for its files"))
    elif a.gpu_cmd == "status":
        root = G._root(None)
        kernels = a.kernel or list(dict.fromkeys(r["kernel"] for r in G.runs(root) if r.get("status") in G.ACTIVE))
        for k in kernels:
            s, text = G.kernel_status(k)
            G.note_status(root, k, s)
            print(f"{k}: {s}" + (f"  ({text})" if s == "unknown" else ""))
        if not kernels:
            print("no active remote runs")
    elif a.gpu_cmd == "release":
        root = G._root(None)
        n = G.release(root, a.kernel)
        print(f"{a.kernel}: {n} reservation(s) released" if n else f"{a.kernel}: nothing in flight in the run log")
        print(G.quota_report(root, refresh=False))
    elif a.gpu_cmd == "cancel":
        s, _ = G.kernel_status(a.kernel)
        print(G.cancel_advice(a.kernel, s))
    elif a.gpu_cmd == "wait":
        s = G.wait(None, a.kernel, interval=a.interval, max_hours=a.max_hours)
        if s == "complete" and a.collect:
            G.collect(None, a.kernel)
        sys.exit(0 if s == "complete" else 1)
    elif a.gpu_cmd == "collect":
        G.collect(None, a.kernel, from_dir=a.from_dir, weights=a.weights)
    elif a.gpu_cmd == "log":
        print(G.run_table(n=a.n))


# ---------------------------------------------------------------------------- parser
def _validation_args(s) -> None:
    s.add_argument("--validation", default="kfold", choices=["kfold", "holdout+lb"],
                   help="kfold: OOF over data/folds.csv; holdout+lb: a fixed labelled holdout plus the public LB "
                        "(no OOF files, e.g. a small expert-labelled set in a code competition)")
    s.add_argument("--holdout-n", type=int, help="holdout+lb: rows in the labelled holdout")
    s.add_argument("--eval-truth", help="FILE:COL[,COL...] labels of the evaluation set (holdout+lb: the holdout "
                                        "labels, one column per label of a multilabel target)")
    s.add_argument("--eval-groups", help="FILE:COL unit resampled whole when measuring noise (patient, study, query)")
    s.add_argument("--public-lb-n", type=int, help="rows in the public LB split (noise projected to its size)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kgkit", description="Kaggle Grandmaster toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create a competition workspace")
    s.add_argument("slug")
    s.add_argument("--dir", default=".")
    s.add_argument("--metric")
    s.add_argument("--lower-is-better", action="store_true")
    s.add_argument("--task", choices=["tabular", "cv", "nlp", "llm", "timeseries", "audio", "rl", "optimization",
                                      "recsys", "other"])
    s.add_argument("--target")
    s.add_argument("--id-col")
    s.add_argument("--code-competition", action="store_true")
    s.add_argument("--runtime-hours", type=float)
    s.add_argument("--no-internet", action="store_true")
    s.add_argument("--daily-subs", type=int, default=5)
    s.add_argument("--deadline")
    _validation_args(s)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("adopt", help="adopt an existing project: write .kaggle-gm/ only, import past submissions")
    s.add_argument("slug", nargs="?", help="default: detected from kernel-metadata.json / README competition URLs")
    s.add_argument("--dir", default=".")
    s.add_argument("--metric")
    s.add_argument("--lower-is-better", action="store_true")
    s.add_argument("--task", choices=["tabular", "cv", "nlp", "llm", "timeseries", "audio", "rl", "optimization",
                                      "recsys", "other"])
    s.add_argument("--target")
    s.add_argument("--id-col")
    s.add_argument("--code-competition", action="store_true")
    s.add_argument("--runtime-hours", type=float)
    s.add_argument("--no-internet", action="store_true")
    s.add_argument("--daily-subs", type=int, default=5)
    s.add_argument("--deadline")
    _validation_args(s)
    s.add_argument("--force", action="store_true", help="rewrite an existing competition.json")
    s.add_argument("--submissions-file", help="saved `kaggle competitions submissions <slug> --format json` output")
    s.add_argument("--no-submissions", action="store_true", help="do not import the submission history")
    s.set_defaults(fn=cmd_adopt)

    s = sub.add_parser("config", help="show or set competition.json fields, e.g. validation=holdout+lb holdout_n=58")
    s.add_argument("set", nargs="*", metavar="KEY=VALUE")
    s.set_defaults(fn=cmd_config)

    s = sub.add_parser("status", help="workspace summary")
    s.add_argument("--dir", default=".")
    s.add_argument("-n", type=int, default=10)
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("folds", help="assign CV folds")
    s.add_argument("train")
    s.add_argument("--strategy", default="auto")
    s.add_argument("--target", help="column, or comma-separated columns for multilabel")
    s.add_argument("--group")
    s.add_argument("--n-splits", type=int, default=5)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--id-col")
    s.add_argument("--out", default="folds.csv")
    s.set_defaults(fn=cmd_folds)

    s = sub.add_parser("eda", help="quick EDA report")
    s.add_argument("train")
    s.add_argument("--test")
    s.add_argument("--target")
    s.add_argument("--id-col")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_eda)

    s = sub.add_parser("adv", help="adversarial validation")
    s.add_argument("train")
    s.add_argument("test")
    s.add_argument("--drop", help="comma-separated columns to exclude (ids, target)")
    s.add_argument("--n-splits", type=int, default=5)
    s.add_argument("--save-p", help="write per-train-row p(test) to this csv")
    s.set_defaults(fn=cmd_adv)

    s = sub.add_parser("score", help="score predictions")
    s.add_argument("metric")
    s.add_argument("truth", help="FILE:COLUMN[,COLUMN...]")
    s.add_argument("pred", help="FILE:COLUMN[,COLUMN...] or .npy")
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("metrics", help="list metrics")
    s.set_defaults(fn=cmd_metrics)

    s = sub.add_parser("ledger", help="experiment ledger")
    s.add_argument("action", choices=["list", "best", "show", "lb", "add", "compare", "noise", "decide", "baseline",
                                      "lineage", "import"])
    s.add_argument("args", nargs="*")
    s.add_argument("-n", type=int, default=15)
    s.add_argument("--parent", help="baseline experiment this one was compared against")
    s.add_argument("--truth", help="FILE:COL[,COL...] target (default: eval_truth; compare: paired bootstrap on the "
                                   "stored predictions; import/add --pred: the score)")
    s.add_argument("--folds", help="FILE:COLUMN fold ids (per-fold paired scores; import: fold scores)")
    s.add_argument("--fold", type=int, help="compare: restrict to one fold (1-fold screens)")
    s.add_argument("--groups", help="compare: FILE:COLUMN groups to resample whole (patients, sessions; default "
                                    "eval_groups)")
    s.add_argument("--boot", type=int, help="compare/noise: bootstrap resamples (default 1000, 300 above 20k rows)")
    s.add_argument("--method", default="auto", choices=["auto", "bootstrap"],
                   help="compare/noise: how noise is measured")
    s.add_argument("--public-n", type=int, help="compare/noise: rows in the public LB (default public_lb_n): noise "
                                                "projected to the LB's sample size")
    s.add_argument("--metric")
    s.add_argument("--test", help="import: test predictions (.npy or FILE:COLUMN)")
    s.add_argument("--cv", type=float, help="import: reported CV when no --truth is given")
    s.add_argument("--source", default="external", help="import: teammate | notebook | external")
    s.add_argument("--notes")
    s.add_argument("--tag", action="append", help="add: tag the record (lb-blend: blended with public files, "
                                                  "kept out of the CV-LB correlation); repeatable")
    s.add_argument("--pred", help="add: FILE:COL[,COL...] or .npy predictions on the evaluation set (holdout+lb: "
                                  "the holdout; else OOF); the score is recomputed and the predictions stored")
    s.add_argument("--variant", help="add: name shared by re-runs of one pipeline with other seeds (seed noise)")
    s.set_defaults(fn=cmd_ledger)

    s = sub.add_parser("backlog", help="ranked idea backlog")
    s.add_argument("action", choices=["add", "list", "set", "render", "fidelity"])
    s.add_argument("args", nargs="*")
    s.add_argument("--gain", type=float, help="expected gain 1-5 (1 = noise level, 5 = leak/new data scale)")
    s.add_argument("--prob", type=float, help="probability it works, 0-1")
    s.add_argument("--cost", type=float, help="hours of hands-on work (incl. local runs you wait for)")
    s.add_argument("--gpu-hours", type=float, help="unattended Kaggle GPU hours (ranked against the quota left "
                                                   "before the deadline, not as human time)")
    s.add_argument("--evidence")
    s.add_argument("--source", help="error-analysis | eda | forum | notebook | prior-art | brainstorm | domain")
    s.add_argument("--first-experiment")
    s.add_argument("--status", choices=["todo", "running", "done", "dropped"])
    s.add_argument("--exp", help="ledger id of an experiment testing this idea")
    s.add_argument("--result")
    s.add_argument("--screen-gain", type=float, help="gain measured by the cheap screen")
    s.add_argument("--full-gain", type=float, help="gain measured by the full CV run")
    s.add_argument("--all", action="store_true", help="list: include done/dropped")
    s.add_argument("-n", type=int, default=30)
    s.add_argument("--out", help="render: output path (default reports/backlog.md)")
    s.set_defaults(fn=cmd_backlog)

    s = sub.add_parser("features", help="automated feature search")
    s.add_argument("action", choices=["search"])
    s.add_argument("train")
    s.add_argument("--folds", required=True, help="FILE:COLUMN fold ids, same row order as train")
    s.add_argument("--target")
    s.add_argument("--truth", help="FILE:COLUMN target if not a column of train")
    s.add_argument("--test", help="test file (pooled for count / groupby statistics)")
    s.add_argument("--id-col")
    s.add_argument("--drop", help="comma-separated columns to ignore")
    s.add_argument("--cat", help="categorical columns (default: non-numeric)")
    s.add_argument("--num", help="numeric columns (default: numeric)")
    s.add_argument("--kinds", default="cnt,te,grp,num2")
    s.add_argument("--aggs", default="mean,std,nunique,diff_mean,rank")
    s.add_argument("--metric")
    s.add_argument("--batch", type=int, default=40)
    s.add_argument("--max-candidates", type=int, default=3000)
    s.add_argument("--max-batches", type=int)
    s.add_argument("--minutes", type=float, help="time budget")
    s.add_argument("--screen-folds", help="e.g. 0,1: screen on a subset of folds for speed")
    s.add_argument("--model", default="auto", choices=["auto", "lgbm", "hgb"])
    s.add_argument("--recheck-seed", type=int, help="re-check the kept set on folds re-drawn with this seed")
    s.add_argument("--group", help="group column for the recheck split")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="reports/featsearch.json")
    s.set_defaults(fn=cmd_features)

    s = sub.add_parser("discussions", help="competition forum, live via the Kaggle CLI or from Meta Kaggle")
    s.add_argument("action", choices=["sync", "top", "search", "solutions", "read"])
    s.add_argument("query", nargs="?", help="search: regex on titles; read: topic id")
    s.add_argument("--competition", help="default: competition.json slug (any slug works, e.g. a past competition)")
    s.add_argument("--source", default="auto", choices=["auto", "live", "meta"],
                   help="live: Kaggle CLI 2.2.4+ (any competition); meta: Meta Kaggle export (finished competitions "
                        "only, has view counts and medals); auto: live, else Meta Kaggle")
    s.add_argument("--max-pages", type=int, default=50, help="live sync: pages of 20 topics, most voted first")
    s.add_argument("-n", type=int, default=20)
    s.add_argument("--sort", default="votes", choices=["votes", "recent", "replies", "views"])
    s.add_argument("--messages", action="store_true",
                   help="sync: also cache Meta Kaggle's ForumMessages.csv (~1.8 GB) for offline `read`")
    s.add_argument("--max-age", type=float, default=3, help="refresh cached files older than this many days")
    s.add_argument("--force", action="store_true", help="re-download the cached files")
    s.add_argument("--max-chars", type=int, default=4000, help="read: characters per message")
    s.set_defaults(fn=cmd_discussions)

    s = sub.add_parser("pages", help="competition pages (evaluation, rules, data description) as text")
    s.add_argument("--competition", help="default: competition.json slug")
    s.add_argument("--page", help="page name, e.g. evaluation, rules, data-description (a unique prefix works)")
    s.add_argument("--save", metavar="DIR", help="write every page to DIR/<name>.md")
    s.add_argument("--max-chars", type=int, default=20000)
    s.set_defaults(fn=cmd_pages)

    s = sub.add_parser("recheck", help="does the baseline's gain survive a re-drawn fold split?")
    s.add_argument("--seed", type=int, default=7, help="seed for the re-drawn split")
    s.add_argument("--run", action="append", default=[],
                   help="shell command training one pipeline (old first, current last); runs with KG_FOLDS_FILE set")
    s.add_argument("--existing", nargs=2, metavar=("OLD", "NEW"), help="use already-run recheck records instead of --run")
    s.add_argument("--exp", nargs=2, metavar=("OLD", "NEW"), help="the same pipelines' records on the frozen split")
    s.add_argument("--truth", help="FILE:COLUMN: add a paired OOF bootstrap")
    s.add_argument("--metric")
    s.add_argument("--boot", type=int, default=300)
    s.add_argument("--strategy", help="override the recorded fold strategy (e.g. a seeded one)")
    s.add_argument("--target")
    s.add_argument("--group")
    s.add_argument("--n-splits", type=int)
    s.add_argument("--train")
    s.add_argument("--id-col")
    s.set_defaults(fn=cmd_recheck)

    s = sub.add_parser("kernels", help="study and reproduce public notebooks")
    s.add_argument("action", choices=["top", "pull", "review"])
    s.add_argument("ref", nargs="?", help="owner/slug (pull, review)")
    s.add_argument("--competition", help="default: competition.json slug")
    s.add_argument("-n", type=int, default=20)
    s.add_argument("--sort-by", default="voteCount", help="voteCount | scoreDescending | dateRun | hotness")
    s.add_argument("--dest", default="ref")
    s.add_argument("--check-access", action="store_true", help="check attached datasets are accessible")
    s.set_defaults(fn=cmd_kernels)

    s = sub.add_parser("blend", help="blend ledger experiments")
    s.add_argument("--exp", nargs="+", required=True, help="experiment ids")
    s.add_argument("--truth", required=True, help="FILE:COLUMN with the training target, same row order as OOFs")
    s.add_argument("--metric")
    s.add_argument("--method", choices=["hill", "weights", "rank", "stack", "residual"], default="hill")
    s.add_argument("--levels", type=int, default=1, help="stack: number of meta-model layers before the final blend")
    s.add_argument("--base", help="residual: the experiment whose errors stage 2 learns (default: best single)")
    s.add_argument("--prune", type=int, help="keep at most N models, dropping near-duplicates (corr > threshold)")
    s.add_argument("--corr-threshold", type=float, default=0.995)
    s.add_argument("--allow-negative", action="store_true")
    s.add_argument("--folds", help="FILE:COLUMN of fold ids for an honest nested blend score")
    s.add_argument("--name", default="blend")
    s.add_argument("--sample", help="sample_submission.csv to fill with the blended test prediction")
    s.add_argument("--pred-cols", help="comma-separated prediction columns (default: all but the first)")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_blend)

    s = sub.add_parser("validate", help="validate a submission file")
    s.add_argument("sub")
    s.add_argument("sample")
    s.add_argument("--id-col")
    s.add_argument("--prob-cols")
    s.add_argument("--metric", help="kgkit metric name (default: from competition.json)")
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("vendor", help="copy kgkit into DEST/kgkit")
    s.add_argument("dest")
    s.set_defaults(fn=cmd_vendor)

    s = sub.add_parser("datasets", help="download a Kaggle dataset (per-file fallback when the archive 404s)")
    s.add_argument("action", choices=["pull"])
    s.add_argument("dataset", help="owner/slug")
    s.add_argument("dest", help="folder, e.g. data/ext/<slug>")
    s.add_argument("--files", help="only files whose dataset path matches this glob (implies per-file)")
    s.add_argument("--per-file", action="store_true", help="skip the whole-dataset archive")
    s.add_argument("--force", action="store_true", help="download again even if present")
    s.add_argument("--no-unzip", action="store_true")
    s.set_defaults(fn=cmd_datasets)

    s = sub.add_parser("whoami", help="your Kaggle username (never the key or token)")
    s.set_defaults(fn=cmd_whoami)

    g = sub.add_parser("gpu", help="train on Kaggle GPUs/TPUs: quota, remote kernels, accelerator-hour accounting")
    gs = g.add_subparsers(dest="gpu_cmd", required=True)
    s = gs.add_parser("quota", help="live weekly GPU quota and reset countdown")
    s.add_argument("--cached", action="store_true", help="do not call the API; use the cached value")
    s = gs.add_parser("plan", help="GPU-hour budget until the deadline")
    s.add_argument("--cached", action="store_true")
    s.add_argument("--deadline", help="YYYY-MM-DD (default: competition.json)")
    s = gs.add_parser("build", help="write a remote training kernel for a local training command")
    s.add_argument("--name", required=True, help="experiment name (artifacts/<name>/ fold files)")
    s.add_argument("--cmd", required=True,
                   help='training command run from the project root; "{fold}" = one fold per process/GPU')
    s.add_argument("--hours", type=float, required=True, help="session cap (~1.3x the estimate, <= 12; TPU <= 9)")
    s.add_argument("--folds", type=int, nargs="+", help="folds to run (default: all in data/folds.csv)")
    s.add_argument("--accelerator", default="NvidiaTeslaT4",
                   help="NvidiaTeslaT4 (2x T4) | NvidiaTeslaP100 | TpuV5E8 (TPU quota, one process per chip) | none")
    s.add_argument("--user", help="Kaggle username (default: KAGGLE_USERNAME / kaggle.json / `kaggle config view`)")
    s.add_argument("--dataset", nargs="*", default=[], help="owner/slug datasets linked into data/")
    s.add_argument("--kernel", nargs="*", default=[], help="owner/slug kernel outputs linked into data/")
    s.add_argument("--model", nargs="*", default=[], help="Kaggle model sources")
    s.add_argument("--resume-from", nargs="*", default=[], help="previous training kernel(s) to resume from")
    s.add_argument("--include", nargs="*", default=["src"], help="code paths bundled into the kernel")
    s.add_argument("--no-internet", action="store_true")
    s.add_argument("--pip", nargs="*", default=[], help="extra packages to pip install in the session")
    s.add_argument("--wheels", nargs="*", default=[], help="datasets holding wheels for offline pip installs")
    s.add_argument("--assemble-cmd", help='default: --cmd with "--folds {fold}" replaced by "--assemble"')
    s.add_argument("--extra-job", nargs=2, action="append", metavar=("NAME", "CMD"),
                   help="another job in the same session (e.g. a second 1-fold screen for the other T4)")
    s.add_argument("--post-cmd", nargs="*", default=[], help="commands run after assembly (e.g. pseudo-labels)")
    s.add_argument("--prune-against", help="'best', a ledger id/name, or artifacts/<name>: stop losing runs early")
    s.add_argument("--prune-margin", type=float, help="default: the baseline's CV fold std")
    s.add_argument("--prune-min-frac", type=float, default=0.3)
    s.add_argument("--slug", help="kernel slug (default: <name>-train)")
    s.add_argument("--public", action="store_true")
    s = gs.add_parser("push", help="check quota, inputs and session slots, then push a kernel folder")
    s.add_argument("dir", help="a `gpu build` folder, or any kernel folder with kernel-metadata.json (needs --hours)")
    s.add_argument("--hours", type=float, help="session cap (-t); required for folders not made by `gpu build`")
    s.add_argument("--force", action="store_true",
                   help="push even if the cap exceeds the plannable quota, an input kernel is still running, or "
                        "the session slots look full")
    s.add_argument("--no-source-check", action="store_true", help="skip the check that input kernels have finished")
    s.add_argument("--no-session-check", action="store_true", help="skip the busy-session preflight")
    s = gs.add_parser("status", help="status of active remote runs")
    s.add_argument("kernel", nargs="*")
    s = gs.add_parser("wait", help="poll a kernel until it finishes")
    s.add_argument("kernel")
    s.add_argument("--interval", type=float, default=90)
    s.add_argument("--max-hours", type=float, default=13)
    s.add_argument("--collect", action="store_true", help="collect the outputs when it completes")
    s = gs.add_parser("collect", help="download outputs, import ledger records, account GPU hours")
    s.add_argument("kernel")
    s.add_argument("--from-dir", help="use an already-downloaded output folder")
    s.add_argument("--weights", action="store_true", help="also download model weights (*.pt etc.)")
    s = gs.add_parser("release", help="drop the quota reservation of a run that finished or never started")
    s.add_argument("kernel")
    s = gs.add_parser("cancel", help="how to stop a running or queued kernel (prints the steps; deletes nothing)")
    s.add_argument("kernel")
    s = gs.add_parser("log", help="remote GPU runs and their cost")
    s.add_argument("-n", type=int, default=20)
    g.set_defaults(fn=cmd_gpu)
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
