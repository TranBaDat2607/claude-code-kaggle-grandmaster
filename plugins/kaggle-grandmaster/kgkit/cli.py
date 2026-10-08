"""Command line interface: ``python -m kgkit <command>``.

Commands
  init        create a competition workspace (.kaggle-gm/, folders, CLAUDE.md)
  status      one-screen summary: competition, best CV, CV-LB correlation, recent experiments
  folds       write folds.csv with a leak-free strategy and print the fold report
  eda         markdown EDA report with red flags
  adv         adversarial validation (train vs test)
  score       score a prediction column against a truth column with a registered metric
  metrics     list registered metrics
  ledger      list | best | show | lb | add  — the experiment ledger
  blend       hill-climb / weight-optimise / rank-average ledger experiments' OOFs and write a submission
  validate    validate a submission against sample_submission
  vendor      copy kgkit into a project or a Kaggle dataset folder (for offline kernels)
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


# ---------------------------------------------------------------------------- commands
def cmd_init(a):
    from . import state as S

    root = Path(a.dir).resolve()
    st = S.CompetitionState(
        slug=a.slug, metric=a.metric or "", task=a.task or "", target=a.target or "", id_col=a.id_col or "",
        code_competition=a.code_competition, runtime_limit_hours=a.runtime_hours, internet_allowed=not a.no_internet,
        daily_submissions=a.daily_subs, deadline=a.deadline or "",
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


def cmd_status(a):
    from . import state as S
    from .experiment import Ledger

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
    print("; ".join(bits))
    print(f"experiments logged: {len(recs)}")
    if recs:
        b = led.best(1)[0]
        print(f"best CV: {b['cv']:.5f} ({b['id']})")
        with_lb = [r for r in recs if r.get("lb_public") is not None]
        if with_lb:
            bl = sorted(with_lb, key=lambda r: r["lb_public"], reverse=st.greater_is_better)[0]
            print(f"best public LB: {bl['lb_public']:.5f} ({bl['id']}, CV {bl['cv']:.5f})")
        corr = led.cv_lb_correlation()
        if corr:
            print(f"CV-LB correlation over {corr['n']} subs: pearson {corr['pearson']:.2f}, spearman {corr['spearman']:.2f}")
        print()
        print(led.table(a.n))


def cmd_folds(a):
    from .cv import assign_folds, fold_report

    df = _read(a.train)
    target = a.target.split(",") if a.target and "," in a.target else a.target
    folds = assign_folds(df, a.n_splits, a.strategy, target=target, group=a.group, seed=a.seed)
    df["fold"] = folds
    keep = [c for c in [a.id_col] if c and c in df.columns] + ["fold"]
    df[keep].to_csv(a.out, index=False)
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

    y, p = _col(a.truth), _col(a.pred)
    print(f"{a.metric}: {M.score(a.metric, y.to_numpy(), p.to_numpy()):.6f}")


def cmd_metrics(a):
    from . import metrics as M

    for m in M.list_metrics():
        print(f"{m.name:20s} {'max' if m.greater_is_better else 'min'}  {m.kind:8s} {m.description}")


def cmd_ledger(a):
    from .experiment import Ledger

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
        print(f"{r['id']}: CV {r['cv']:.5f} -> public LB {r['lb_public']}")
        corr = led.cv_lb_correlation()
        if corr:
            print(f"CV-LB correlation now: pearson {corr['pearson']:.2f} over {corr['n']} subs")
    elif a.action == "add":
        if len(a.args) < 2:
            raise SystemExit("usage: ledger add <name> <cv> [notes]")
        r = led.log(a.args[0], float(a.args[1]), notes=" ".join(a.args[2:]))
        print(f"logged {r['id']}")


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
    y = _col(a.truth).to_numpy()
    oofs = {e: led.load_oof(e) for e in a.exp}
    for e, o in oofs.items():
        if len(o) != len(y):
            raise SystemExit(f"OOF of {e} has {len(o)} rows but truth has {len(y)}")
    print("OOF correlation:\n" + E.oof_correlation(oofs).round(4).to_string() + "\n")
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


def cmd_validate(a):
    from .submission import format_report, validate_submission

    res = validate_submission(a.sub, a.sample, a.id_col, a.prob_cols.split(",") if a.prob_cols else None)
    print(format_report(res))
    sys.exit(0 if res["ok"] else 1)


def cmd_vendor(a):
    dest = Path(a.dest) / "kgkit"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(PLUGIN_ROOT / "kgkit", dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"copied kgkit to {dest}")


# ---------------------------------------------------------------------------- parser
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
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init)

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
    s.add_argument("truth", help="FILE:COLUMN")
    s.add_argument("pred", help="FILE:COLUMN")
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("metrics", help="list metrics")
    s.set_defaults(fn=cmd_metrics)

    s = sub.add_parser("ledger", help="experiment ledger")
    s.add_argument("action", choices=["list", "best", "show", "lb", "add"])
    s.add_argument("args", nargs="*")
    s.add_argument("-n", type=int, default=15)
    s.set_defaults(fn=cmd_ledger)

    s = sub.add_parser("blend", help="blend ledger experiments")
    s.add_argument("--exp", nargs="+", required=True, help="experiment ids")
    s.add_argument("--truth", required=True, help="FILE:COLUMN with the training target, same row order as OOFs")
    s.add_argument("--metric")
    s.add_argument("--method", choices=["hill", "weights", "rank"], default="hill")
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
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("vendor", help="copy kgkit into DEST/kgkit")
    s.add_argument("dest")
    s.set_defaults(fn=cmd_vendor)
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
