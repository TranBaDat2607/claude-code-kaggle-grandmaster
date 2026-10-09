"""Does the accepted baseline's gain survive a fresh fold split?

After dozens of keep/discard decisions on the same frozen folds, part of the measured progress is
selection noise: changes that happened to land positive on *these* folds. The test is to re-draw
the folds with the same strategy and a different seed, re-run the old and the current pipeline on
the new split, and compare. A real gain survives, while an overfit one shrinks or vanishes.

    python -m kgkit recheck --seed 7 \\
        --run "git stash && python src/train_gbdt.py --name root_rc; git stash pop" \\
        --run "python src/train_gbdt.py --name base_rc" \\
        --exp 0001 0042 --truth data/train.csv:target

Each ``--run`` is a shell command that trains one pipeline and logs one ledger record (the plugin's
templates do). It runs with ``KG_FOLDS_FILE`` pointing at the re-drawn split (``{folds}`` in the
command is replaced by its path too) and ``KG_RECHECK=<seed>``, so its record is tagged and never
becomes a baseline or "best CV". The first run is the reference (old pipeline), the last is the
current one. ``--exp OLD NEW`` gives the original-split records for the before/after report.

The split recipe comes from ``.kaggle-gm/folds_spec.json``, which ``kgkit folds`` writes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import state as S
from .compare import compare_folds, format_comparison, paired_bootstrap, verdict
from .experiment import Ledger


def folds_spec(root: Path) -> dict:
    p = root / S.STATE_DIR / "folds_spec.json"
    if not p.is_file():
        raise SystemExit("no .kaggle-gm/folds_spec.json: re-create the frozen folds once with `kgkit folds ...` "
                         "(it records the recipe), or pass --strategy/--target/--group to recheck")
    return json.loads(p.read_text(encoding="utf-8"))


def redraw(root: Path, spec: dict, seed: int) -> Path:
    """Write data/folds_recheck_s<seed>.csv with the frozen split's recipe and a new seed."""
    from .cv import assign_folds

    train = pd.read_csv(root / spec["train"]) if not Path(spec["train"]).is_absolute() else pd.read_csv(spec["train"])
    target = spec.get("target")
    target = target.split(",") if target and "," in target else target
    folds = assign_folds(train, int(spec.get("n_splits") or 5), spec.get("strategy") or "auto", target=target,
                         group=spec.get("group"), seed=seed)
    orig_path = root / (spec.get("out") or "data/folds.csv")
    if orig_path.is_file():
        orig = pd.read_csv(orig_path)["fold"].to_numpy()
        if len(orig) == len(folds) and (orig == folds).all():
            raise SystemExit(f"strategy {spec.get('strategy')!r} produced the identical split for seed {seed} "
                             "(a deterministic splitter such as GroupKFold): pass --strategy stratified_group "
                             "(seeded) for the recheck")
    out = root / "data" / f"folds_recheck_s{seed}.csv"
    df = pd.DataFrame({"fold": folds})
    id_col = spec.get("id_col")
    if id_col and id_col in train.columns:
        df.insert(0, id_col, train[id_col].to_numpy())
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def run_pipelines(root: Path, cmds: list[str], folds_path: Path, seed: int, kgkit_home: str | None = None) -> list[dict]:
    """Run each command on the re-drawn split; return the last ledger record each one added."""
    led = Ledger(root)
    env = dict(os.environ, KG_FOLDS_FILE=str(folds_path), KG_RECHECK=str(seed))
    home = kgkit_home or str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in [home, env.get("PYTHONPATH", "")] if p)
    env.setdefault("KGKIT_HOME", home)
    recs = []
    for i, cmd in enumerate(cmds):
        n0 = len(led.records())
        cmd = cmd.replace("{folds}", str(folds_path))
        print(f"[recheck] run {i + 1}/{len(cmds)}: {cmd}", flush=True)
        r = subprocess.run(cmd, shell=True, cwd=root, env=env)
        if r.returncode != 0:
            raise SystemExit(f"[recheck] command failed (exit {r.returncode}): {cmd}")
        new = led.records()[n0:]
        if not new:
            raise SystemExit(f"[recheck] the command logged nothing to the ledger: {cmd}")
        rec = new[-1]
        if not rec.get("recheck"):
            led.update(rec["id"], recheck=str(seed))
            rec = led.get(rec["id"])
        recs.append(rec)
        print(f"[recheck]   -> {rec['id']} CV {rec['cv']:.5f}", flush=True)
    return recs


def report(old_rc: dict, new_rc: dict, gib: bool, orig: tuple[dict, dict] | None = None, boot: dict | None = None) -> str:
    rc = compare_folds(new_rc["fold_scores"], old_rc["fold_scores"], gib) if (
        old_rc.get("fold_scores") and new_rc.get("fold_scores")
        and len(old_rc["fold_scores"]) == len(new_rc["fold_scores"])) else None
    v = verdict(rc, boot)
    lines = ["# Recheck on a fresh fold split", "", format_comparison(new_rc["id"], old_rc["id"], v)]
    if orig is not None:
        o_old, o_new = orig
        o = (o_new["cv"] - o_old["cv"]) * (1 if gib else -1)
        r_gain = v["gain"]
        kept = r_gain / o if o > 0 else float("nan")
        lines += ["", f"original split: {o_old['id']} -> {o_new['id']} gain {o:+.5f}",
                  f"fresh split:    gain {r_gain:+.5f} ({kept:.0%} of the original gain survived)" if o > 0
                  else f"fresh split:    gain {r_gain:+.5f}"]
        if o <= 0:
            lines.append("VERDICT: nothing to recheck - the current pipeline was not better than the old one on the "
                         "frozen split either (check the order: --exp OLD NEW, old --run first).")
        elif r_gain <= 0 or v["decision"] == "DISCARD":
            lines.append("VERDICT: the gain did NOT survive. The progress on the frozen folds was largely selection "
                         "noise: reverse-ablate the marginal KEEPs (`kgkit ledger lineage`).")
        elif kept < 0.5:
            lines.append("VERDICT: the gain shrank by more than half. The CV is optimistic: trust only the big steps "
                         "and reverse-ablate the marginal ones.")
        else:
            lines.append("VERDICT: the gain holds on an independent split.")
    return "\n".join(lines)


def main_recheck(a) -> None:
    root = S.find_root()
    if root is None:
        raise SystemExit("not inside a competition workspace")
    st = S.load(root)
    gib = st.greater_is_better if st else True
    have_spec = (root / S.STATE_DIR / "folds_spec.json").is_file()
    if not have_spec and not a.strategy:
        folds_spec(root)  # raises with instructions
    spec = dict(folds_spec(root)) if have_spec else {"train": "data/train.csv", "out": "data/folds.csv",
                                                      "id_col": st.id_col if st else None}
    for k in ("strategy", "target", "group", "n_splits", "train", "id_col"):  # explicit flags override the recipe
        if getattr(a, k, None):
            spec[k] = getattr(a, k)
    path = redraw(root, spec, a.seed)
    print(f"[recheck] re-drawn split: {path.relative_to(root)} (strategy {spec.get('strategy')}, seed {a.seed})")
    if not a.run and not a.existing:
        print("no --run commands: train the old and the current pipeline with KG_FOLDS_FILE pointing at this file, "
              "then rerun with --existing <old_id> <new_id>")
        return
    led = Ledger(root)
    if a.existing:
        old_rc, new_rc = led.get(a.existing[0]), led.get(a.existing[1])
    else:
        if len(a.run) < 2:
            raise SystemExit("need two --run commands: the old pipeline first, the current one last")
        recs = run_pipelines(root, a.run, path, a.seed)
        old_rc, new_rc = recs[0], recs[-1]
    boot = None
    if a.truth and old_rc.get("oof_path") and new_rc.get("oof_path"):
        from .cli import _truth

        boot = paired_bootstrap(_truth(a.truth), led.load_oof(new_rc["id"]), led.load_oof(old_rc["id"]),
                                (st.metric if st else None) or a.metric, n_boot=a.boot)
    orig = (led.get(a.exp[0]), led.get(a.exp[1])) if a.exp else None
    text = report(old_rc, new_rc, gib, orig, boot)
    print(text)
    out = root / "reports" / f"recheck_s{a.seed}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(root)}", file=sys.stderr)
