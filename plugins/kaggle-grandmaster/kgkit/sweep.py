"""Sweeps (#134): many variants of one pipeline, screened cheaply, the best confirmed by a full run.

``kgkit sweep new`` defines one: the baseline, a command template, the variants (``--variant NAME=ARGS``, or a
``--grid`` of flag values) and the rungs, cheapest first (``--rung screen=s30,epochs=3``: a screen of
``kgkit screen new`` and cheaper settings); the full run is always the last rung. ``kgkit sweep run`` runs it by
successive halving: each rung runs the survivors of the one before, compares each with that rung's *control* (the
baseline's pipeline run at the same fidelity, made once and reused) and promotes the best ``1/eta``; the last rung's
runs are compared with the baseline itself, as the best of every variant the sweep tried (``compare.best_of_n_z``),
so a lucky draw among many does not pass as a gain.

The command is any pipeline's: kgkit fills in ``{base}`` (the baseline's id), ``{sweep}``, ``{variant}``, ``{rung}``,
``{args}`` (the variant's arguments) and ``{rung_args}`` (the rung's settings as ``--key value``), and sets
``KG_SCREEN`` / ``KG_FIDELITY`` / ``KG_CONTROL_OF`` for the pipeline to honour (``kgkit.screen``) and ``KG_SWEEP``,
``KG_SWEEP_VARIANT``, ``KG_RUNG``, which ``Ledger.log`` records as ``sweep`` on the run. Each run's output goes to
``.kaggle-gm/sweeps/<name>/<variant>_r<rung>.log``; the state is saved after every run, so an interrupted sweep
continues where it stopped (and a run that logged before the interruption is not run again).
"""

from __future__ import annotations

import itertools
import json
import math
import os
import re
from pathlib import Path

from . import state as S

DIR = "sweeps"
ENV, VARIANT_ENV, RUNG_ENV = "KG_SWEEP", "KG_SWEEP_VARIANT", "KG_RUNG"
CONTROL = "control"
FIELDS = ("base", "sweep", "variant", "rung", "rung_args", "args")


# ---------------------------------------------------------------------------- definition
def _clean(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "v"


def variants(specs, grid) -> list[dict]:
    """``NAME=ARGS`` specs, then the product of ``--flag=v1,v2`` grid axes (named ``flagv1_otherv2``)."""
    out = []
    for spec in specs or ():
        name, sep, args = spec.partition("=")
        if not sep or not name.strip():
            raise ValueError(f"a variant is NAME=ARGS (e.g. lr3=--lr 3e-4), not {spec!r}")
        out.append({"name": _clean(name.strip()), "args": args.strip()})
    axes = []
    for spec in grid or ():
        flag, sep, values = spec.partition("=")
        if not sep or not values:
            raise ValueError(f"a grid axis is --flag=v1,v2 (e.g. --lr=1e-3,3e-4), not {spec!r}")
        flag = flag if flag.startswith("-") else f"--{flag}"
        axes.append([(flag, v.strip()) for v in values.split(",") if v.strip()])
    for combo in itertools.product(*axes) if axes else ():
        out.append({"name": _clean("_".join(f"{f.lstrip('-')}{v}" for f, v in combo)),
                    "args": " ".join(f"{f} {v}" for f, v in combo)})
    seen = set()
    for v in out:
        if v["name"] in seen:
            raise ValueError(f"the variant {v['name']!r} is named twice")
        if v["name"] == CONTROL:
            raise ValueError("'control' is the name of the baseline's run at each rung: name the variant otherwise")
        seen.add(v["name"])
    return out


def _value(raw: str):
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def rungs(specs) -> list[dict]:
    """``screen=s30,epochs=3`` per rung, cheapest first; the full run (no screen, no settings) is appended."""
    out = []
    for spec in specs or ():
        rung = {"screen": None, "settings": {}}
        for part in (p for p in spec.split(",") if p.strip()):
            key, sep, raw = part.partition("=")
            if not sep:
                raise ValueError(f"a rung is screen=<name>,<setting>=<value>... (e.g. screen=s30,epochs=3), not {spec!r}")
            key = key.strip()
            if key == "screen":
                rung["screen"] = raw.strip() or None
            else:
                rung["settings"][key] = _value(raw.strip())
        out.append(rung)
    full = {"screen": None, "settings": {}}
    if not out or out[-1] != full:
        out.append(full)
    return out


def rung_args(settings: dict) -> str:
    """A rung's settings as command-line flags: {"img_size": 256, "amp": true} -> '--img-size 256 --amp'."""
    out = []
    for k, v in settings.items():
        flag = "--" + str(k).replace("_", "-")
        if v is True:
            out.append(flag)
        elif v is False or v is None:
            continue
        else:
            out.append(f"{flag} {v}")
    return " ".join(out)


def survivors(runs: dict, eta: int = 3) -> list[str]:
    """The variants a rung promotes: the best ``ceil(n / eta)`` of those compared (by their gain over the rung's
    control), at least one."""
    done = [(name, r["gain"]) for name, r in runs.items()
            if r.get("status") == "done" and isinstance(r.get("gain"), (int, float)) and r["gain"] == r["gain"]]
    if not done:
        return []
    k = max(1, math.ceil(len(done) / max(1, int(eta))))
    return [name for name, _ in sorted(done, key=lambda t: -t[1])[:k]]


def random_checks(runs: dict, keep: list[str], n: int, seed: str) -> list[str]:
    """``n`` of a rung's pruned variants drawn at random (seeded by the sweep and the rung): promoted too, they show
    how the next rung ranks variants this rung's own pruning would have dropped."""
    import random

    pool = sorted(v for v, r in runs.items() if v not in keep and r.get("status") == "done"
                  and isinstance(r.get("gain"), (int, float)))
    if n <= 0 or not pool:
        return []
    return sorted(random.Random(seed).sample(pool, min(int(n), len(pool))))


MIN_PAIRS = 4  # fewer variants scored at two rungs say nothing about how a screen ranks
FAITHFUL_SPEARMAN, FAITHFUL_SIGN = 0.5, 0.7


def fidelity(st: dict) -> list[dict]:
    """Per pair of consecutive rungs: how the gains of the variants that ran at both agree (Spearman rank
    correlation, share of equal signs), and how many of them were random checks rather than survivors."""
    from .backlog import _spearman

    out = []
    for r in range(len(st["rungs"]) - 1):
        pairs = []
        for v, runs in st["runs"].items():
            a, b = (runs.get(str(r)) or {}).get("gain"), (runs.get(str(r + 1)) or {}).get("gain")
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                pairs.append((v, a, b))
        names = {v for v, _, _ in pairs}
        f = {"from": r, "to": r + 1, "n": len(pairs),
             "checks": sum(1 for v in (st.get("checks") or {}).get(str(r + 1)) or () if v in names)}
        if len(pairs) >= 2:
            f["sign_agreement"] = sum(1 for _, a, b in pairs if (a > 0) == (b > 0)) / len(pairs)
            f["spearman"] = _spearman([a for _, a, _ in pairs], [b for _, _, b in pairs])
        out.append(f)
    return out


def fidelity_lines(st: dict) -> list[str]:
    """``fidelity`` in words, with a warning when a rung ranks variants unlike the next one."""
    out = []
    for f in fidelity(st):
        head = (f"rung {f['from']} -> rung {f['to']} ({describe_rung(st, f['from'])} -> "
                f"{describe_rung(st, f['to'])})")
        if f["n"] < MIN_PAIRS:
            out.append(f"{head}: {f['n']} variant(s) scored at both, too few to measure how the cheap rung ranks "
                       f"(at least {MIN_PAIRS}; `--check 2` on `sweep new` promotes random pruned ones too)")
            continue
        rho = f["spearman"]
        out.append(f"{head}: {f['n']} variants, Spearman {rho:.2f}, sign agreement {f['sign_agreement']:.0%}"
                   + (f" ({f['checks']} random checks among them)" if f["checks"] else
                      " (survivors only: their range is cut, so this is biased low; `--check 2` measures it on "
                      "random pruned ones too)"))
        if not (rho == rho and rho >= FAITHFUL_SPEARMAN) or f["sign_agreement"] < FAITHFUL_SIGN:
            out.append(f"  WARNING: rung {f['from']} ({describe_rung(st, f['from'])}) ranks variants unlike rung "
                       f"{f['to']}: its pruning may drop the best one. Raise its fidelity (more rows or folds, fewer "
                       "cheap settings) before the next sweep relies on it")
    return out


# ---------------------------------------------------------------------------- state
def path(root, name: str) -> Path:
    return Path(root) / S.STATE_DIR / DIR / f"{name}.json"


def logs(root, name: str) -> Path:
    return Path(root) / S.STATE_DIR / DIR / name


def load(root, name: str) -> dict:
    p = path(root, name)
    if not p.is_file():
        known = sorted(q.stem for q in p.parent.glob("*.json")) if p.parent.is_dir() else []
        raise ValueError(f"no sweep {name!r} ({', '.join(known) or 'none yet'}: `kgkit sweep new <name> ...`)")
    return json.loads(p.read_text(encoding="utf-8"))


def save(root, st: dict) -> None:
    p = path(root, st["name"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def listed(root) -> list[dict]:
    d = Path(root) / S.STATE_DIR / DIR
    return [load(root, p.stem) for p in sorted(d.glob("*.json"))] if d.is_dir() else []


def new(root, name: str, base: str, cmd: str, variant_list: list[dict], rung_list: list[dict], eta: int = 3,
        base_args: str = "", check: int = 0) -> dict:
    """Define a sweep (see the module docstring). ``check``: at each rung, promote that many pruned variants at
    random too, so that ``fidelity`` measures the screen on a sample its own pruning did not choose."""
    if not name or _clean(name) != name:
        raise ValueError(f"a sweep name is one word of letters, digits, '.', '_' or '-', not {name!r}")
    if "{args}" not in cmd:
        raise ValueError("the command has no {args}: it is where each variant's arguments go (e.g. "
                         "--cmd 'python src/train_gbdt.py --parent {base} --name {sweep}_{variant} {rung_args} {args}')")
    unknown = sorted(set(re.findall(r"{(\w+)}", cmd)) - set(FIELDS))
    if unknown:
        raise ValueError(f"the command names {', '.join('{' + u + '}' for u in unknown)}: it may use "
                         + ", ".join("{" + f + "}" for f in FIELDS))
    if not variant_list:
        raise ValueError("a sweep needs variants: --variant NAME=ARGS or --grid --flag=v1,v2")
    if path(root, name).is_file():
        raise ValueError(f"a sweep {name!r} exists already (`kgkit sweep status {name}`): name the new one otherwise")
    st = {"name": name, "base": base, "cmd": cmd, "base_args": base_args, "eta": int(eta), "check": int(check),
          "variants": variant_list, "rungs": rung_list, "runs": {v["name"]: {} for v in variant_list},
          "controls": {}, "survivors": {"0": [v["name"] for v in variant_list]}, "checks": {}, "state": "rung 0"}
    save(root, st)
    return st


# ---------------------------------------------------------------------------- one run
def tag() -> dict | None:
    """The sweep run this process is (``KG_SWEEP``...), recorded on its ledger record; None outside a sweep."""
    name = os.environ.get(ENV, "").strip()
    if not name:
        return None
    rung = os.environ.get(RUNG_ENV, "").strip()
    return {"name": name, "variant": os.environ.get(VARIANT_ENV, "").strip() or None,
            "rung": int(rung) if rung.lstrip("-").isdigit() else None}


def command(st: dict, variant: str, rung: int) -> str:
    args = st.get("base_args", "") if variant == CONTROL else next(v["args"] for v in st["variants"]
                                                                  if v["name"] == variant)
    fields = {"base": st["base"], "sweep": st["name"], "variant": variant, "rung": str(rung),
              "rung_args": rung_args(st["rungs"][rung]["settings"]), "args": args}
    out = st["cmd"]
    for k, v in fields.items():
        out = out.replace("{" + k + "}", v)
    return re.sub(r"  +", " ", out).strip()


def environment(st: dict, variant: str, rung: int) -> dict:
    """The variables a run is given (only those that differ from a full run's)."""
    r = st["rungs"][rung]
    env = {ENV: st["name"], VARIANT_ENV: variant, RUNG_ENV: str(rung)}
    if r["screen"]:
        env["KG_SCREEN"] = r["screen"]
    if r["settings"]:
        env["KG_FIDELITY"] = json.dumps(r["settings"], sort_keys=True)
    if variant == CONTROL:
        env["KG_CONTROL_OF"] = st["base"]
    return env


def rung_fidelity(root, st: dict, rung: int) -> dict | None:
    """The fidelity a run at ``rung`` is recorded with (what ``Ledger.log`` writes): None for the full run."""
    from . import screen as SC

    r = st["rungs"][rung]
    out = {**(SC.load(root, r["screen"]).fidelity() if r["screen"] else {}), **r["settings"]}
    return out or None


def describe_rung(st: dict, rung: int) -> str:
    r = st["rungs"][rung]
    if rung == len(st["rungs"]) - 1:
        return "full run"
    parts = ([f"screen {r['screen']}"] if r["screen"] else []) + [f"{k}={v}" for k, v in r["settings"].items()]
    return ", ".join(parts)


def furthest(st: dict, variant: str) -> tuple[int, str]:
    """(the last rung a variant ran at, what became of it): 'pruned at rung 0', 'failed at rung 1', 'done',
    'running at rung 2' (still to run there)."""
    runs = st["runs"].get(variant, {})
    last = len(st["rungs"]) - 1
    for r in range(last + 1):
        if variant not in st["survivors"].get(str(r), []):
            return r - 1, f"pruned at rung {r - 1}"
        run = runs.get(str(r))
        if run is None:
            return r, f"to run at rung {r}"
        if run.get("status") != "done":
            return r, f"{run.get('status', 'failed')} at rung {r}"
    return last, "done"


# ---------------------------------------------------------------------------- running
def _tagged(led, name: str, variant: str, rung: int) -> dict | None:
    """The last record a run of this sweep, variant and rung logged."""
    hits = [r for r in led.records() if (r.get("sweep") or {}).get("name") == name
            and r["sweep"].get("variant") == variant and r["sweep"].get("rung") == rung]
    return hits[-1] if hits else None


def _env(st: dict, variant: str, rung: int, kgkit_home: str | None) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("KG_SCREEN", "KG_FIDELITY", "KG_CONTROL_OF", ENV, VARIANT_ENV, RUNG_ENV)}
    env.update(environment(st, variant, rung))
    home = kgkit_home or str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in [home, env.get("PYTHONPATH", "")] if p)
    env.setdefault("KGKIT_HOME", home)
    return env


def _shown(st: dict, variant: str, rung: int) -> str:
    env = " ".join(f"{k}='{v}'" if k == "KG_FIDELITY" else f"{k}={v}"
                   for k, v in environment(st, variant, rung).items())
    return f"{env} {command(st, variant, rung)}"


def _tail(p: Path, n: int = 5) -> list[str]:
    try:
        return p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def _execute(root: Path, jobs: list[dict], parallel: int, on_done) -> None:
    """Run the jobs, ``parallel`` at a time; ``on_done(job, exit code)`` in this thread as each one ends."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def go(job):
        job["log"].parent.mkdir(parents=True, exist_ok=True)
        with job["log"].open("w", encoding="utf-8", errors="replace") as f:
            f.write(f"$ {job['shown']}\n")
            f.flush()
            r = subprocess.run(job["cmd"], shell=True, cwd=root, env=job["env"], stdout=f, stderr=subprocess.STDOUT)
        return job, r.returncode

    if not jobs:
        return
    with ThreadPoolExecutor(max_workers=max(1, int(parallel))) as ex:
        for fut in as_completed([ex.submit(go, j) for j in jobs]):
            on_done(*fut.result())


def _quiet_compare(led, new_id: str, ref_id: str, tries: int) -> dict:
    import contextlib
    import io

    from .cli import exp_compare

    with contextlib.redirect_stdout(io.StringIO()):
        return exp_compare(led, new_id, ref_id, {"tries": tries}, fallback=True, hint=False) or {}


def scored(st: dict) -> int:
    """How many variants gave a score at the first rung: the field a finalist was picked from (a run that failed
    gave noise no chance)."""
    return sum(1 for v in st["variants"] if (st["runs"][v["name"]].get("0") or {}).get("record"))


def final_tries(led, st: dict, rec_id: str) -> int:
    """A finalist is the best of every variant the sweep scored (and of the baseline's other children)."""
    return max(scored(st), led.tries(rec_id, st["base"]))


def best_finalist(st: dict) -> str | None:
    """The finalist with the largest gain over the baseline: the one to keep (keeping several would make the last
    one kept the baseline, whatever their order)."""
    last = str(len(st["rungs"]) - 1)
    done = [(v, st["runs"][v][last]["gain"]) for v in st["runs"]
            if isinstance((st["runs"][v].get(last) or {}).get("gain"), (int, float))]
    return max(done, key=lambda t: t[1])[0] if done else None


def _finalists_note(st: dict) -> str:
    last = str(len(st["rungs"]) - 1)
    fin = [v for v in st["runs"] if (st["runs"][v].get(last) or {}).get("record")]
    best = best_finalist(st)
    if len(fin) < 2 or best is None:
        return ""
    others = ", ".join(st["runs"][v][last]["record"] for v in fin if v != best)
    return (f"{len(fin)} finalists: keep the best, {best} ({st['runs'][best][last]['record']}), if its verdict is "
            f"KEEP; the others lost to it (`exp finish <id> discard \"lost to {best} in sweep {st['name']}\"` for "
            f"{others})")


def _control(root, led, st: dict, rung: int, new_control: bool) -> str | None:
    """The control of ``rung``: this sweep's, else one on file at the same fidelity (unless ``new_control``)."""
    if st["controls"].get(str(rung)):
        return st["controls"][str(rung)]
    mine = _tagged(led, st["name"], CONTROL, rung)
    have = mine or (None if new_control else led.control(st["base"], {"fidelity": rung_fidelity(root, st, rung)}))
    if have is None:
        return None
    st["controls"][str(rung)] = have["id"]
    if mine is None:
        print(f"[sweep {st['name']}] rung {rung} ({describe_rung(st, rung)}): control {have['id']} on file, reused")
    return have["id"]


def _dry_run(st: dict, rung: int, jobs: list[str]) -> None:
    last, name = len(st["rungs"]) - 1, st["name"]
    print(f"[sweep {name}] rung {rung} ({describe_rung(st, rung)}): {len(jobs)} run(s)")
    for v in jobs:
        print(f"  {_shown(st, v, rung)}")
    n = len(st["survivors"][str(rung)])
    for r in range(rung + 1, last + 1):
        n = max(1, math.ceil(n / max(1, st["eta"])))
        print(f"[sweep {name}] rung {r} ({describe_rung(st, r)}): the best {n} of rung {r - 1}"
              + ("" if r == last else ", and their control"))


def _print_rung(led, st: dict, rung: int, ref: str, runs: dict, keep: list[str]) -> None:
    from .compare import fmt_gain

    checks = set((st.get("checks") or {}).get(str(rung + 1)) or ())

    ctl = led.get(ref)
    print(f"[sweep {st['name']}] rung {rung} ({describe_rung(st, rung)}): {len(runs)} run(s) vs control {ref}"
          + (f" (CV {ctl['cv']:.5f})" if isinstance(ctl.get("cv"), (int, float)) else ""))

    def order(v):
        g = runs[v].get("gain")
        return -g if isinstance(g, (int, float)) else math.inf

    for v in sorted(runs, key=order):
        r = runs[v]
        if r["status"] != "done":
            print(f"  {v:<16} {r['status']}")
            continue
        z = r.get("z")
        print(f"  {v:<16} {r['record']:<28} gain {fmt_gain(r['gain'])}"
              + (f" (z {z:.2f})" if isinstance(z, (int, float)) and z == z else "")
              + ((f"  -> rung {rung + 1}" + (" (check)" if v in checks else "")) if v in keep else "  pruned"))


def run(root, name: str, parallel: int = 1, one: bool = False, dry_run: bool = False, new_control: bool = False,
        kgkit_home: str | None = None) -> dict:
    """Run a sweep from where it stopped (see the module docstring); ``one``: one rung, then stop."""
    from .cli import exp_compare
    from .experiment import Ledger

    root = Path(root)
    st, led = load(root, name), Ledger(root)
    last = len(st["rungs"]) - 1
    if st["state"] == "done":
        print(f"sweep {name} is already done: `kgkit sweep status {name}`; `kgkit sweep decide {name}` records "
              "what became of each run")
        return st
    if not st["state"].startswith("rung "):
        st["state"] = f"rung {max(int(k) for k in st['survivors'])}"
    while True:
        rung = int(st["state"].split()[1])
        here = st["survivors"][str(rung)]
        jobs = []
        if rung < last and _control(root, led, st, rung, new_control) is None:
            jobs.append(CONTROL)
        for v in here:
            if str(rung) in st["runs"][v]:
                continue
            rec = _tagged(led, name, v, rung)
            if rec is not None:  # logged before an interruption: not run again
                st["runs"][v][str(rung)] = {"status": "done", "record": rec["id"], "cv": rec.get("cv")}
            else:
                jobs.append(v)
        if dry_run:
            _dry_run(st, rung, jobs)
            return st
        save(root, st)
        if jobs:
            print(f"[sweep {name}] rung {rung} ({describe_rung(st, rung)}): running {len(jobs)} ({parallel} at a "
                  f"time; logs in {logs(root, name).relative_to(root).as_posix()}/)", flush=True)

        def done(job, code, rung=rung):
            v, log = job["variant"], job["log"]
            rel = log.relative_to(root).as_posix()
            rec = _tagged(led, name, v, rung) if code == 0 else None
            if rec is None:
                print(f"  {v}: failed ({f'exit {code}' if code else 'it logged nothing to the ledger'}): {rel}",
                      flush=True)
                for line in _tail(log):
                    print(f"    | {line}")
                if v != CONTROL:
                    st["runs"][v][str(rung)] = {"status": "failed", "exit": code, "log": rel}
            else:
                if not rec.get("parent"):
                    led.update(rec["id"], parent=st["base"])
                cv = rec.get("cv")
                print(f"  {v}: {rec['id']}" + (f" CV {cv:.5f}" if isinstance(cv, (int, float)) else ""), flush=True)
                if v == CONTROL:
                    st["controls"][str(rung)] = rec["id"]
                else:
                    st["runs"][v][str(rung)] = {"status": "done", "record": rec["id"], "cv": cv, "log": rel}
            save(root, st)

        _execute(root, [{"variant": v, "cmd": command(st, v, rung), "shown": _shown(st, v, rung),
                         "env": _env(st, v, rung, kgkit_home), "log": logs(root, name) / f"{v}_r{rung}.log"}
                        for v in jobs], parallel, done)
        ref = st["controls"].get(str(rung)) if rung < last else st["base"]
        if ref is None:
            raise SystemExit(f"[sweep {name}] rung {rung}: the control failed, so no run of it can be compared: fix "
                             f"the command (its log is above), then `kgkit sweep run {name}` again")
        runs = {v: st["runs"][v][str(rung)] for v in here if str(rung) in st["runs"][v]}
        for v, r in runs.items():
            if r["status"] != "done" or "decision" in r:
                continue
            try:
                if rung < last:
                    res = _quiet_compare(led, r["record"], ref, len(here))
                else:
                    print(f"\n[sweep {name}] finalist {v} (picked from {scored(st)} variants):")
                    res = exp_compare(led, r["record"], ref, {"tries": final_tries(led, st, r["record"])},
                                      fallback=True) or {}
            except (SystemExit, ValueError, KeyError) as e:
                r.update(status="uncompared", why=str(e))
                print(f"  {v}: not compared with {ref}: {e}")
                continue
            r.update(gain=res.get("gain"), z=res.get("strength"), decision=str(res.get("decision") or "").upper())
            save(root, st)
        if rung == last:
            st["state"] = "done"
            save(root, st)
            note = _finalists_note(st)
            print(f"\n[sweep {name}] done. `kgkit sweep decide {name}` records the screens (pruned: discard, "
                  "promoted: inconclusive); decide the finalist with its `exp finish` call above"
                  + (f".\n{note}" if note else ""))
            if any(f["n"] >= MIN_PAIRS for f in fidelity(st)):
                print("\n" + "\n".join(fidelity_lines(st)))
            return st
        keep = survivors(runs, st["eta"])
        extra = random_checks(runs, keep, int(st.get("check") or 0), f"{name}:{rung}")
        st.setdefault("checks", {})[str(rung + 1)] = extra
        keep = keep + extra
        st["survivors"][str(rung + 1)] = keep
        _print_rung(led, st, rung, ref, runs, keep)
        if not keep:
            st["state"] = "stopped"
            save(root, st)
            raise SystemExit(f"[sweep {name}] no run of rung {rung} could be compared with its control: nothing to "
                             "promote (the logs are listed above)")
        st["state"] = f"rung {rung + 1}"
        save(root, st)
        if one:
            print(f"\nnext: `kgkit sweep run {name}` runs rung {rung + 1} ({describe_rung(st, rung + 1)}) with "
                  f"{', '.join(keep)}")
            return st


# ---------------------------------------------------------------------------- after a sweep
def decide(root, name: str) -> dict:
    """Record what became of each screen of a sweep: pruned at its rung (discard) or promoted (inconclusive). The
    finalists, full runs, are left to `exp finish`. Records decided already are left as they are."""
    from .cli import _finish_command
    from .experiment import Ledger

    st, led = load(root, name), Ledger(root)
    last, counts, finalists = len(st["rungs"]) - 1, {"discard": 0, "inconclusive": 0}, []
    for v in (x["name"] for x in st["variants"]):
        for r_key, run in sorted(st["runs"][v].items(), key=lambda t: int(t[0])):
            r = int(r_key)
            if not run.get("record"):
                continue
            rec = led.get(run["record"])
            if r == last:
                finalists.append((v, run, rec))
                continue
            nxt = st["survivors"].get(str(r + 1))
            if rec.get("decision") or nxt is None:
                continue
            gain = run.get("gain")
            how = (f"gain {gain:+.5g} vs control {st['controls'].get(r_key)}" if isinstance(gain, (int, float))
                   else run.get("status", ""))
            if v in nxt:
                led.decide(rec["id"], "inconclusive", f"sweep {name}: promoted to rung {r + 1} ({how})")
                counts["inconclusive"] += 1
            else:
                led.decide(rec["id"], "discard", f"sweep {name}: pruned at rung {r} ({how}; not in the best "
                                                 f"{len(nxt)} of {len(st['survivors'][r_key])})")
                counts["discard"] += 1
    print(f"sweep {name}: {counts['discard']} screen(s) recorded discard (pruned), {counts['inconclusive']} "
          "inconclusive (promoted)")
    best = best_finalist(st)
    for v, run, rec in finalists:
        if rec.get("decision"):
            print(f"  finalist {v}: {rec['id']} decided {rec['decision']} already")
            continue
        dec = (run.get("decision") or "inconclusive").lower()
        if len(finalists) > 1 and v != best:
            print(f"  finalist {v}: {rec['id']} lost to {best}: "
                  f"kgkit exp finish {rec['id']} discard \"lost to {best} in sweep {name}\"")
        else:
            print(f"  finalist {v}: {rec['id']} {dec.upper()} against {st['base']}: "
                  f"{_finish_command(led, rec['id'], st['base'], dec)}")
    blend = _blend_line(led, st, name, [rec for _, _, rec in finalists])
    if blend:
        print(blend)
    return counts


def _blend_line(led, st: dict, name: str, finalists: list[dict]) -> str | None:
    """The finalists are full runs with their OOF on file: one that lost on its own can still add to the ensemble.
    The `kgkit blend` call that tests them with the accepted ensemble (else the base), paired with the best accepted
    blend. Screens are not in it: their OOF covers the screen only."""
    from . import evalset as E
    from .cli import _blend_args
    from . import state as S

    full = [r["id"] for r in finalists if r.get("oof_path") and not E.partial(r)]
    if not full:
        return None
    recs = led.records()
    ens = E.track_heads(recs).get(E.ENSEMBLE)
    members = list(E.resolve_members(ens, recs)) if ens is not None else [led.get(st["base"])["id"]]
    ids = list(dict.fromkeys([*members, *full]))
    if len(ids) < 2:
        return None
    return (f"  the {len(full)} finalist(s) keep their OOF and test predictions: test them in the ensemble, paired "
            f"with the best accepted blend: `kgkit blend --exp {' '.join(ids)}{_blend_args(led, S.load(led.root))} "
            f"--name {name}_blend`")


def status_lines(root, name: str) -> list[str]:
    from .compare import fmt_gain

    st = load(root, name)
    last = len(st["rungs"]) - 1
    out = [f"sweep {name}: {len(st['variants'])} variant(s) of {st['base']}, eta {st['eta']}, {st['state']}"]
    for r in range(last + 1):
        ran = sum(1 for v in st["variants"] if str(r) in st["runs"][v["name"]])
        ref = f"control {st['controls'].get(str(r)) or '(none yet)'}" if r < last else f"vs {st['base']}"
        nxt = st["survivors"].get(str(r + 1))
        out.append(f"  rung {r}: {describe_rung(st, r):<28} {ref}: {ran} run"
                   + (f", {len(nxt)} promoted" if nxt is not None and r < last else ""))
    for v in (x["name"] for x in st["variants"]):
        cells = []
        for r in range(last + 1):
            run = st["runs"][v].get(str(r))
            if run is None:
                continue
            g = run.get("gain")
            cells.append(f"r{r} {fmt_gain(g)}" if isinstance(g, (int, float)) else f"r{r} {run.get('status')}")
            if r == last and run.get("decision"):
                cells[-1] += f" {run['decision']}"
        out.append(f"  {v:<16} {'  '.join(cells):<52} {furthest(st, v)[1]}")
    return out
