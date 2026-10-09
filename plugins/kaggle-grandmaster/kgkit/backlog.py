"""The idea backlog: ``.kaggle-gm/backlog.json``, ranked by (expected gain x probability) / cost.

Grandmasters never wonder what to run next: they keep a ranked list of hypotheses with the
evidence behind each, and they record what happened to every idea, so failed ideas are not
re-tried and their sources (error analysis, forum, prior art, brainstorm) can be judged.

    python -m kgkit backlog add "OOF target encoding of city x month" --gain 3 --prob 0.5 --cost 1 \\
        --evidence "error slice: city with < 50 rows, logloss 0.61 vs 0.48" --source error-analysis
    python -m kgkit backlog list
    python -m kgkit backlog set 4 --status done --exp 0012 --result "KEEP +0.0021" --screen-gain 0.003 --full-gain 0.0021
    python -m kgkit backlog fidelity      # do cheap 1-fold screens rank ideas like full CV does?
    python -m kgkit backlog render        # reports/backlog.md

``gain`` is on a 1-5 scale (1 = within noise, 3 = a solid step, 5 = a jump like a leak or a new
data source); ``prob`` is the chance it works at all; ``cost`` is hours of human + compute time.
Standard library only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import state as S

STATUSES = ("todo", "running", "done", "dropped")
FIELDS = ("idea", "evidence", "source", "gain", "prob", "cost", "status", "first_experiment", "result",
          "screen_gain", "full_gain")


def score(item: dict) -> float:
    cost = max(float(item.get("cost") or 1.0), 0.05)
    return float(item.get("gain") or 0) * float(item.get("prob") or 0) / cost


class Backlog:
    def __init__(self, root: str | Path | None = None):
        root = Path(root) if root is not None else (S.find_root() or Path.cwd())
        self.root = root
        self.path = root / S.STATE_DIR / "backlog.json"

    def items(self) -> list[dict]:
        if not self.path.is_file():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return data if isinstance(data, list) else []

    def _save(self, items: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, idea: str, gain: float = 2, prob: float = 0.3, cost: float = 1, evidence: str = "",
            source: str = "", first_experiment: str = "") -> dict:
        items = self.items()
        dup = next((i for i in items if i["idea"].strip().lower() == idea.strip().lower()), None)
        if dup is not None:
            return dup
        item = {"id": 1 + max([i["id"] for i in items] or [0]), "idea": idea, "evidence": evidence,
                "source": source, "gain": float(gain), "prob": float(prob), "cost": float(cost), "status": "todo",
                "first_experiment": first_experiment, "exp_ids": [], "result": "", "screen_gain": None,
                "full_gain": None, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        items.append(item)
        self._save(items)
        return item

    def set(self, item_id: int, exp: str | None = None, **fields: Any) -> dict:
        items = self.items()
        for i in items:
            if i["id"] == int(item_id):
                for k, v in fields.items():
                    if v is None:
                        continue
                    if k not in FIELDS:
                        raise ValueError(f"unknown backlog field {k!r}")
                    if k == "status" and v not in STATUSES:
                        raise ValueError(f"status must be one of {STATUSES}")
                    i[k] = float(v) if k in ("gain", "prob", "cost", "screen_gain", "full_gain") else v
                if exp and exp not in i["exp_ids"]:
                    i["exp_ids"].append(exp)
                i["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                self._save(items)
                return i
        raise KeyError(f"backlog item {item_id} not found")

    def ranked(self, statuses: tuple[str, ...] = ("todo", "running")) -> list[dict]:
        return sorted((i for i in self.items() if i.get("status") in statuses), key=score, reverse=True)

    def fidelity(self) -> dict | None:
        """Agreement between cheap screens and full CV on ideas that went through both.
        Low agreement means the screen (1 fold, low resolution, few epochs) is not a faithful proxy:
        screen at higher fidelity, or on two folds."""
        pairs = [(i["screen_gain"], i["full_gain"]) for i in self.items()
                 if i.get("screen_gain") is not None and i.get("full_gain") is not None]
        if len(pairs) < 3:
            return None
        sign = sum(1 for a, b in pairs if (a > 0) == (b > 0)) / len(pairs)
        return {"n": len(pairs), "sign_agreement": sign, "spearman": _spearman([a for a, _ in pairs],
                                                                                [b for _, b in pairs])}

    def table(self, include_closed: bool = False, n: int = 30) -> str:
        items = self.ranked(STATUSES if include_closed else ("todo", "running"))[:n]
        if not items:
            return "(backlog is empty: run /kg-ideas, or `kgkit backlog add`)"
        lines = ["| # | score | idea | gain | prob | cost | status | source | result |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for i in items:
            idea = i["idea"].replace("|", "/")[:70]
            lines.append(f"| {i['id']} | {score(i):.2f} | {idea} | {i['gain']:g} | {i['prob']:g} | {i['cost']:g} | "
                         f"{i['status']} | {i.get('source') or ''} | {(i.get('result') or '').replace('|', '/')[:40]} |")
        return "\n".join(lines)

    def render(self, out: str | Path | None = None) -> Path:
        out = Path(out) if out else self.root / "reports" / "backlog.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        open_items = self.ranked()
        closed = [i for i in self.items() if i.get("status") in ("done", "dropped")]
        parts = ["# Backlog", "", "Ranked by (gain x prob) / cost. Generated by `kgkit backlog render`; edit via the CLI.",
                 "", "## Open", "", self.table()]
        for i in open_items:
            if i.get("evidence") or i.get("first_experiment"):
                parts.append(f"- **#{i['id']}** evidence: {i.get('evidence') or '-'}; first experiment: "
                             f"{i.get('first_experiment') or '-'}")
        parts += ["", "## Tried", ""]
        parts += [f"- #{i['id']} {i['idea']} -> {i['status']}: {i.get('result') or '-'} "
                  f"(exps {', '.join(i.get('exp_ids') or []) or '-'})" for i in closed] or ["(none yet)"]
        fid = self.fidelity()
        if fid:
            parts += ["", f"Screen fidelity over {fid['n']} promoted ideas: sign agreement "
                          f"{fid['sign_agreement']:.0%}, spearman {fid['spearman']:.2f}"]
        out.write_text("\n".join(parts) + "\n", encoding="utf-8")
        return out


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def _spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _ranks(a), _ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")
