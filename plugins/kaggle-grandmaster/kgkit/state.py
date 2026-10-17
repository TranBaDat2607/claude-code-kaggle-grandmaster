"""Competition workspace state, stored in ``.kaggle-gm/competition.json``.

The plugin's hooks and commands read this file to know which competition you are in,
what the metric is and which direction is better, how many submissions you get per day
and whether it is a code competition with runtime limits.

This module must stay standard-library only: hooks import it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATE_DIR = ".kaggle-gm"
STATE_FILE = "competition.json"
VALIDATION_MODES = ("kfold", "holdout+lb")


@dataclass
class CompetitionState:
    slug: str
    metric: str = ""
    greater_is_better: bool = True
    task: str = ""                    # tabular | cv | nlp | llm | timeseries | audio | rl | optimization | other
    target: str = ""
    id_col: str = ""
    code_competition: bool = False
    runtime_limit_hours: float | None = None
    internet_allowed: bool = True
    gpu: str = ""
    daily_submissions: int = 5
    final_submissions: int = 2
    deadline: str = ""                # ISO date of the final submission deadline (UTC)
    team_merger_deadline: str = ""
    external_data_allowed: str = "unknown"
    # kfold: decisions on OOF over data/folds.csv. holdout+lb: a fixed labelled holdout (holdout_n rows,
    # e.g. a small expert-labelled set) plus the public LB; there are no OOF files to blend or compare.
    validation: str = "kfold"
    holdout_n: int | None = None
    holdout_se: float | None = None   # standard error of the holdout score, if measured (bootstrap)
    notes: list[str] = field(default_factory=list)

    @property
    def holdout(self) -> bool:
        return self.validation.startswith("holdout")

    def save(self, root: str | Path = ".") -> Path:
        d = Path(root) / STATE_DIR
        d.mkdir(parents=True, exist_ok=True)
        p = d / STATE_FILE
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return p


def find_root(start: str | Path = ".") -> Path | None:
    """Walk up from ``start`` to the directory that contains ``.kaggle-gm``."""
    p = Path(start).resolve()
    for d in [p, *p.parents]:
        if (d / STATE_DIR / STATE_FILE).is_file():
            return d
    return None


def load(start: str | Path = ".") -> CompetitionState | None:
    root = find_root(start)
    if root is None:
        return None
    data = json.loads((root / STATE_DIR / STATE_FILE).read_text(encoding="utf-8"))
    known = {k: v for k, v in data.items() if k in CompetitionState.__dataclass_fields__}
    return CompetitionState(**known)
