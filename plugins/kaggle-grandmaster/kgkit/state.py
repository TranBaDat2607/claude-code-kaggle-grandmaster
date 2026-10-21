"""Competition workspace state, stored in ``.kaggle-gm/competition.json``.

The plugin's hooks and commands read this file to know which competition you are in,
what the metric is and which direction is better, how many submissions you get per day
and whether it is a code competition with runtime limits. The submission log
(``.kaggle-gm/submissions.jsonl``) and the deadline countdown live here too, so the
session brief, the submit hooks and ``kgkit status`` count them the same way.

This module must stay standard-library only: hooks import it.
"""

from __future__ import annotations

import datetime as dt
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
    holdout_se: float | None = None   # legacy, hand-set: SE of one absolute holdout score, never of a difference
    # The evaluation set comparisons are measured on (kgkit.evalset): its labels as FILE:COL[,COL...] (holdout
    # labels, or the training target in kfold mode) and the unit rows are resampled by (patient, study, query).
    eval_truth: str = ""
    eval_groups: str = ""
    public_lb_n: int | None = None    # rows in the public LB split: projects noise to the LB's sample size
    rope: float | None = None         # gains within +/- rope are "practically equivalent"
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


# ---------------------------------------------------------------------------- submissions and deadline
def submissions_log(root: Path) -> Path:
    return Path(root) / STATE_DIR / "submissions.jsonl"


def today_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def submission_records(root: Path) -> list[dict]:
    p = submissions_log(root)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def submissions_today(root: Path, records: list[dict] | None = None) -> int:
    recs = submission_records(root) if records is None else records
    return sum(1 for r in recs if str(r.get("time_utc", "")).startswith(today_utc()) and r.get("status") != "failed")


def server_remaining_today(root: Path, records: list[dict] | None = None) -> int | None:
    """What Kaggle said was left today (UTC) after the latest logged submit: it counts the whole team and
    the website, which the local log cannot see. None when that submit printed no count (an older, larger
    count would be stale) or no submit was logged today."""
    recs = submission_records(root) if records is None else records
    for r in reversed(recs):
        if not str(r.get("time_utc", "")).startswith(today_utc()) or r.get("status") == "failed":
            continue
        left = r.get("remaining_today")
        return left if isinstance(left, int) else None
    return None


def submissions_line(root: Path, st: CompetitionState, records: list[dict] | None = None) -> str:
    """Today's submissions against the daily limit, and Kaggle's own count when the last submit printed one."""
    recs = submission_records(root) if records is None else records
    s = f"Submissions logged today (UTC): {submissions_today(root, recs)}/{st.daily_submissions}"
    left = server_remaining_today(root, recs)
    return s + (f"; Kaggle reported {left} left today for the team after the last submit" if left is not None else "")


def deadline_line(st: CompetitionState) -> str:
    """The deadline, days left and the endgame phase that applies."""
    if not st.deadline:
        return "deadline: unknown (`python -m kgkit config deadline=YYYY-MM-DD`)"
    try:
        d = dt.date.fromisoformat(st.deadline[:10])
    except ValueError:
        return f"deadline: {st.deadline}"
    days = (d - dt.datetime.now(dt.timezone.utc).date()).days
    phase = ("ENDED — run /kg-postmortem" if days < 0 else "FINAL DAYS — robustness + /kg-final" if days <= 3
             else "endgame — freeze ensemble design" if days <= 14 else "exploration")
    return f"deadline {d.isoformat()} ({days} days left; phase: {phase})"
