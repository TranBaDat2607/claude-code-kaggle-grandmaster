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
FINAL_DAYS = 3  # the endgame's last days: robustness checks and final picks (`deadline_line`, `kgkit.finals`)


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
    # Ledger ids chosen as finals: what the user intends (`kgkit config final_picks=`), or Kaggle's own selection once
    # `kgkit final` has read it back (then final_picks_confirmed holds when). Only a confirmed list silences the warning.
    final_picks: list[str] = field(default_factory=list)
    final_picks_confirmed: str = ""   # UTC time `kgkit final` read exactly final_picks from Kaggle; "" = intended only
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
    # What the work is about now ("stacking public OOFs", "final week"): every new ledger record carries it, and the
    # journey of the finished competition is told phase by phase (`kgkit ledger phase`, `kgkit ledger journey`).
    phase: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def holdout(self) -> bool:
        return self.validation.startswith("holdout")

    def save(self, root: str | Path = ".") -> Path:
        d = Path(root) / STATE_DIR
        d.mkdir(parents=True, exist_ok=True)
        p = d / STATE_FILE
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        ensure_state_ignores(root)
        return p


# Generated inside .kaggle-gm/ and never committed (the ledger, backlog and logs there are): what `kgkit gpu collect`
# downloads from a remote run (its artifacts, per-fold logs) and the kernel folders `kgkit gpu build` writes (the
# code is embedded in them as base64).
STATE_IGNORES = ("remote/", "build/")


def ensure_state_ignores(root: str | Path) -> list[str]:
    """Make ``.kaggle-gm/.gitignore`` name :data:`STATE_IGNORES`, keeping what it has. Returns the lines added.
    Never raises."""
    p = Path(root) / STATE_DIR / ".gitignore"
    try:
        text = p.read_text(encoding="utf-8") if p.is_file() else ""
        have = {line.strip().strip("/") for line in text.splitlines()}
        add = [x for x in STATE_IGNORES if x.strip("/") not in have]
        if add:
            head = "" if text else "# written by kgkit: generated here, never committed\n"
            sep = "\n" if text and not text.endswith("\n") else ""
            p.write_text(text + sep + head + "".join(f"{x}\n" for x in add), encoding="utf-8")
        return add
    except OSError:
        return []


def cache_dir(root: str | Path) -> Path:
    """``<root>/.kaggle-gm/cache/``: other users' text (the forum index); ignores itself, as .kaggle-gm/ is committed."""
    d = Path(root) / STATE_DIR / "cache"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / ".gitignore").exists():
        (d / ".gitignore").write_text("*\n", encoding="utf-8")
    return d


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


def kaggle_list(root: Path) -> Path:
    """Kaggle's list of the team's submissions with their public and private scores, as ``kgkit ledger sync-lb`` last
    read it (``kgkit ledger predictiveness`` reads it, linked to a ledger record or not)."""
    return Path(root) / STATE_DIR / "kaggle_submissions.json"


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


def sync_stamp(root: Path) -> Path:
    return Path(root) / STATE_DIR / "submissions_sync.json"


def write_sync_stamp(root: Path, error: str | None, added: int = 0) -> None:
    """Record that the log was checked against Kaggle's list (``kgkit.adopt.sync``). Never raises."""
    try:
        p = sync_stamp(root)
        p.parent.mkdir(parents=True, exist_ok=True)
        at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        p.write_text(json.dumps({"at": at, "error": error, "added": added}), encoding="utf-8")
    except OSError:
        pass


def synced_today(root: Path) -> str | None:
    """HH:MM (UTC) of today's successful check of the log against Kaggle's list, None when there was none."""
    try:
        d = json.loads(sync_stamp(root).read_text(encoding="utf-8"))
        at = str(d.get("at", ""))
        return at[11:16] if not d.get("error") and at.startswith(today_utc()) else None
    except (OSError, ValueError, AttributeError):
        return None


def submissions_line(root: Path, st: CompetitionState, records: list[dict] | None = None) -> str:
    """Today's submissions against the daily limit, Kaggle's own count when the last submit printed one, and where
    the number comes from: the hooks log only the submits they see, so a count is unchecked until
    ``kgkit status`` has compared the log with Kaggle's list."""
    recs = submission_records(root) if records is None else records
    s = f"Submissions logged today (UTC): {submissions_today(root, recs)}/{st.daily_submissions}"
    left = server_remaining_today(root, recs)
    s += f"; Kaggle reported {left} left today for the team after the last submit" if left is not None else ""
    at = synced_today(root)
    if at:
        return s + f"; checked against Kaggle's list at {at} UTC"
    return (s + "; local log only, not checked against Kaggle today (submits made by a script, on the website or "
            "by a teammate may be missing): `python -m kgkit status` checks")


def deadline_date(st: CompetitionState) -> dt.date | None:
    """The deadline's date, None when it is unset or not an ISO date."""
    try:
        return dt.date.fromisoformat(st.deadline[:10]) if st.deadline else None
    except ValueError:
        return None


def deadline_days(st: CompetitionState) -> int | None:
    """Days from today (UTC) to the deadline's date (0 on the day itself, negative after it), None when unknown."""
    d = deadline_date(st)
    return None if d is None else (d - dt.datetime.now(dt.timezone.utc).date()).days


def deadline_line(st: CompetitionState) -> str:
    """The deadline, days left and the endgame phase that applies."""
    if not st.deadline:
        return "deadline: unknown (`python -m kgkit config deadline=YYYY-MM-DD`)"
    d, days = deadline_date(st), deadline_days(st)
    if d is None:
        return f"deadline: {st.deadline}"
    phase = ("ENDED — run /kg-postmortem, then /kg-showcase to show the work" if days < 0
             else "FINAL DAYS — robustness + /kg-final" if days <= FINAL_DAYS
             else "endgame — freeze ensemble design" if days <= 14 else "exploration")
    when = f"ended {-days} day{'s' if days != -1 else ''} ago" if days < 0 else f"{days} days left"
    return f"deadline {d.isoformat()} ({when}; phase: {phase})"
