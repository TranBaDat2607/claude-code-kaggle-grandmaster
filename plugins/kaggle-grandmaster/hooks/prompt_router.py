"""UserPromptSubmit: route Kaggle-flavoured prompts to the right plugin skills.

Skill descriptions alone often don't make the model load a skill before answering, so the
checklists inside never reach the context. This hook detects competition work (Kaggle vocabulary
in the prompt, or a prompt typed inside a competition workspace), picks up to three relevant
skills by keyword and tells Claude to load them first. Unrelated prompts produce no output.

Only user-typed prompts are routed: harness-injected text (goal check-ins, task notifications) is
ignored. Each skill is suggested at most once per session (state kept in the temp dir, keyed by
session_id), and inside a workspace whose task is known, other domains' skills are not suggested.

A slash-only command (``disable-model-invocation``) cannot be loaded by Claude, so its route (HINTS) adds a line
telling Claude to suggest the command to the user instead.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

try:
    import _common as C
except Exception:  # pragma: no cover - fail open
    sys.exit(0)

HARNESS = re.compile(
    r"^\s*goal check-in\b|<task-notification>|\[SYSTEM NOTIFICATION|<system-reminder>|"
    r"^\s*<(command-name|command-message|local-command-stdout)>",
    re.IGNORECASE,
)

GATE = re.compile(
    r"\bkaggle\b|\bleaderboard\b|\b(public|private)\s+(lb|score|leaderboard)\b|\blb\b|\boof\b|out[- ]of[- ]fold|"
    r"\bcompetition\b|\bshake[- ]?up\b|\bgrandmaster\b|\bsubmission\b",
    re.IGNORECASE,
)

ROUTES: list[tuple[str, str]] = [
    ("kg-adopt", r"\badopt\b|existing (repo|repository|project|code ?base|workspace|pipeline)|mid[- ]competition|"
                 r"(already|been) (working|competing) on|weeks? into (the|this|a) competition|"
                 r"set ?up (the )?(plugin|kgkit|workspace|kaggle-gm) (for|in|on) (my|this|an?)\b"),
    ("cv-lb-debugging", r"(cv|validation|local)\b.{0,60}\b(lb|leaderboard)|(lb|leaderboard)\b.{0,60}\bcv\b|"
                        r"\bgap\b|doesn'?t (match|transfer|track)|disagree|too good|suspicious"),
    ("grandmaster-playbook", r"\bnoise\b|significan|fold[- ]std|(gain|improvement)s? (is|are|was|were) real|"
                             r"keep or discard|reverse ablation|overfit\w* (the |my )?cv|\bbacklog\b|what (to|should i) try"),
    ("validation-strategy", r"cross[- ]?validation|\bk-?fold|\bfolds?\b|\bsplit\b|\bholdout\b|\bleak|adversarial|"
                            r"\bgroup(ed)?\b|stratif"),
    ("ensembling", r"\bensembl|\bblend|\bstack(ing)?\b|hill[- ]?climb|\boof\b|out[- ]of[- ]fold|weights?\b"),
    ("leaderboard-boosters", r"pseudo[- ]?label|distill|\btta\b|test[- ]time aug|external data|\bpseudo\b"),
    ("metric-optimization", r"\bqwk\b|kappa|threshold|\bf1\b|f-?beta|\bauc\b|log ?loss|\brmsle?\b|\bmae\b|map@|"
                            r"ndcg|\bdice\b|\biou\b|rounding|calibrat|\bmetric\b"),
    ("final-submission-selection", r"final (submission|pick|selection)|which (two|2) (submission|should)|"
                                   r"\bselect(ion)?\b.{0,40}submission|shake[- ]?up|private (lb|leaderboard)"),
    ("kg-postmortem", r"post-?mortem|lessons? (learned|from (this|the) competition)|"
                      r"(competition|deadline) (is over|has (ended|passed)|ended|passed)|after the (competition|deadline)|"
                      r"final (rank|standing|result)s?\b"),
    ("code-competitions", r"code competition|internet (off|disabled)|offline|\bkernel\b|notebook submission|"
                          r"runtime limit|hidden test|\bwheels?\b"),
    ("tabular-mastery", r"lightgbm|\blgbm\b|xgboost|catboost|\bgbdt\b|tabular|feature engineering|playground|"
                        r"target encod|feature search|tabpfn|autogluon|groupby"),
    ("computer-vision", r"\bimages?\b|x-?ray|\bct\b|\bmri\b|segmentation|detection|\btimm\b|\byolo\b|convnext|"
                        r"efficientnet|\bvit\b|\bcnn\b|dicom"),
    ("nlp-and-llm", r"deberta|transformer|\bllm\b|\bnlp\b|\btext\b|essay|token|\blora\b|\bbert\b|vllm|prompt"),
    ("time-series", r"forecast|time[- ]series|\bsales\b|\blags?\b|rolling|temporal|horizon"),
    ("audio-and-signal", r"\baudio\b|spectrogram|\beeg\b|\becg\b|birdclef|\bsound\b|waveform"),
    ("simulation-and-optimization", r"simulation|\bagents?\b.{0,30}\b(game|environment)|\bsanta\b|\btsp\b|"
                                    r"combinatorial"),
    ("recsys-and-ranking", r"recommend|\branking\b|candidate generation|co-?visitation|"
                           r"session[- ]based|user sessions?|sessions? (ids?|features|length)"),
    ("hyperparameter-tuning", r"optuna|hyper-?parameter|(?<!fine-)(?<!fine )\btuning\b|search space"),
    ("deep-learning-training", r"learning rate|scheduler|mixed precision|\bamp\b|\bema\b|loss (curve|spike)|"
                               r"nan loss|batch size|gradient"),
    ("kaggle-gpu", r"\bgpu (quota|hours?|budget|time)|\bquota\b|\bp100\b|\bt4s?\b|accelerator|"
                   r"\b(train|run)\w*\b.{0,40}\bon kaggle\b|kaggle(?:'s)? (gpus?|notebooks?|kernels?)\b.{0,40}\btrain|"
                   r"remote(ly)? train|session (limit|timeout)"),
    ("competition-recon", r"public (notebooks?|kernels?)|top(-voted)? notebooks?|\bdiscussions?\b|write-?ups?|"
                          r"prior art|\brecon\b|reproduce (a|the|this) (notebook|kernel)|"
                          r"(competition|the) rules\b|rules page|evaluation page|data description"),
    ("competition-strategy", r"\bteam(ing|mates?| up)\b|\bmerg(e|er|ing)\b.{0,40}\bteams?\b|\bteams?\b.{0,40}\bmerg|"
                             r"which competitions?|grandmaster (title|tier|rank)|solo gold|\bgold medals?\b|"
                             r"join(ing|ed)? (late|the competition late)|several competitions"),
    ("kaggle-cli", r"kaggle (api|cli)|kaggle competitions|kaggle kernels|kaggle datasets|download (the )?data"),
]
COMPILED = [(name, re.compile(rx, re.IGNORECASE)) for name, rx in ROUTES]

# slash-only commands: (name, pattern, what to tell Claude)
HINTS: list[tuple[str, str, str]] = [
    ("kg-showcase",
     r"show ?(off|case)? (my|our|the) (work|solution|results?)|showcase|portfolio|"
     r"make (it|my|our|the|this) (repo|repository|notebook|code|work|solution|weights)? ?public|"
     r"publish (my|our|the) (solution|work|code|notebook|weights|repo)|(my|our) (kaggle )?write-?up|"
     r"write (a|my|our|the) (kaggle )?(solution )?write-?up|"
     r"(update|polish|tidy|clean up|rebuild) (my|our|the) (repo|repository|readme|project)\b",
     "To show the work after the competition (README rebuilt around the result, own work separated from public "
     "work, license, figures, a public notebook and weights, a Kaggle write-up draft), suggest the user run "
     "`/kg-showcase`, a slash command: it audits the repo first and asks before each public step. Make nothing "
     "public from here."),
]
COMPILED_HINTS = [(name, re.compile(rx, re.IGNORECASE), text) for name, rx, text in HINTS]

# competition.json `task` -> its domain skill; inside a workspace the other domains are not suggested
TASK_DOMAIN = {"tabular": "tabular-mastery", "cv": "computer-vision", "nlp": "nlp-and-llm", "llm": "nlp-and-llm",
               "timeseries": "time-series", "audio": "audio-and-signal", "rl": "simulation-and-optimization",
               "optimization": "simulation-and-optimization"}
DOMAIN_SKILLS = set(TASK_DOMAIN.values()) | {"recsys-and-ranking"}


def hints(prompt: str, in_workspace: bool) -> list[str]:
    """Slash-only commands to suggest for this prompt (HINTS)."""
    if not (GATE.search(prompt) or in_workspace):
        return []
    return [name for name, rx, _ in COMPILED_HINTS if rx.search(prompt)]


def route(prompt: str, in_workspace: bool, task: str = "") -> list[str]:
    if not (GATE.search(prompt) or in_workspace):
        return []
    own = TASK_DOMAIN.get(task.strip().lower())
    hits = []
    for name, rx in COMPILED:
        if own and name in DOMAIN_SKILLS and name != own:
            continue
        n = len(rx.findall(prompt))
        if n:
            hits.append((n, name))
    # keep ROUTES priority order among the top hits (debugging/validation before domain skills)
    order = {name: i for i, (name, _) in enumerate(ROUTES)}
    top = sorted(hits, key=lambda t: (-t[0], order[t[1]]))[:3]
    chosen = sorted((name for _, name in top), key=lambda n: order[n])
    if chosen or hints(prompt, in_workspace):  # a slash command to suggest is answer enough without the fallback
        return chosen
    return ["grandmaster-playbook"]


def _seen_file(payload: dict) -> Path | None:
    sid = re.sub(r"[^A-Za-z0-9_-]", "", str(payload.get("session_id") or ""))[:64]
    return Path(tempfile.gettempdir()) / "kaggle-gm-router" / f"{sid}.json" if sid else None


def _seen(path: Path | None) -> set[str]:
    if path is None:
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return set()


def _remember(path: Path | None, skills: set[str]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(skills)), encoding="utf-8")
    except OSError:
        pass


def _task(root: Path | None) -> str:
    if root is None:
        return ""
    try:
        state = C.S.load(root)
    except (OSError, ValueError, TypeError):
        return ""
    return state.task if state else ""


def main() -> None:
    payload = C.read_input()
    prompt = payload.get("prompt") or ""
    if not prompt.strip() or prompt.lstrip().startswith("/"):
        return  # slash commands already carry their own instructions
    if HARNESS.search(prompt):
        return  # goal check-ins and task notifications are not the user asking
    root = C.workspace(payload)
    seen_file = _seen_file(payload)
    seen = _seen(seen_file)
    skills = [s for s in route(prompt, root is not None, _task(root)) if s not in seen]
    suggest = [h for h in hints(prompt, root is not None) if f"hint:{h}" not in seen]
    if not (skills or suggest):
        return
    _remember(seen_file, seen | set(skills) | {f"hint:{h}" for h in suggest})
    lines = []
    if skills:
        names = ", ".join(f"kaggle-grandmaster:{s}" for s in skills)
        lines.append(f"[kaggle-grandmaster] This is competition work. Before answering, load these skills with the "
                     f"Skill tool and follow their checklists: {names}. Ground recommendations in them (validation "
                     f"first, leakage mechanisms, honest OOF-based estimates), not just general ML advice.")
    text = {name: t for name, _, t in COMPILED_HINTS}
    lines += [f"[kaggle-grandmaster] {text[h]}" for h in suggest]
    C.emit("UserPromptSubmit", additionalContext="\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
