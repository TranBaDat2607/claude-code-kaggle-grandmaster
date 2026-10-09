"""UserPromptSubmit: route Kaggle-flavoured prompts to the right plugin skills.

Skill descriptions alone often don't make the model load a skill before answering, so the
checklists inside never reach the context. This hook detects competition work (Kaggle vocabulary
in the prompt, or a prompt typed inside a competition workspace), picks up to three relevant
skills by keyword and tells Claude to load them first. Unrelated prompts produce no output.
"""

from __future__ import annotations

import re
import sys

try:
    import _common as C
except Exception:  # pragma: no cover - fail open
    sys.exit(0)

GATE = re.compile(
    r"\bkaggle\b|\bleaderboard\b|\b(public|private)\s+(lb|score|leaderboard)\b|\blb\b|\boof\b|out[- ]of[- ]fold|"
    r"\bcompetition\b|\bshake[- ]?up\b|\bgrandmaster\b|\bsubmission\b",
    re.IGNORECASE,
)

ROUTES: list[tuple[str, str]] = [
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
    ("recsys-and-ranking", r"recommend|\branking\b|candidate generation|co-?visitation|\bsession\b"),
    ("hyperparameter-tuning", r"optuna|hyper-?parameter|(?<!fine-)(?<!fine )\btuning\b|search space"),
    ("deep-learning-training", r"learning rate|scheduler|mixed precision|\bamp\b|\bema\b|loss (curve|spike)|"
                               r"nan loss|batch size|gradient"),
    ("kaggle-gpu", r"\bgpu (quota|hours?|budget|time)|\bquota\b|\bp100\b|\bt4s?\b|accelerator|"
                   r"\b(train|run)\w*\b.{0,40}\bon kaggle\b|kaggle(?:'s)? (gpus?|notebooks?|kernels?)\b.{0,40}\btrain|"
                   r"remote(ly)? train|session (limit|timeout)"),
    ("competition-recon", r"public (notebooks?|kernels?)|top(-voted)? notebooks?|\bdiscussions?\b|write-?ups?|"
                          r"prior art|\brecon\b|reproduce (a|the|this) (notebook|kernel)"),
    ("competition-strategy", r"\bteam(ing|mates?| up)\b|\bmerg(e|er|ing)\b.{0,40}\bteams?\b|\bteams?\b.{0,40}\bmerg|"
                             r"which competitions?|grandmaster (title|tier|rank)|solo gold|\bgold medals?\b|"
                             r"join(ing|ed)? (late|the competition late)|several competitions"),
    ("kaggle-cli", r"kaggle (api|cli)|kaggle competitions|kaggle kernels|kaggle datasets|download (the )?data"),
]
COMPILED = [(name, re.compile(rx, re.IGNORECASE)) for name, rx in ROUTES]


def route(prompt: str, in_workspace: bool) -> list[str]:
    if not (GATE.search(prompt) or in_workspace):
        return []
    hits = []
    for name, rx in COMPILED:
        n = len(rx.findall(prompt))
        if n:
            hits.append((n, name))
    # keep ROUTES priority order among the top hits (debugging/validation before domain skills)
    order = {name: i for i, (name, _) in enumerate(ROUTES)}
    top = sorted(hits, key=lambda t: (-t[0], order[t[1]]))[:3]
    chosen = sorted((name for _, name in top), key=lambda n: order[n])
    return chosen or ["grandmaster-playbook"]


def main() -> None:
    payload = C.read_input()
    prompt = payload.get("prompt") or ""
    if not prompt.strip() or prompt.lstrip().startswith("/"):
        return  # slash commands already carry their own instructions
    skills = route(prompt, C.workspace(payload) is not None)
    if not skills:
        return
    names = ", ".join(f"kaggle-grandmaster:{s}" for s in skills)
    C.emit("UserPromptSubmit", additionalContext=(
        f"[kaggle-grandmaster] This is competition work. Before answering, load these skills with the Skill tool "
        f"and follow their checklists: {names}. Ground recommendations in them (validation first, leakage "
        f"mechanisms, honest OOF-based estimates), not just general ML advice."))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
