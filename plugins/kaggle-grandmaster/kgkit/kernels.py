"""Study and reproduce public notebooks — the fastest prior art in any competition.

    python -m kgkit kernels top -n 20                 # top-voted notebooks of this competition
    python -m kgkit kernels pull <owner>/<slug>       # pull + review + local script, under ref/<slug>/

``pull`` writes ``ref/<slug>/REVIEW.md``: the CV scheme the notebook uses, libraries / model
families, scores printed in its outputs, every input it reads and which of those are *not* the
competition data (external datasets, other notebooks' outputs, models — check they are public and
allowed, and that you can access them), and ``ref/<slug>/<slug>_local.py``: the code cells as a
script with Kaggle paths mapped to the local workspace (competition data -> ``data/``, other inputs
-> ``data/ext/<name>``, ``/kaggle/working`` -> ``artifacts/ref/<slug>``) and shell/magic lines
commented out. If it saves OOF predictions, bring them into the ledger on *our* folds with
``python -m kgkit ledger import`` and they become one more ensemble candidate.

The best public notebook is the baseline to beat, not the solution: re-validate its CV on the
frozen folds before believing its LB number.
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess
from pathlib import Path

CV_PATTERNS = {
    "StratifiedGroupKFold": r"StratifiedGroupKFold",
    "GroupKFold": r"(?<!Stratified)GroupKFold",
    "StratifiedKFold": r"StratifiedKFold",
    "MultilabelStratifiedKFold": r"MultilabelStratifiedKFold",
    "KFold": r"(?<![A-Za-z])KFold",
    "TimeSeriesSplit": r"TimeSeriesSplit",
    "train_test_split (single holdout)": r"train_test_split",
}
LIB_PATTERNS = {
    "LightGBM": r"\blightgbm\b|\blgb\.",
    "XGBoost": r"\bxgboost\b|\bxgb\.",
    "CatBoost": r"\bcatboost\b",
    "sklearn models": r"from sklearn\.(linear_model|svm|neighbors|ensemble)",
    "PyTorch": r"\bimport torch\b|\bfrom torch\b",
    "timm": r"\btimm\b",
    "transformers": r"\btransformers\b",
    "PEFT/LoRA": r"\bpeft\b|LoraConfig",
    "vLLM": r"\bvllm\b",
    "TensorFlow/Keras": r"\btensorflow\b|\bkeras\b",
    "TabPFN": r"\btabpfn\b",
    "AutoGluon": r"\bautogluon\b",
    "Optuna": r"\boptuna\b",
    "RAPIDS (cuDF/cuML)": r"\bcudf\b|\bcuml\b",
    "Polars": r"\bpolars\b",
}
SCORE_RX = re.compile(r"(?i)\b(cv|oof|valid(?:ation)?|val|lb|score|auc|rmse|logloss|f1|qwk|map@\d+)\b[^\n\d-]{0,25}"
                      r"(-?\d+\.\d{3,})")
INPUT_RX = re.compile(r"/kaggle/input/(?:competitions/|datasets/[^/'\"\s]+/)?([A-Za-z0-9._-]+)")


def _kaggle(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    exe = shutil.which("kaggle")
    if exe is None:
        raise SystemExit("kaggle CLI not found: pip install kaggle (and authenticate; see skill kaggle-cli)")
    return subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout)


def top(slug: str, n: int = 20, sort_by: str = "voteCount") -> list[dict]:
    r = _kaggle(["kernels", "list", "--competition", slug, "--sort-by", sort_by, "--page-size", str(n), "--csv"])
    if r.returncode != 0:
        raise SystemExit(f"kaggle kernels list failed: {r.stderr.strip() or r.stdout.strip()}")
    return parse_list_csv(r.stdout)


def parse_list_csv(text: str) -> list[dict]:
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("Warning")]
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def code_cells(path: Path) -> list[str]:
    """Source of a notebook's code cells, or a script's text as one cell."""
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        return ["".join(c.get("source", "")) if isinstance(c.get("source"), list) else c.get("source", "")
                for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    return [path.read_text(encoding="utf-8", errors="replace")]


def output_text(path: Path) -> str:
    if path.suffix != ".ipynb":
        return ""
    nb = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for c in nb.get("cells", []):
        for o in c.get("outputs", []) or []:
            t = o.get("text") or (o.get("data") or {}).get("text/plain") or ""
            parts.append("".join(t) if isinstance(t, list) else str(t))
    return "\n".join(parts)


def review(code: str, outputs: str, meta: dict, competition: str | None) -> dict:
    cv = [name for name, rx in CV_PATTERNS.items() if re.search(rx, code)]
    libs = [name for name, rx in LIB_PATTERNS.items() if re.search(rx, code, re.IGNORECASE)]
    scores = [f"{m.group(1)} {m.group(2)}" for m in SCORE_RX.finditer(outputs)][:15]
    inputs = sorted(set(INPUT_RX.findall(code)))
    comp_names = {competition} if competition else set()
    comp_names |= {s.split("/")[-1] for s in meta.get("competition_sources", []) or []}
    external = {
        "datasets": list(meta.get("dataset_sources", []) or []),
        "kernels": list(meta.get("kernel_sources", []) or []),
        "models": list(meta.get("model_sources", []) or []),
    }
    other_inputs = [i for i in inputs if i not in comp_names]
    return {"cv": cv, "libs": libs, "scores": scores, "inputs": inputs, "external": external,
            "other_inputs": other_inputs, "seeds_fixed": bool(re.search(r"seed|random_state", code))}


def to_local_script(cells: list[str], competition: str | None, ref_slug: str) -> str:
    out = [f"# Local reproduction of a public notebook (generated by kgkit kernels pull).",
           "# Kaggle paths are mapped to this workspace; shell/magic lines are commented out.",
           "# Run from the project root. Re-validate its CV on data/folds.csv before trusting any number.", ""]
    for i, cell in enumerate(cells):
        out.append(f"# %% cell {i}")
        for line in cell.splitlines():
            s = line.lstrip()
            if s.startswith(("!", "%")):
                out.append("# " + line)
                continue
            if competition:
                line = re.sub(rf"/kaggle/input/(?:competitions/)?{re.escape(competition)}", "data", line)
            line = re.sub(r"/kaggle/input/(?:competitions/|datasets/[^/'\"\s]+/)?([A-Za-z0-9._-]+)", r"data/ext/\1", line)
            line = line.replace("/kaggle/working", f"artifacts/ref/{ref_slug}")
            out.append(line)
        out.append("")
    return "\n".join(out)


def format_review(ref: str, rev: dict) -> str:
    ext = rev["external"]
    lines = [f"# Review: {ref}", "",
             f"- **CV scheme**: {', '.join(rev['cv']) or 'none found (single fit? LB-tuned? treat its scores with suspicion)'}",
             f"- **Libraries / model families**: {', '.join(rev['libs']) or '-'}",
             f"- **Scores printed in outputs**: {'; '.join(rev['scores']) or 'none found'}",
             f"- **Inputs read**: {', '.join(rev['inputs']) or '-'}",
             f"- **Non-competition inputs**: {', '.join(rev['other_inputs']) or 'none'}",
             f"- **Attached datasets**: {', '.join(ext['datasets']) or '-'}; **kernels**: "
             f"{', '.join(ext['kernels']) or '-'}; **models**: {', '.join(ext['models']) or '-'}",
             f"- **Seeds fixed**: {'yes' if rev['seeds_fixed'] else 'no'}", "",
             "## Checklist", "",
             "- [ ] External inputs are public, allowed by the rules, and accessible (private/deleted ones block reproduction)",
             "- [ ] Its CV scheme matches ours (grouping / time); if not, its CV number is not comparable",
             "- [ ] Re-run on data/folds.csv; log with `kgkit ledger import` (OOF) so it can join the blend",
             "- [ ] Note the ideas worth stealing in the backlog: `kgkit backlog add ... --source notebook`"]
    return "\n".join(lines) + "\n"


def pull(ref: str, dest_root: str | Path = "ref", competition: str | None = None, check_access: bool = False) -> dict:
    slug = ref.split("/")[-1]
    dest = Path(dest_root) / slug
    dest.mkdir(parents=True, exist_ok=True)
    r = _kaggle(["kernels", "pull", ref, "-p", str(dest), "-m"])
    if r.returncode != 0:
        raise SystemExit(f"kaggle kernels pull failed: {r.stderr.strip() or r.stdout.strip()}")
    return analyse_dir(dest, ref, competition, check_access)


def analyse_dir(dest: Path, ref: str, competition: str | None = None, check_access: bool = False) -> dict:
    slug = ref.split("/")[-1]
    meta_p = dest / "kernel-metadata.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
    code_files = [p for p in dest.iterdir() if p.suffix in (".ipynb", ".py", ".r", ".R")
                  and not p.name.endswith("_local.py")]
    if not code_files:
        raise SystemExit(f"no notebook or script found in {dest}")
    src = code_files[0]
    cells = code_cells(src)
    rev = review("\n".join(cells), output_text(src), meta, competition)
    if check_access:
        rev["inaccessible"] = [d for d in rev["external"]["datasets"]
                               if _kaggle(["datasets", "files", d], timeout=60).returncode != 0]
    text = format_review(ref, rev)
    if rev.get("inaccessible"):
        text += "\n**Inaccessible datasets** (private or deleted): " + ", ".join(rev["inaccessible"]) + "\n"
    (dest / "REVIEW.md").write_text(text, encoding="utf-8")
    if src.suffix in (".ipynb", ".py"):
        (dest / f"{slug}_local.py").write_text(to_local_script(cells, competition, slug), encoding="utf-8")
    rev["dir"] = str(dest)
    return rev
