"""scripts/eval_table.py: the with/without table from `claude plugin eval` results."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_table as ET  # noqa: E402


def _run(path, cases, started, model="haiku", ablation="with-without", version="1.30.0"):
    path.write_text(json.dumps({
        "startedAt": started, "suite": {"ablation": ablation, "modelOverride": model, "judgeModel": "sonnet",
                                        "plugins": [{"name": "kaggle-grandmaster", "version": version}]},
        "cases": [{"name": n, "aggregates": {"score": w, "scoreWithout": wo, "delta": w - wo}}
                  for n, w, wo in cases]}))
    return path


def _evals(root):
    for name, tags in {"neg-a": "portable, capability, negative", "cv-b": "portable, regression, validation",
                       "ws-c": "shell, regression, workflow"}.items():
        (root / name).mkdir(parents=True)
        (root / name / "prompt.md").write_text(f"---\ndescription: x\ntags: [{tags}]\n---\n\nq\n")
    return root


def test_table_merges_runs_and_keeps_the_latest_case(tmp_path):
    evals = _evals(tmp_path / "evals")
    a = _run(tmp_path / "a.json", [("neg-a", 1.0, 1.0), ("cv-b", 0.5, 0.0)], "2026-11-01T00:00:00Z")
    b = _run(tmp_path / "b.json", [("cv-b", 1.0, 0.0), ("ws-c", 1.0, 0.5)], "2026-11-02T00:00:00Z")
    out = ET.table(ET.load([a, b]), evals)
    assert "### haiku (3 cases; plugin 1.30.0; judge sonnet; run 2026-11-01, 2026-11-02)" in out
    assert "| negative | 1 | 1.00 | 1.00 | +0.00 |" in out
    assert "| regression | 2 | 1.00 | 0.25 | +0.75 |" in out  # cv-b from the later run
    assert "| area: validation | 1 | 1.00 | 0.00 | +1.00 |" in out and "area: portable" not in out
    assert "| **all** | 3 | 1.00 | 0.50 | +0.50 |" in out
    assert out.index("| cv-b | regression |") < out.index("| ws-c |") < out.index("| neg-a | negative |")


def test_runs_without_a_baseline_are_refused(tmp_path):
    p = _run(tmp_path / "a.json", [("cv-b", 1.0, 0.0)], "2026-11-01T00:00:00Z", ablation="none")
    with pytest.raises(SystemExit, match="no baseline arm"):
        ET.load([p])


def test_cli_prints_the_table(tmp_path):
    p = _run(tmp_path / "a.json", [("cv-b", 1.0, 0.0)], "2026-11-01T00:00:00Z", model="sonnet")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_table.py"), str(p), "--evals",
                        str(_evals(tmp_path / "evals"))], capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONIOENCODING": "cp1252"})  # a Windows console's encoding
    assert r.returncode == 0 and "### sonnet (1 cases" in r.stdout and "| Δ |" in r.stdout, r.stderr
