"""Files a competition repository never publishes: what its .gitignore names, and the check on a commit or a push.

A competition's repository goes public when the competition ends (/kg-showcase), and Kaggle's rules forbid
publishing the competition's data. Two layers keep the rest out of GitHub:

- **.gitignore** (``kgkit init``, ``kgkit repo-guard install``) names what is never code: the data, outputs and
  model weights, logs and scratch, credentials, tool caches. It names nothing that /kg-showcase removes at cleanup
  (copies of scripts such as ``train_v2.py``, probes outside ``work/``, scratch notebooks): those stay tracked until
  then, because an experiment's snapshot (``refs/kg/*``, :mod:`kgkit.snapshot`) leaves ignored files out and the
  journey links to the code that ran. Claude reads ignored files (Read, Glob and Bash do; only the Grep tool skips
  them), so ignoring never-code files hides nothing it needs.
- **The guard** checks what a ``git add``, ``git commit`` or ``git push`` would send, for what .gitignore cannot
  stop: ``git add -f``, a token pasted into code, a copy of a competition file under ``src/``. It runs in the
  PreToolUse hook for Claude's own git commands and as the repository's git hooks (pre-commit, pre-push) for the
  user's. Credentials and competition data are refused; other never-public files (submissions, predictions,
  weights, logs, kernel outputs) are asked about in Claude Code and warned about by the git hooks.

Standard library only (the hooks import it, and the git hooks run it); nothing here raises.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# (comment, [(pattern, a path the pattern ignores)]): the probe paths let `git check-ignore` tell whether a
# repository's own .gitignore already covers a pattern, whatever it is spelled like.
GITIGNORE = (
    ("# Competition data (copies and subsets too): Kaggle's rules forbid publishing it", [
        ("/data/", "data/kgkit-probe.csv"),
        ("*.zip", "kgkit-probe.zip"),
    ]),
    ("# Outputs: models, predictions, submissions", [
        ("/artifacts/", "artifacts/kgkit-probe.csv"),
        ("/subs/*.csv", "subs/kgkit-probe.csv"),
        ("*.npy", "src/kgkit-probe.npy"), ("*.npz", "src/kgkit-probe.npz"), ("*.pkl", "src/kgkit-probe.pkl"),
        ("*.joblib", "src/kgkit-probe.joblib"),
        ("*.parquet", "src/kgkit-probe.parquet"), ("*.feather", "src/kgkit-probe.feather"),
        ("*.pt", "src/kgkit-probe.pt"), ("*.pth", "src/kgkit-probe.pth"), ("*.ckpt", "src/kgkit-probe.ckpt"),
        ("*.safetensors", "src/kgkit-probe.safetensors"), ("*.onnx", "src/kgkit-probe.onnx"),
        ("*.h5", "src/kgkit-probe.h5"),
        ("wandb/", "src/wandb/x"), ("lightning_logs/", "src/lightning_logs/x"),
        ("catboost_info/", "src/catboost_info/x"),
        ("mlruns/", "src/mlruns/x"),
    ]),
    ("# Scratch: probes, logs, downloads", [
        ("/work/", "work/kgkit-probe.py"),
        ("*.log", "src/kgkit-probe.log"),
    ]),
    ("# Credentials", [
        ("kaggle.json", "src/kaggle.json"),
        ("access_token", "src/access_token"),
        (".env", "src/.env"), (".env.*", "src/.env.kgkit-probe"), ("!.env.example", None),
        ("*.pem", "src/kgkit-probe.pem"),
        ("/.claude/settings.local.json", ".claude/settings.local.json"),
    ]),
    ("# Caches", [
        ("__pycache__/", "src/__pycache__/a.pyc"),
        (".ipynb_checkpoints/", "src/.ipynb_checkpoints/a.ipynb"),
        (".pytest_cache/", ".pytest_cache/x"),
    ]),
)


def _git(root: Path, *args: str, data: str | None = None, timeout: float = 20) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=str(root), input=data,
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def _covered(root: Path, text: str) -> set[str]:
    """The patterns of :data:`GITIGNORE` that ``root`` already ignores: by git's own rules (any spelling, a global
    ignore file too) in a repository, else by the lines of ``text``."""
    lines = {ln.strip() for ln in text.splitlines()}
    bare = {x.strip("/") for x in lines}
    have = {p for _, group in GITIGNORE for p, _ in group if p in lines or p.strip("/") in bare}
    probes = {probe: p for _, group in GITIGNORE for p, probe in group if probe}
    r = _git(root, "check-ignore", "--no-index", "--stdin", data="\n".join(probes) + "\n")
    if r is not None and r.returncode in (0, 1):
        have |= {probes[x] for x in r.stdout.splitlines() if x in probes}
    return have


def ensure_gitignore(root: str | Path) -> list[str]:
    """Add to ``root/.gitignore`` each pattern of :data:`GITIGNORE` it does not cover yet, under its group's
    comment, keeping the file's own lines and line endings. Returns the patterns added. Never raises."""
    root = Path(root)
    gi = root / ".gitignore"
    try:
        raw = gi.read_bytes() if gi.is_file() else b""
        text = raw.decode("utf-8", "replace")
        have = _covered(root, text)
        lines = {ln.strip() for ln in text.splitlines()}
        blocks, added = [], []
        for comment, group in GITIGNORE:
            new = [p for p, probe in group if (p not in have if probe else p not in lines)]
            if not [p for p in new if not p.startswith("!")]:
                continue  # a negation only goes with a pattern it qualifies
            blocks.append([comment, *new])
            added += [p for p in new if not p.startswith("!")]
        if not blocks:
            return []
        nl = "\r\n" if b"\r\n" in raw else "\n"
        lead = nl if raw and not raw.endswith(b"\n") else ""
        if raw:
            lead += nl  # a blank line between the repository's own lines and kgkit's
        body = (nl + nl).join(nl.join(b) for b in blocks) + nl
        with open(gi, "a", encoding="utf-8", newline="") as f:
            f.write(lead + ("" if raw else f"# Written by kgkit init: never-public files{nl}{nl}") + body)
        return added
    except OSError:
        return []

