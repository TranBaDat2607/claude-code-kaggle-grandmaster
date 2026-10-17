"""Download a Kaggle dataset, falling back to one request per file.

``kaggle datasets download <owner>/<slug>`` returns 404 for some public datasets (seen on OOF libraries
with many versions or recent updates: ``404 Client Error: Not Found for url: .../DownloadDataset``)
while ``kaggle datasets files`` lists them and ``-f <file>`` downloads every file. :func:`pull` tries the
whole-dataset archive first and, when that fails, lists the files (all pages) and downloads them one by
one into the same layout, unzipping the per-file archives the CLI makes of large files::

    python -m kgkit datasets pull dariushafshar/s6e10-golem-oof-library data/ext/golem
    python -m kgkit datasets pull <owner>/<slug> data/ext/x --files "*.npy" --per-file
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
import zipfile
from pathlib import Path

from .gpu import run_kaggle

FAILED_RE = re.compile(r"\b[45]\d\d (?:Client|Server) Error\b|^\s*(?:Traceback|\w*Error:)", re.IGNORECASE | re.MULTILINE)


def _error_line(text: str) -> str:
    """The line that says what went wrong (the CLI prints the dataset URL and licence around it)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return next((l for l in lines if FAILED_RE.search(l)), lines[0] if lines else "")


def list_files(ds: str, page_size: int = 200, max_pages: int = 100) -> list[dict]:
    """Every file of a dataset ({name, size, creationDate}), following the CLI's page tokens."""
    out, token = [], None
    for _ in range(max_pages):
        args = ["datasets", "files", ds, "--format", "json", "--page-size", str(page_size)]
        if token:
            args += ["--page-token", token]
        r = run_kaggle(args, timeout=300)
        text = r.stdout or ""
        if r.returncode != 0 or "[" not in text:
            raise SystemExit(f"kaggle datasets files {ds} failed: {(r.stderr or text).strip()[-500:]}")
        try:
            rows = json.loads(text[text.find("["):])
        except ValueError as e:
            raise SystemExit(f"kaggle datasets files {ds}: unreadable output ({e})")
        out += [x for x in rows if isinstance(x, dict) and x.get("name")]
        m = re.search(r"Next Page Token = (\S+)", text[:text.find("[")])
        token = m.group(1) if m else None
        if not token:
            break
    return out


def _unzip_here(z: Path, dest: Path) -> bool:
    if not zipfile.is_zipfile(z):
        return False
    with zipfile.ZipFile(z) as zf:
        root = dest.resolve()
        for name in zf.namelist():  # no member may land outside dest
            if not (dest / name).resolve().is_relative_to(root):
                raise SystemExit(f"{z}: archive member {name!r} points outside {dest}")
        zf.extractall(dest)
    z.unlink()
    return True


def pull(ds: str, dest: str | Path, files: str | None = None, per_file: bool = False, force: bool = False,
         unzip: bool = True) -> dict:
    """Download ``ds`` into ``dest``. ``files`` (a glob on the dataset path) or ``per_file`` skip the
    whole-dataset archive."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    whole_error = None
    if not files and not per_file:
        args = ["datasets", "download", ds, "-p", str(dest)] + (["--unzip"] if unzip else []) + (["--force"] if force else [])
        try:
            r = run_kaggle(args, timeout=4 * 3600)
            text = (r.stdout or "") + (r.stderr or "")
            if r.returncode == 0 and not FAILED_RE.search(text):
                return {"mode": "whole", "dest": str(dest), "files": None, "failed": []}
            whole_error = _error_line(text) or f"exit status {r.returncode}"
        except subprocess.TimeoutExpired:
            whole_error = "timed out"
        print(f"whole-dataset download failed ({whole_error}); downloading file by file")
    wanted = [f for f in list_files(ds) if not files or fnmatch.fnmatch(f["name"], files)]
    if not wanted:
        raise SystemExit(f"{ds}: no files" + (f" match {files!r}" if files else ""))
    done, failed = [], []
    for f in wanted:
        name = f["name"]
        sub = dest / Path(name).parent
        sub.mkdir(parents=True, exist_ok=True)
        target = sub / Path(name).name
        if target.is_file() and not force and (not f.get("size") or target.stat().st_size == f["size"]):
            done.append(name)
            continue
        args = ["datasets", "download", ds, "-f", name, "-p", str(sub), "-q"] + (["--force"] if force else [])
        try:
            r = run_kaggle(args, timeout=4 * 3600)
            text = (r.stdout or "") + (r.stderr or "")
            ok = r.returncode == 0 and not FAILED_RE.search(text)
        except subprocess.TimeoutExpired:
            ok, text = False, "timed out"
        z = sub / (Path(name).name + ".zip")  # the CLI zips large single files
        if ok and unzip and z.is_file() and not target.is_file():
            _unzip_here(z, sub)
        if ok and (target.is_file() or z.is_file()):
            done.append(name)
        else:
            failed.append(f"{name}: {_error_line(text) or 'not downloaded'}")
    return {"mode": "per-file", "dest": str(dest), "files": done, "failed": failed, "whole_error": whole_error}
