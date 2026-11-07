"""Competition data in git: is any of it there, and can the world see the repository?

Kaggle's rules forbid giving Competition Data to anyone outside the competition, and a public repository gives it to
everyone. A ``.gitignore`` that names the downloaded files (``data/train.csv``) keeps those out, but not a file made
from them under another name: in UAT a subset of ``train.csv`` (the report text and expert labels of 58 studies) sat
in a public repository for two and a half months. So ``kgkit audit-repo`` compares contents, not names.

* The reference is the competition's files on this machine, among the files under ``data/`` and ``input/`` that git
  does not track (downloads are kept out of git): those at the top of the folder (or of ``<folder>/<slug>/``), those
  beside a ``train``/``test``/``sample_submission`` file, and those the download zip (``<folder>/<slug>.zip``) lists.
  The rest of the folder (public OOF libraries, notebook outputs, matrices made from the data) is not the
  competition's data and is only checked for exact copies: in UAT a ``data/`` of 39 GB took the audit 39+ minutes.
  ``--data`` names the reference instead. Indexing it gets at most half the time budget.
* Every blob reachable from a branch or tag is checked (history too: a file deleted in a later commit is still
  published), and so are the files ``git add -A`` would commit next. A file shares competition data when it is an
  exact copy of a reference file, holds rows of a reference table (cells compared as values, so a re-saved CSV still
  matches), holds passages of its text (45+ characters, compared without case, punctuation or line endings;
  boilerplate that recurs in many rows does not count), or has label columns whose values equal the competition's
  labels for the same IDs.
* A table keyed by the competition's IDs whose columns are not copies (predictions, labels made by a model, folds) is
  a judgment call and is listed as one; so is a table with the competition's ID and label column names when nothing
  on this machine can tell copies from your own values.
* Files under ``data/`` that git neither tracks nor ignores are listed: ``git add -A`` would commit them.

Whether a GitHub remote is public comes from ``gh repo view --json visibility`` (else GitHub's anonymous API: a
public repository answers, a private one is "not found"). The result is cached in ``.kaggle-gm/audit.json`` (paths and
counts, never contents) for the session brief (``brief_lines``).

This module must stay standard-library only: the SessionStart hook imports it.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path

from . import state as S

DATA_DIRS = ("data", "input")  # where a workspace keeps the competition's files (/kg-start downloads into data/)
CACHE = "audit.json"
TABLE_EXT = {".csv": ",", ".tsv": "\t"}
TEXT_EXT = {".txt", ".json", ".jsonl", ".ndjson"}
LINE_EXT = {".jsonl", ".ndjson"}  # one record per line: lines are compared as rows
NO_CONTENT_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".dcm", ".nii", ".gz", ".zip",
                  ".7z", ".tar", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx", ".pkl",
                  ".joblib", ".h5", ".hdf5", ".parquet", ".feather", ".pdf", ".mp3", ".wav", ".ogg", ".flac", ".mp4"}
COMPETITION_FILE = re.compile(r"^(?:train|test|sample_submission|solution|train_labels|test_labels)"
                              r"\.(?:csv|tsv|json|jsonl|parquet|feather|zip)$", re.IGNORECASE)
MIN_PASSAGE, MAX_PASSAGE, CHUNK = 45, 300, 150  # characters, after lower-casing and dropping punctuation
MIN_ROW_CHARS = 20
LEAK_AT = 3             # matched passages or rows that make a file a leak; fewer is a check
LABEL_AGREE = 0.995     # share of compared rows whose value equals the competition's label for the row's ID
MIN_LABEL_ROWS = 5
ID_OVERLAP = 0.8        # share of a column's values that are competition IDs
MIN_ID_ROWS = 5
CAPS = {"passages": 1_500_000, "rows": 2_000_000, "ids": 2_000_000, "pairs": 1_000_000}  # beyond: sampled by hash
MAX_REF_ROWS = 3_000_000
MAX_REF_BYTES = 2 * 2**30
MAX_SCAN_ROWS = 500_000
MAX_OTHER = 5_000       # files in the data folders that are neither tables nor text, checked for exact copies
WALK_DEPTH, BIG_DIR = 4, 2000
STALE_DAYS = 14
GITHUB_RE = re.compile(r"github\.com[:/]+([\w.-]+)/([\w.-]+?)(?:\.git)?/*$", re.IGNORECASE)
SPLIT_RE = re.compile(r"[\"\u3002\uff01\uff1f]|[.!?;](?=[\s\"']|$)|\n")
NONWORD_RE = re.compile(r"[\W_]+")
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")  # scripts written without spaces
NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")
INT_FLOAT_RE = re.compile(r"(-?\d+)\.0+$")
ESCAPES = (("\\r\\n", "\n"), ("\\n", "\n"), ("\\r", "\n"), ("\\t", " "), ('\\"', '"'))
GITHUB_DOCS = ("https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/"
               "removing-sensitive-data-from-a-repository")


# ---------------------------------------------------------------------------- normalising text and values
def passages(text: str) -> list[str]:
    """Pieces of prose to compare: sentences of MIN_PASSAGE..MAX_PASSAGE characters once lower-cased, with punctuation
    and line endings dropped (a copy checked out with CRLF matches); longer runs are cut into word-aligned pieces of
    about CHUNK characters. JSON and Python escapes are undone first (text in a notebook's outputs)."""
    for a, b in ESCAPES:
        if a in text:
            text = text.replace(a, b)
    out = []
    for frag in SPLIT_RE.split(text.replace("\r\n", "\n").replace("\r", "\n")):
        if len(frag) < MIN_PASSAGE or (len(frag) > 100 and " " not in frag and not CJK_RE.search(frag)):
            continue  # too short, or base64 / a hash / minified data rather than prose
        norm = NONWORD_RE.sub(" ", frag.lower()).strip()
        if len(norm) < MIN_PASSAGE:
            continue
        if len(norm) <= MAX_PASSAGE:
            if _prose(norm):
                out.append(norm)
            continue
        cur, n = [], 0
        for w in norm.split(" "):
            cur.append(w)
            n += len(w) + 1
            if n >= CHUNK:
                piece = " ".join(cur)
                if _prose(piece):
                    out.append(piece)
                cur, n = [], 0
        if n >= MIN_PASSAGE and _prose(" ".join(cur)):
            out.append(" ".join(cur))
    return out


def _prose(norm: str) -> bool:
    """Words, not an identifier: at least four tokens that are not just digits (a DICOM UID such as
    1.2.826.0.1.3680043... is all digits once its dots are gone; IDs are compared as IDs, not as text)."""
    return CJK_RE.search(norm) is not None or sum(not w.isdigit() for w in norm.split()) >= 4


def _cell(v: str) -> str:
    """A cell as a value: numbers in one spelling ("1", "1.0" and "1.00" agree), text with its whitespace collapsed
    (a quoted multi-line cell has CRLF inside in a Windows checkout and LF in git's copy)."""
    if "\n" in v or "\r" in v or "\t" in v or "  " in v:
        v = " ".join(v.split())
    v = v.strip()
    if v and NUM_RE.match(v):
        try:
            return repr(float(v))
        except (ValueError, OverflowError):
            pass
    return v


def _idnorm(v: str) -> str:
    """An ID as text; an integer ID written as a float by pandas ("17.0") is the integer."""
    v = v.strip()
    m = INT_FLOAT_RE.match(v)
    return m.group(1) if m else v


def _row_key(cells: list[str]) -> int | None:
    """A row's fingerprint, None for rows too plain to tell apart from a coincidence."""
    vals = [c for c in cells if c]
    if len(vals) < 3 or len(set(vals)) < 2 or sum(map(len, vals)) < MIN_ROW_CHARS:
        return None
    return hash("\x1f".join(cells))


def _is_sample(name: str) -> bool:
    low = Path(name).name.lower()
    return "sample" in low and "sub" in low


def _csv_limit() -> None:
    """Long text cells (reports, essays) exceed csv's default field limit; C long is 32-bit on Windows."""
    csv.field_size_limit(2**31 - 1)


def blob_id(path: Path, algo: str = "sha1") -> str:
    """The id git gives a file's contents, to find exact copies of any file type without reading git's objects."""
    h = hashlib.new(algo)
    h.update(b"blob %d\0" % path.stat().st_size)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------- the competition's files
class Reference:
    """Fingerprints of the competition's files, the things a copy shares with them: passages of text, rows, the IDs
    and the label value of each (label, ID). Python's ``hash`` is enough, everything is compared within one run. A
    store that outgrows ``CAPS`` keeps only the hashes divisible by a growing modulus, and lookups apply the same
    modulus, so a large copy is still found."""

    def __init__(self, id_names: set[str], targets: set[str]):
        self.id_names, self.targets = id_names, targets
        self.files: list[dict] = []
        self.passages: dict[int, int] = {}  # passage -> first reference row that has it (file << 40 | row)
        self.spread: Counter = Counter()    # passage -> reference rows that have it (boilerplate spreads)
        self.rows: dict[int, int] = {}      # row -> file
        self.ids: dict[int, int] = {}       # ID -> file
        self.pairs: dict[int, int] = {}     # (label, ID) -> value
        self.labels: list[str] = []         # label columns whose values are compared
        self.label_files: list[str] = []
        self.mod = dict.fromkeys(CAPS, 1)
        self.sizes: dict[int, list[int]] = {}
        self.blob_ids: dict[int, str] = {}
        self.dense_ids = False              # IDs 0..N-1: any column of small integers would "overlap" them
        self.n_rows = 0
        self.partial: list[str] = []
        self.deadline = float("inf")        # time.monotonic() at which reading reference files stops

    def late(self) -> bool:
        return time.monotonic() > self.deadline

    @property
    def tables(self) -> bool:
        return any(f["kind"] == "table" for f in self.files)

    def boiler(self) -> int:
        """A passage in more reference rows than this is a template sentence, not evidence."""
        return max(5, self.n_rows // 500)

    def _put(self, kind: str, store: dict, key: int, value: int) -> None:
        if key % self.mod[kind] or key in store:
            return
        store[key] = value
        if len(store) > CAPS[kind]:
            self._shrink(kind, store)

    def _shrink(self, kind: str, store: dict) -> None:
        m = self.mod[kind] = self.mod[kind] * 2
        for k in [k for k in store if k % m]:
            del store[k]
            if kind == "passages":
                self.spread.pop(k, None)

    def _add_passages(self, texts, packed: int) -> None:
        found = {hash(p) for t in texts for p in passages(t)}
        for h in found:
            if h % self.mod["passages"]:
                continue
            self.spread[h] += 1
            if h not in self.passages:
                self.passages[h] = packed
                if len(self.passages) > CAPS["passages"]:
                    self._shrink("passages", self.passages)

    def _file(self, path: Path, label: str, kind: str, role: str = "competition") -> int:
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        self.files.append({"path": label, "kind": kind, "rows": 0, "size": size, "abs": path, "role": role})
        self.sizes.setdefault(size, []).append(len(self.files) - 1)
        return len(self.files) - 1

    def add_table(self, path: Path, label: str, delim: str) -> None:
        fi = self._file(path, label, "table")
        info = self.files[fi]
        if info["size"] > MAX_REF_BYTES:
            self.partial.append(f"{label}: larger than {MAX_REF_BYTES // 2**30} GB, only exact copies are looked for")
            return
        sample = _is_sample(label)  # its "labels" are placeholders: IDs and column names only
        all_int, lo, hi, n_ids = True, None, None, 0
        try:
            with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
                reader = csv.reader(f, delimiter=delim)
                names = [h.strip() for h in next(reader, [])]
                id_i = next((i for i, n in enumerate(names) if n in self.id_names), None)
                lab = [] if sample else [i for i, n in enumerate(names) if n in self.targets and i != id_i]
                if lab:
                    self.label_files.append(label)
                    self.labels += [names[i] for i in lab if names[i] not in self.labels]
                for r, row in enumerate(reader):
                    if r >= MAX_REF_ROWS:
                        self.partial.append(f"{label}: only its first {MAX_REF_ROWS:,} rows were read")
                        break
                    if not r & 0xFFFF and r and self.late():
                        self.partial.append(f"{label}: only its first {r:,} rows were read (time budget)")
                        break
                    info["rows"] += 1
                    cells = [_cell(c) for c in row]
                    k = _row_key(cells)
                    if k is not None:
                        self._put("rows", self.rows, k, fi)
                    longs = [c for c in row if len(c) >= MIN_PASSAGE]
                    if longs:
                        self._add_passages(longs, fi << 40 | r)
                    if id_i is None or id_i >= len(row):
                        continue
                    idv = _idnorm(row[id_i])
                    if not idv:
                        continue
                    self._put("ids", self.ids, hash(idv), fi)
                    n_ids += 1
                    if all_int:
                        try:
                            x = int(idv)
                            lo, hi = (x, x) if lo is None else (min(lo, x), max(hi, x))
                        except ValueError:
                            all_int = False
                    for i in lab:
                        if i < len(cells) and cells[i]:
                            self._put("pairs", self.pairs, hash((names[i], idv)), hash(cells[i]))
        except (OSError, csv.Error) as e:
            self.partial.append(f"{label}: read stopped after {info['rows']:,} rows ({e})")
        self.n_rows += info["rows"]
        if all_int and lo is not None and n_ids >= 2 and hi - lo + 1 == n_ids:
            self.dense_ids = True

    def add_text(self, path: Path, label: str) -> None:
        fi = self._file(path, label, "text")
        info = self.files[fi]
        if info["size"] > MAX_REF_BYTES:
            self.partial.append(f"{label}: larger than {MAX_REF_BYTES // 2**30} GB, only exact copies are looked for")
            return
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError as e:
            self.partial.append(f"{label}: unreadable ({e})")
            return
        lines = Path(label).suffix.lower() in LINE_EXT
        for r, unit in enumerate(text.splitlines() if lines else re.split(r"\n\s*\n", text)):
            if not r & 0xFFF and r and self.late():
                self.partial.append(f"{label}: only its first {r:,} parts were read (time budget)")
                break
            info["rows"] += 1
            if lines and len(unit.strip()) >= MIN_ROW_CHARS:
                self._put("rows", self.rows, hash(("line", unit.strip())), fi)
            if len(unit) >= MIN_PASSAGE:
                self._add_passages([unit], fi << 40 | r)
        self.n_rows += info["rows"]

    def add_other(self, path: Path, label: str, role: str = "competition") -> None:
        self._file(path, label, "file", role)  # exact copies only (parquet, images, archives, derived files)

    # ------------------------------------------------------------------------ checks
    def copy_of(self, size: int, ident: str, algo: str) -> str | None:
        """The reference file whose contents are exactly a blob's, by git blob id (computed only for equal sizes)."""
        for fi in self.sizes.get(size, ()):
            if fi not in self.blob_ids:
                try:
                    self.blob_ids[fi] = blob_id(self.files[fi]["abs"], algo)
                except OSError:
                    self.blob_ids[fi] = ""
            if self.blob_ids[fi] == ident:
                return self.files[fi]["path"]
        return None

    def check(self, text: str, path: str) -> dict:
        """What a file's text shares with the reference (empty when nothing)."""
        ext = Path(path).suffix.lower()
        out: dict = {}
        prose = [text]
        if ext in TABLE_EXT:
            prose = self._check_table(text, TABLE_EXT[ext], out)
        elif ext in LINE_EXT:
            found: Counter = Counter()
            for line in text.splitlines():
                line = line.strip()
                k = hash(("line", line)) if len(line) >= MIN_ROW_CHARS else None
                if k is not None and not k % self.mod["rows"] and k in self.rows:
                    found[self.rows[k]] += 1
            if found:
                out["rows"], out["row_src"] = sum(found.values()), self.files[found.most_common(1)[0][0]]["path"]
        hits: dict[int, int] = {}
        m, boiler = self.mod["passages"], self.boiler()
        for t in prose:
            for p in passages(t):
                h = hash(p)
                if not h % m and h in self.passages and self.spread[h] <= boiler:
                    hits[h] = self.passages[h]
        if hits:
            src = Counter(v >> 40 for v in hits.values()).most_common(1)[0][0]
            out.update(passages=len(hits), passage_rows=len(set(hits.values())),
                       passage_src=self.files[src]["path"])
        return out

    def _check_table(self, text: str, delim: str, out: dict) -> list[str]:
        """Rows, ID and label checks of a table; returns its long cells, the prose to compare."""
        try:
            reader = csv.reader(io.StringIO(text, newline=""), delimiter=delim)
            names = [h.strip() for h in next(reader, [])]
            rows = []
            for row in reader:
                rows.append(row)
                if len(rows) >= MAX_SCAN_ROWS:
                    break
        except csv.Error:
            return [text]
        found: Counter = Counter()
        for row in rows:
            k = _row_key([_cell(c) for c in row])
            if k is not None and not k % self.mod["rows"] and k in self.rows:
                found[self.rows[k]] += 1
        if found:
            out["rows"], out["row_src"] = sum(found.values()), self.files[found.most_common(1)[0][0]]["path"]
        id_named = next((n for n in names if n in self.id_names), None)
        named_labels = [n for n in names if n in self.targets]
        if id_named and named_labels:
            out["named"] = [id_named, named_labels]
        idc = self._id_column(names, rows)
        if idc is not None:
            labels = self._labels(names, rows, idc)
            if labels or not self.dense_ids:  # with IDs 0..N-1, "keyed by them" says nothing (a run number is one)
                out["id_col"], out["id_rows"] = names[idc] or f"column {idc + 1}", len(rows)
            if labels:
                out["labels"] = labels
        return [c for row in rows for c in row if len(c) >= MIN_PASSAGE]

    def _id_column(self, names: list[str], rows: list[list[str]]) -> int | None:
        """The column holding competition IDs: one named like the competition's ID column, else (unless the IDs are
        0..N-1, which any integer column would match) the column whose values are most often competition IDs."""
        named = [i for i, n in enumerate(names) if n in self.id_names]
        cands = named or ([] if self.dense_ids else list(range(min(len(names), 50))))
        m, best = self.mod["ids"], None
        for i in cands:
            checked = hit = 0
            for row in rows[:5000 if named else 1000]:
                v = _idnorm(row[i]) if i < len(row) else ""
                h = hash(v) if v else 1
                if not v or h % m:
                    continue
                checked += 1
                hit += h in self.ids
            share = hit / checked if checked else 0
            if checked >= MIN_ID_ROWS and share >= ID_OVERLAP and (best is None or share > best[1]):
                best = (i, share)
        return best[0] if best else None

    def _labels(self, names: list[str], rows: list[list[str]], idc: int) -> list[str]:
        """Columns whose values equal the competition's label for each row's ID (a copy, not a prediction: a model
        is not right on 99.5% of rows; a constant column does not count)."""
        if not self.pairs:
            return []
        rows = rows[:200_000]
        ids = [_idnorm(row[idc]) if idc < len(row) else "" for row in rows]
        m = self.mod["pairs"]
        known: dict[str, list[tuple[int, int]]] = {}
        for t in self.labels:
            lst = []
            for r, v in enumerate(ids):
                if v:
                    k = hash((t, v))
                    if not k % m and k in self.pairs:
                        lst.append((r, self.pairs[k]))
            if len(lst) >= MIN_LABEL_ROWS:
                known[t] = lst
        out = []
        for c in range(len(names)):
            if c == idc:
                continue
            vals = [_cell(row[c]) if c < len(row) else "" for row in rows]
            for lst in known.values():
                agree = sum(hash(vals[r]) == ref for r, ref in lst)
                if agree >= LABEL_AGREE * len(lst) and len({vals[r] for r, _ in lst}) >= 2:
                    out.append(names[c] or f"column {c + 1}")
                    break
        return out


# ---------------------------------------------------------------------------- git and GitHub
def _git(top, *args, data: bytes | None = None, timeout: float = 300) -> bytes | None:
    """stdout of a git command, None when git is missing or the command fails."""
    try:
        r = subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=str(top), input=data,
                           capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def _z(out: bytes | None) -> list[str]:
    return [p for p in (out or b"").decode("utf-8", "replace").split("\0") if p]


def toplevel(start) -> Path | None:
    out = _git(start, "rev-parse", "--show-toplevel", timeout=30)
    return Path(out.decode("utf-8", "replace").strip()) if out else None


def github_repo(url: str) -> str | None:
    """``owner/repo`` of a GitHub remote URL (https, ssh or scp form), None for any other host."""
    m = GITHUB_RE.search((url or "").strip())
    return f"{m.group(1)}/{m.group(2)}" if m else None


def safe_url(url: str) -> str:
    """A remote URL without the user and token some https remotes carry."""
    return re.sub(r"(?<=//)[^/@\s]+@", "", url or "")


def remotes(top) -> list[dict]:
    out, seen = [], set()
    text = (_git(top, "config", "--get-regexp", r"^remote\..*\.url$", timeout=30) or b"").decode("utf-8", "replace")
    for line in text.splitlines():
        key, _, url = line.partition(" ")
        name = key[len("remote."):-len(".url")]
        if (name, url) not in seen:
            seen.add((name, url))
            out.append({"name": name, "url": safe_url(url), "repo": github_repo(url)})
    return out


def visibility(repo: str, timeout: float = 20) -> dict:
    """``{"visibility": PUBLIC | PRIVATE | INTERNAL | NOT_PUBLIC | UNKNOWN, "forks", "via", "error"}`` of a GitHub
    repository: ``gh repo view`` (the user's own gh login), else GitHub's anonymous API, where a repository that
    anyone can see answers and any other is "not found". Only the repository's name is sent. Never raises."""
    if os.environ.get("KGKIT_OFFLINE"):
        return {"visibility": "UNKNOWN", "forks": None, "via": "", "error": "KGKIT_OFFLINE is set"}
    error = None
    gh = shutil.which("gh")
    if gh:
        try:
            r = subprocess.run([gh, "repo", "view", repo, "--json", "visibility,forkCount"], capture_output=True,
                               encoding="utf-8", errors="replace", timeout=timeout,
                               env={**os.environ, "GH_PROMPT_DISABLED": "1", "GH_NO_UPDATE_NOTIFIER": "1"})
            if r.returncode == 0:
                d = json.loads(r.stdout)
                return {"visibility": str(d.get("visibility") or "UNKNOWN").upper(), "forks": d.get("forkCount"),
                        "via": "gh", "error": None}
            error = "gh: " + " ".join((r.stderr or r.stdout).strip().splitlines()[-1:])[:200]
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            error = f"gh: {e}"[:200]
    import urllib.error
    import urllib.request

    req = urllib.request.Request(f"https://api.github.com/repos/{repo}",
                                 headers={"Accept": "application/vnd.github+json", "User-Agent": "kgkit-audit"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            d = json.loads(resp.read().decode("utf-8", "replace"))
        vis = "PRIVATE" if d.get("private") else str(d.get("visibility") or "public").upper()
        return {"visibility": vis, "forks": d.get("forks_count"), "via": "api.github.com", "error": None}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"visibility": "NOT_PUBLIC", "forks": None, "via": "api.github.com", "error": None}
        error = f"{error + '; ' if error else ''}GitHub API: HTTP {e.code}"
    except (OSError, ValueError) as e:
        error = f"{error + '; ' if error else ''}GitHub API: {e}"[:300]
    return {"visibility": "UNKNOWN", "forks": None, "via": "", "error": error}


def _history(top) -> list[tuple[str, int, str]]:
    """(blob id, size, path) of every blob reachable from a branch, tag or remote-tracking branch (not stashes):
    what a push publishes, deleted files included."""
    objs = _git(top, "rev-list", "--objects", "--exclude=refs/stash", "--all")
    if not objs:
        return []
    out = _git(top, "cat-file", "--batch-check=%(objecttype) %(objectname) %(objectsize) %(rest)", data=objs)
    blobs = []
    for line in (out or b"").decode("utf-8", "replace").splitlines():
        parts = line.split(" ", 3)
        if len(parts) >= 3 and parts[0] == "blob":
            blobs.append((parts[1], int(parts[2]), parts[3] if len(parts) > 3 else ""))
    return blobs


def _cat(top, ids: list[str]):
    """(blob id, contents) for each id, through one ``git cat-file --batch`` process (one process per blob is slow,
    on Windows most of all). Stops the process when the caller stops early."""
    import threading

    p = subprocess.Popen(["git", "cat-file", "--batch"], cwd=str(top), stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def feed():
        try:
            for i in ids:
                p.stdin.write(i.encode() + b"\n")
            p.stdin.close()
        except OSError:
            pass

    threading.Thread(target=feed, daemon=True).start()
    try:
        for _ in ids:
            head = p.stdout.readline().split()
            if not head:
                return
            if len(head) < 3:  # "<id> missing"
                continue
            data = p.stdout.read(int(head[2]))
            p.stdout.read(1)
            yield head[0].decode(), data
    finally:
        if p.poll() is None:
            p.kill()
        p.wait()


def _ignored(base: Path, d: str) -> bool:
    """Whether git ignores new files in folder ``d`` of ``base`` (from .gitignore when ``base`` is no repository)."""
    try:
        # no extension: a `*.csv` rule ignores the tables in the folder, not the folder (parquet, json, images)
        r = subprocess.run(["git", "check-ignore", "-q", "--no-index", f"{d}/kgkit-probe"], cwd=str(base),
                           capture_output=True, timeout=30)
        if r.returncode in (0, 1):
            return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        lines = (base / ".gitignore").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    pats = {f"{lead}{d}{tail}" for lead in ("", "/") for tail in ("", "/", "/*", "/**")}
    return any(line.strip() in pats for line in lines)


def ensure_ignored(base, dirs) -> list[str]:
    """Add each folder of ``dirs`` (relative to ``base``) that git does not ignore to ``base/.gitignore`` and return
    the patterns added. Files under them that git already tracks stay tracked (``git rm --cached`` untracks one)."""
    base = Path(base)
    added = [f"/{d.strip('/')}/" for d in dirs if not _ignored(base, d.strip("/"))]
    if not added:
        return []
    gi = base / ".gitignore"
    raw = gi.read_bytes() if gi.is_file() else b""
    nl = "\r\n" if b"\r\n" in raw else "\n"
    lead = nl if raw and not raw.endswith(b"\n") else ""
    with open(gi, "a", encoding="utf-8", newline="") as f:
        f.write(lead + nl.join(["# Competition data: Kaggle's rules forbid publishing it, copies and subsets included",
                                *added]) + nl)
    return added


# ---------------------------------------------------------------------------- the audit
def _walk(base: Path, notes: list[str]) -> list[Path]:
    """Files under ``base``, WALK_DEPTH folders deep; a folder of more than BIG_DIR entries (images, series) is not
    descended into and only its tables and text files count."""
    files, stack = [], [(base, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        big = len(entries) > BIG_DIR
        if big:
            notes.append(f"{d.as_posix()}: {len(entries):,} entries, not searched below (only its tables were read)")
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    if not big and depth < WALK_DEPTH and not e.name.startswith("."):
                        stack.append((Path(e.path), depth + 1))
                elif e.is_file(follow_symlinks=False):
                    ext = Path(e.name).suffix.lower()
                    if not big or ext in TABLE_EXT or ext in TEXT_EXT:
                        files.append(Path(e.path))
            except OSError:
                continue
    return files


def _rel(p: Path, top: Path) -> str:
    try:
        return p.relative_to(top).as_posix()  # no resolve() per file: a syscall each, slow on Windows
    except ValueError:
        pass
    try:
        return p.resolve().relative_to(top.resolve()).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def _names(st, heads: dict[str, list[str]]) -> tuple[set[str], set[str]]:
    """The competition's ID and label column names: competition.json (id_col, target), the sample submission's
    header (its ID, then its labels), else the training table's columns that the test table lacks."""
    ids = {st.id_col.strip()} if st is not None and st.id_col.strip() else set()
    targets = {t.strip() for t in (st.target if st is not None else "").split(",") if t.strip()}
    for path, head in heads.items():
        if _is_sample(path) and head:
            ids.add(head[0])
            targets |= set(head[1:])
    if not targets:
        train = [h for p, h in heads.items() if "train" in Path(p).name.lower()]
        test = [h for p, h in heads.items() if "test" in Path(p).name.lower()]
        if train and test:
            targets = set(train[0]) - set().union(*map(set, test))
    if not ids:
        ids = {n for h in heads.values() for n in h if n.lower() == "id"}
    return ids, targets - ids


def _header(path: Path, delim: str) -> list[str]:
    try:
        with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
            return [h.strip() for h in next(csv.reader(f, delimiter=delim), [])]
    except (OSError, csv.Error):
        return []


def _zip_listing(base: Path, slug: str) -> set[str]:
    """Paths (relative to ``base``) the competition's download zip lists, when it is still in a data folder: only
    its directory is read, not its contents."""
    out: set[str] = set()
    for d in DATA_DIRS if slug else ():
        z = base / d / f"{slug}.zip"
        try:
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    n = n.replace("\\", "/").strip("/")
                    if n:
                        out |= {f"{d}/{n}", f"{d}/{slug}/{n}"}  # extracted into the folder or into <slug>/
        except (OSError, zipfile.BadZipFile, ValueError):
            continue
    return out


def _competition_files(base: Path, files: list[Path], slug: str) -> set[Path]:
    """Which of the data folders' ``files`` are the competition's own: at the top of a data folder or of its
    ``<slug>/`` subfolder, in a folder that also holds a train/test/sample_submission file, or listed by the
    download zip. The others (``data/ext/``, ``data/oof/``, notebook outputs) were made or downloaded later."""
    listed = _zip_listing(base, slug)
    rel = {}
    for f in files:
        try:
            rel[f] = f.relative_to(base).as_posix()
        except ValueError:
            rel[f] = ""
    comp_dirs = {f.parent for f in files if COMPETITION_FILE.match(f.name)}
    out = set()
    for f, r in rel.items():
        parts = r.split("/")
        if (len(parts) == 2 or (len(parts) == 3 and slug and parts[1] == slug) or f.parent in comp_dirs
                or r in listed):
            out.add(f)
    return out


def _size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


def _sizes(paths) -> int:
    total = 0
    for p in paths:
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def reference(top: Path, base: Path, st, tracked: set[str], data=(), *, deadline: float = float("inf"),
              log=None) -> tuple[Reference, list[str]]:
    """The competition's files on this machine, fingerprinted. ``data`` (files or folders, used as given even when
    tracked) names them; without it they are the untracked files under ``base``'s data folders that
    ``_competition_files`` picks, and the folders' other files are only checked for exact copies. Reading stops
    at ``deadline`` (``time.monotonic()``); what was not read is said in the notes. ``log`` gets progress lines."""
    notes: list[str] = []
    found: dict[str, Path] = {}
    derived: dict[str, Path] = {}
    if data:
        for spec in data:
            p = Path(spec).expanduser()
            if p.is_dir():
                for f in _walk(p, notes):
                    found.setdefault(_rel(f, top), f)
            elif p.is_file():
                found.setdefault(_rel(p, top), p)
            else:
                notes.append(f"--data {spec}: no such file or folder")
    else:
        walked = []
        for d in DATA_DIRS:
            if (base / d).is_dir():
                walked += [f for f in _walk(base / d, notes) if _rel(f, top) not in tracked]
        own = _competition_files(base, walked, st.slug if st is not None else "")
        for f in walked:
            (found if f in own else derived).setdefault(_rel(f, top), f)
    readable = {rel: p for rel, p in found.items() if p.suffix.lower() in TABLE_EXT or p.suffix.lower() in TEXT_EXT}
    others = [(rel, p, "competition") for rel, p in found.items() if rel not in readable]
    others += [(rel, p, "other") for rel, p in derived.items()]
    if derived:
        notes.append(f"{len(derived):,} other file(s) in the data folders ({_size(_sizes(derived.values()))}, e.g. "
                     f"{_few(sorted({str(Path(r).parent.as_posix()) for r in derived}), 3)}) are not the competition's "
                     "files (not at the top of the folder, not beside train/test files, not in the download zip): "
                     "only checked for exact copies. `--data <files>` names the competition's files instead")
    if log:
        log(f"audit-repo: indexing {len(readable):,} competition file(s) ({_size(_sizes(readable.values()))})"
            + (f"; {len(derived):,} other file(s) in the data folders only checked for exact copies" if derived else "")
            + ("" if data else "; `--data <files>` names the competition's files instead"))
    heads = {rel: _header(p, TABLE_EXT[p.suffix.lower()]) for rel, p in readable.items()
             if p.suffix.lower() in TABLE_EXT}
    ref = Reference(*_names(st, heads))
    ref.deadline = deadline

    def order(item):  # who owns a passage both have: the competition's own tables before files made from them
        rel, p = item
        name = Path(rel).name.lower()
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        return _is_sample(name), not COMPETITION_FILE.match(name), -size

    queue = sorted(readable.items(), key=order)
    for i, (rel, p) in enumerate(queue):
        if ref.late():  # the rest is still looked for as exact copies
            notes.append(f"time budget: {len(queue) - i:,} of {len(queue):,} competition file(s) were only checked "
                         f"for exact copies, e.g. {_few([r for r, _ in queue[i:]], 3)} (`--max-seconds` for more, or "
                         "`--data <files>` for fewer)")
            others = [(r, q, "competition") for r, q in queue[i:]] + others
            break
        if log and _sizes([p]) > 100 * 2**20:
            log(f"audit-repo: reading {rel} ({_size(_sizes([p]))})")
        if p.suffix.lower() in TABLE_EXT:
            ref.add_table(p, rel, TABLE_EXT[p.suffix.lower()])
        else:
            ref.add_text(p, rel)
    for rel, p, role in others[:MAX_OTHER]:  # images, archives, parquet, derived files: exact copies only
        ref.add_other(p, rel, role)
    if len(others) > MAX_OTHER:
        notes.append(f"{len(others) - MAX_OTHER:,} more files in the data folders (images, archives) were not checked "
                     "for exact copies")
    return ref, notes


def _priority(path: str) -> int:
    ext = Path(path).suffix.lower()
    return 0 if ext in TABLE_EXT or ext in TEXT_EXT else 1 if ext in (".ipynb", ".md", ".py", ".html", ".xml") else 2


CONTENT = ("exact", "rows", "passages", "labels", "id_col")  # evidence from contents, not from names


def _merge(agg: dict, path: str, where: str, ident: str, found: dict) -> None:
    """Fold one version's findings into its path's: the largest counts with their sources, every label column, and
    which versions had evidence in their contents (``hits``)."""
    f = agg.setdefault(path, {"path": path, "where": set(), "versions": set(), "hits": set()})
    f["where"].add(where)
    f["versions"].add(ident)
    if any(found.get(k) for k in CONTENT):
        f["hits"].add(ident)
    if found.get("passages", 0) > f.get("passages", 0):
        f.update({k: found[k] for k in ("passages", "passage_rows", "passage_src")})
    if found.get("rows", 0) > f.get("rows", 0):
        f.update(rows=found["rows"], row_src=found["row_src"])
    if found.get("labels"):
        f["labels"] = list(dict.fromkeys(f.get("labels", []) + found["labels"]))
    if found.get("id_col") and found.get("id_rows", 0) >= f.get("id_rows", 0):
        f.update(id_col=found["id_col"], id_rows=found["id_rows"])
    for k in ("exact", "named", "competition_name", "in_data"):
        if found.get(k) and not f.get(k):
            f[k] = found[k]


def _verdict(f: dict, tables: bool) -> tuple[str, list[str]]:
    """``leak`` (competition data in the file) or ``check`` (a judgment call), with the reasons."""
    why, leak = [], False
    if f.get("exact"):
        why.append(f"an exact copy of {f['exact']}")
        leak = True
    if f.get("rows"):
        why.append(f"{f['rows']:,} row(s) of {f['row_src']}")
        leak |= f["rows"] >= LEAK_AT
    if f.get("passages"):
        why.append(f"{f['passages']:,} passage(s) of competition text, from {f['passage_rows']:,} row(s) of "
                   f"{f['passage_src']}")
        leak |= f["passages"] >= LEAK_AT
    if f.get("labels"):
        labels = f["labels"]
        why.append(f"label column(s) equal to the competition's for the same IDs ({f['id_col']}, {f['id_rows']:,} "
                   f"rows): {', '.join(labels[:6])}" + (f" and {len(labels) - 6} more" if len(labels) > 6 else ""))
        leak = True
    if leak:
        return "leak", why
    if f.get("id_col"):
        why.append(f"keyed by the competition's IDs ({f['id_col']}, {f['id_rows']:,} rows) with columns that are not "
                   "copies of its labels: predictions or labels made from the data are a judgment call")
    elif f.get("named"):
        idn, labs = f["named"]
        why.append(f"columns named like the competition's ID ({idn}) and labels ("
                   + ", ".join(labs[:4]) + (f" and {len(labs) - 4} more" if len(labs) > 4 else "") + "): "
                   + ("its IDs are not in the files compared" if tables else
                      "no competition table on this machine to compare its values with"))
    if f.get("competition_name") and not f.get("exact") and (
            not tables or Path(f["path"]).suffix.lower() in NO_CONTENT_EXT):
        why.append("named like a competition file; contents not compared")
    if f.get("in_data") and not tables and not why:
        why.append("tracked in a data folder, where the competition's files go")
    return "check", why


def run(start=".", data=(), *, max_seconds: float = 300, max_blob_mb: float = 50, online: bool = True,
        fix_gitignore: bool = False, save: bool = True, log=None) -> dict:
    """Audit the git repository around ``start``; see the module docstring. Returns the result (``report`` prints it)
    and caches its summary in ``.kaggle-gm/audit.json`` when ``start`` is in a workspace. ``max_seconds`` bounds the
    whole audit, of which reading the competition's files gets at most half; ``log`` gets progress lines."""
    t0 = time.monotonic()
    _csv_limit()
    start = Path(start).resolve()
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    res: dict = {"at": now, "top": None, "slug": None, "error": None, "remotes": [], "public": None, "head": "",
                 "references": [], "notes": [], "scanned": {}, "findings": [], "leaks": [], "checks": [],
                 "unignored": [], "unignored_n": 0, "gitignore_added": [], "partial": None, "tables": False}
    top = toplevel(start)
    if top is None:
        res["error"] = "not a git repository, so nothing is in git yet (`git init` makes one)"
        return res
    res["top"] = top.as_posix()
    ws = S.find_root(start)
    base = ws if ws is not None and (ws == top or top in ws.parents) else top
    st = S.load(ws) if ws is not None else None
    res["slug"] = st.slug if st is not None else None
    res["head"] = (_git(top, "rev-parse", "HEAD", timeout=30) or b"").decode().strip()

    old = (load_cache(ws) or {}) if ws is not None else {}
    known = {r.get("repo"): r for r in old.get("remotes") or [] if isinstance(r, dict)
             and r.get("visibility") not in (None, "UNKNOWN")}
    for rem in remotes(top):
        if rem["repo"]:
            rem.update(visibility(rem["repo"]) if online else
                       {"visibility": "UNKNOWN", "forks": None, "via": "", "error": "not asked: --offline"})
            prev = known.get(rem["repo"])
            if rem["visibility"] == "UNKNOWN" and prev:  # an offline run must not forget that the repo is public
                rem.update(visibility=prev["visibility"], forks=prev.get("forks"),
                           via=f"the audit of {str(old.get('at') or '')[:10]} (now: {rem.get('error') or 'unknown'})",
                           error=None)
        res["remotes"].append(rem)
    seen = [r.get("visibility") for r in res["remotes"]]
    res["public"] = (True if "PUBLIC" in seen else
                     False if seen and all(v in ("PRIVATE", "INTERNAL", "NOT_PUBLIC") for v in seen) else None)

    dirs = [d for d in DATA_DIRS if (base / d).is_dir()]
    if fix_gitignore:
        res["gitignore_added"] = ensure_ignored(base, dirs)
    tracked = set(_z(_git(top, "ls-files", "-z")))
    pending = list(dict.fromkeys(_z(_git(top, "ls-files", "-z", "--others", "--exclude-standard"))
                                 + _z(_git(top, "ls-files", "-z", "-m"))))
    ref, res["notes"] = reference(top, base, st, tracked, data, deadline=t0 + max_seconds / 2, log=log)
    res["notes"] += ref.partial
    res["tables"] = ref.tables
    res["references"] = [{k: f[k] for k in ("path", "kind", "rows", "size", "role")} for f in ref.files]
    algo = "sha256" if (_git(top, "rev-parse", "--show-object-format", timeout=30) or b"").strip() == b"sha256" else "sha1"
    prefixes = tuple(_rel(base / d, top) + "/" for d in dirs)
    max_blob = int(max_blob_mb * 2**20)

    def names(path: str) -> dict:
        out = {}
        if COMPETITION_FILE.match(Path(path).name):
            out["competition_name"] = True
        if path.startswith(prefixes):
            out["in_data"] = True
        return out

    agg: dict = {}
    head_tree = {}
    for entry in _z(_git(top, "ls-tree", "-r", "-z", "HEAD")):
        meta, _, path = entry.partition("\t")
        parts = meta.split()
        if len(parts) == 3 and parts[1] == "blob":
            head_tree[path] = parts[2]
    blobs = _history(top)
    n_path: Counter = Counter()
    queue, skipped = [], 0
    for ident, size, path in blobs:  # rev-list lists newer blobs first: each path's newest versions come first
        found = names(path)
        exact = ref.copy_of(size, ident, algo) if size in ref.sizes else None
        if exact:
            found["exact"] = exact
        if Path(path).suffix.lower() in NO_CONTENT_EXT or size > max_blob:
            skipped += not exact
            if found:
                _merge(agg, path, "history", ident, found)
            continue
        queue.append((_priority(path), n_path[path], size, ident, path, found))
        n_path[path] += 1
    queue.sort()
    by_id = {q[3]: q for q in queue}
    if log:
        log(f"audit-repo: reading {len(queue):,} blob(s) of git history "
            f"({max(0, max_seconds - (time.monotonic() - t0)):.0f} s left of --max-seconds {max_seconds:.0f})")
    done = 0
    blobs_read = _cat(top, [q[3] for q in queue])
    for ident, data_ in blobs_read:
        if time.monotonic() - t0 > max_seconds:
            res["partial"] = (f"stopped after {max_seconds:.0f} s: {done:,} of {len(queue):,} blobs read "
                              "(`--max-seconds` for more)")
            break
        done += 1
        _, _, _, _, path, found = by_id[ident]
        found = dict(found)
        if b"\0" not in data_[:8000]:
            found.update(ref.check(data_.decode("utf-8", "replace").lstrip("\ufeff"), path))
        else:
            skipped += 1
        if found:
            _merge(agg, path, "history", ident, found)
    blobs_read.close()

    # Files `git add -A` would commit. Untracked files in the data folders are listed, not read: they are the
    # competition's files (the reference) or made from them, and the fix is to ignore the folder.
    unignored, n_pending = [], 0
    for rel in pending:
        if rel.startswith(prefixes) and rel not in tracked:
            unignored.append(rel)
            continue
        p = top / rel
        if res["partial"] is None and time.monotonic() - t0 > max_seconds:
            res["partial"] = f"stopped after {max_seconds:.0f} s, before every uncommitted file was read"
        if res["partial"] is not None or not p.is_file():
            continue
        n_pending += 1
        found = names(rel)
        try:
            size = p.stat().st_size
            if size in ref.sizes:
                exact = ref.copy_of(size, blob_id(p, algo), algo)
                if exact:
                    found["exact"] = exact
            if p.suffix.lower() not in NO_CONTENT_EXT and size <= max_blob:
                raw = p.read_bytes()
                if b"\0" not in raw[:8000]:
                    found.update(ref.check(raw.decode("utf-8", "replace").lstrip("\ufeff"), rel))
        except OSError:
            continue
        if found:
            _merge(agg, rel, "pending", "worktree", found)
    res["unignored"], res["unignored_n"] = unignored[:50], len(unignored)

    for path, f in agg.items():
        severity, why = _verdict(f, ref.tables)
        if not why:
            continue
        where = sorted(f["where"])
        evidence = f["hits"] or f["versions"]  # the versions whose contents matched (all of them for a name)
        in_head = path in head_tree and head_tree[path] in evidence
        item = {"path": path, "severity": severity, "why": why, "where": where, "in_head": in_head,
                "in_tree": path in head_tree, "versions": len(evidence - {"worktree"})}
        if "history" in where:  # how long it has been in git (and, for a public repo, published)
            first = (_git(top, "log", "--all", "--reverse", "--format=%h %ad", "--date=short", "--", path,
                          timeout=60) or b"").decode("utf-8", "replace").splitlines()
            item["first"] = first[0] if first else ""
        res["findings"].append(item)
    res["findings"].sort(key=lambda x: (x["severity"] != "leak", x["where"] == ["history"] and not x["in_head"],
                                        x["path"]))
    res["leaks"] = [x["path"] for x in res["findings"] if x["severity"] == "leak"]
    res["checks"] = [x["path"] for x in res["findings"] if x["severity"] == "check"]
    refs = (_git(top, "for-each-ref", "--format=%(refname)", timeout=60) or b"").decode("utf-8", "replace")
    n_refs = sum(1 for r in refs.splitlines() if r and r != "refs/stash")
    res["scanned"] = {"blobs": len(blobs), "refs": n_refs, "pending": n_pending, "skipped": skipped}
    if save and ws is not None:
        save_cache(ws, res)
    return res


# ---------------------------------------------------------------------------- the report
def _vis(r: dict) -> str:
    v = r.get("visibility") or "UNKNOWN"
    word = {"PUBLIC": "PUBLIC: anyone can read it", "PRIVATE": "private", "INTERNAL": "internal (enterprise members)",
            "NOT_PUBLIC": "not public (GitHub does not show it to anonymous visitors)"}.get(v)
    if word is None:
        word = "of unknown visibility" + (f" ({r['error']})" if r.get("error") else "")
    forks = r.get("forks")
    via = r.get("via") or ""
    return (word + (f"; {forks} fork(s), each with its own copy of the history" if forks else "")
            + (f" (as of {via})" if via.startswith("the audit of") else ""))


def _where(x: dict) -> str:
    if x["where"] == ["pending"]:
        return "not committed yet: `git add -A` would commit it"
    if x["in_head"]:
        return ("in the current tree (HEAD)" + (f", in git since {x['first']}" if x.get("first") else "")
                + ("; also uncommitted changes" if "pending" in x["where"] else ""))
    s = "in history" + (f" since {x['first']}" if x.get("first") else "")
    s += "; a later version is in HEAD" if x["in_tree"] else "; not in HEAD (deleted, or on another branch)"
    return s + (f"; {x['versions']} version(s)" if x["versions"] > 1 else "")


def _few(paths: list[str], n: int = 3) -> str:
    return ", ".join(paths[:n]) + (f" and {len(paths) - n} more" if len(paths) > n else "")


def steps(res: dict) -> list[str]:
    """What to do about the leaks: the user's decisions, printed as steps (nothing here rewrites history)."""
    leaks = [x for x in res["findings"] if x["severity"] == "leak"]
    committed = [x for x in leaks if "history" in x["where"]]
    head = [x["path"] for x in leaks if x["in_head"]]
    pending = [x["path"] for x in leaks if x["where"] == ["pending"]]
    public = res.get("public")
    out = []
    if committed and public:
        out.append("Make the repository private first (GitHub: Settings > General > Danger Zone): until then anyone "
                   "can clone or fork it.")
    if head:
        out.append(f"Take them out of the current tree and keep them out: `git rm --cached <path>` for {_few(head)}, "
                   "add each path to .gitignore, commit.")
    if committed:
        paths = " ".join(f"--path {x['path']}" for x in committed)
        out.append("Removing a file in a new commit leaves it in history. Rewrite history on a fresh clone with "
                   f"git-filter-repo (`pip install git-filter-repo`): `git filter-repo --invert-paths {paths}`, then "
                   "force-push every branch and tag. It changes every commit id: only when you are sure.")
    if committed and public is not False and any(r.get("repo") for r in res["remotes"]):
        out.append("GitHub keeps old commits reachable through pull-request refs (refs/pull/*) and in forks, which a "
                   f"force-push does not touch: ask GitHub Support to purge them ({GITHUB_DOCS}).")
    if committed and public:
        out.append("Kaggle's competition rules usually oblige you to notify Kaggle when you learn that Competition "
                   "Data was exposed: check the rules (`python -m kgkit pages --page rules`).")
    if pending:
        out.append(f"Not committed yet: keep {_few(pending)} out of git with .gitignore.")
    if res["unignored"]:
        out.append("Ignore the data folders: `python -m kgkit audit-repo --fix-gitignore` adds them to .gitignore.")
    return out


def report(res: dict) -> list[str]:
    if res.get("error"):
        return [f"audit-repo: {res['error']}"]
    out = [f"Competition data in git: audit of {res['top']}" + (f" ({res['slug']})" if res.get("slug") else "")]
    if not res["remotes"]:
        out.append("  remote: none, nothing is pushed anywhere yet")
    for r in res["remotes"]:
        out.append(f"  remote {r['name']}: " + (f"github.com/{r['repo']} is {_vis(r)}" if r.get("repo") else
                                               f"{r['url']} (not GitHub: check who can read it yourself)"))
    refs = [x for x in res["references"] if x.get("role", "competition") == "competition"]
    if refs:
        shown = "; ".join(f"{x['path']}" + (f" ({x['rows']:,} rows)" if x["kind"] != "file" else "") for x in refs[:5])
        out.append(f"  compared with {len(refs)} competition file(s) on this machine that git does not track: {shown}"
                   + (f"; and {len(refs) - 5} more" if len(refs) > 5 else ""))
    if not res["tables"]:
        out.append("  no competition table on this machine to compare contents with (looked for untracked files in "
                   "data/ and input/), so only names and folders were checked: pass `--data <folder or files>`, e.g. "
                   "train.csv downloaded to a folder outside the repo")
    sc = res["scanned"]
    out.append(f"  scanned {sc.get('blobs', 0):,} blob(s) reachable from {sc.get('refs', 0)} branch(es)/tag(s), and "
               f"{sc.get('pending', 0)} file(s) `git add -A` would commit; {sc.get('skipped', 0)} binary or large "
               "blob(s) only checked for exact copies")
    for note in res["notes"] + ([res["partial"]] if res["partial"] else []):
        out.append(f"  note: {note}")
    if res["gitignore_added"]:
        out.append(f"  added to .gitignore: {', '.join(res['gitignore_added'])} (files git already tracks there stay "
                   "tracked: `git rm --cached <path>` untracks one)")
    for severity, title in (("leak", "LEAK: competition data in git"),
                            ("check", "CHECK: your call (keep it in git only if it is your own work)")):
        items = [x for x in res["findings"] if x["severity"] == severity]
        if items:
            out += ["", f"{title} ({len(items)}):"]
            for x in items:
                out.append(f"  {x['path']}  [{_where(x)}]")
                out += [f"    - {w}" for w in x["why"]]
    if res["unignored"]:
        n = res.get("unignored_n", len(res["unignored"]))
        shown = res["unignored"][:8]
        out += ["", f"NOT IGNORED: files in the data folders that `git add -A` would commit ({n:,}):",
                f"  {', '.join(shown)}" + (f" and {n - len(shown):,} more" if n > len(shown) else "")]
    todo = steps(res)
    if todo:
        out += ["", "What to do" + (" (the repository is PUBLIC)" if res.get("public") else "") + ":"]
        out += [f"  {i}. {s}" for i, s in enumerate(todo, 1)]
    if not res["findings"] and not res["unignored"]:
        out += ["", "No competition data found in git" + ("" if res["tables"] else " by name (contents not compared)")
                + "."]
    return out


# ---------------------------------------------------------------------------- cache and session brief
def cache_path(root) -> Path:
    return Path(root) / S.STATE_DIR / CACHE


def save_cache(root, res: dict) -> None:
    """The result's summary (paths and counts, never contents or remote URLs) for the session brief. Never raises."""
    summary = {"at": res["at"], "head": res["head"], "public": res["public"], "leaks": res["leaks"],
               "checks": res["checks"], "unignored": res["unignored"], "partial": bool(res["partial"]),
               "compared": res["tables"],
               "remotes": [{k: r.get(k) for k in ("name", "repo", "visibility", "forks")} for r in res["remotes"]]}
    try:
        p = cache_path(root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    except OSError:
        pass


def load_cache(root) -> dict | None:
    try:
        d = json.loads(cache_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _git_config_text(start: Path) -> str:
    """The repository's git config, read directly (no git process: the brief must be fast)."""
    for d in [start, *start.parents]:
        g = d / ".git"
        try:
            if g.is_dir():
                return (g / "config").read_text(encoding="utf-8", errors="replace")
            if g.is_file():  # a worktree or submodule: "gitdir: <path>", whose config may be in its common dir
                m = re.match(r"gitdir:\s*(.+)", g.read_text(encoding="utf-8", errors="replace").strip())
                if not m:
                    return ""
                gd = Path(m.group(1).strip())
                gd = gd if gd.is_absolute() else d / gd
                common = gd / "commondir"
                if common.is_file():
                    c = Path(common.read_text(encoding="utf-8").strip())
                    gd = c if c.is_absolute() else gd / c
                return (gd / "config").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
    return ""


def config_github_repos(start) -> list[str]:
    urls = re.findall(r"(?m)^\s*url\s*=\s*(\S+)", _git_config_text(Path(start).resolve()))
    return list(dict.fromkeys(r for r in map(github_repo, urls) if r))


def brief_lines(root) -> list[str]:
    """The session brief's lines about competition data in git, from the cached audit (no network, no scan): a
    warning while it found leaks, files in the data folders that `git add -A` would commit, a reminder when a public
    repository's audit is old, and a nudge when a repository with a GitHub remote was never audited. Never raises."""
    try:
        return _brief(Path(root))
    except Exception:
        return []


def _brief(root: Path) -> list[str]:
    c = load_cache(root)
    if c is None:
        repos = config_github_repos(root)
        if not repos:
            return []
        return [f"Competition data in git: not checked yet, and the repo has a GitHub remote ({_few(repos, 2)}). Run "
                "`python -m kgkit audit-repo`: it asks GitHub whether the repo is public and checks files and history "
                "for copies of the competition's data, which Kaggle's rules forbid publishing."]
    when = str(c.get("at") or "")[:10]
    vis = {True: "the repo is PUBLIC", False: "the repo is not public"}.get(c.get("public"), "visibility unknown")
    out = []
    leaks = [str(p) for p in c.get("leaks") or []]
    if leaks:
        out.append(f"WARNING: competition data in git ({vis}; audit of {when}): {_few(leaks)}. Kaggle's rules forbid "
                   "publishing it, and deleting a file in a new commit leaves it in history: `python -m kgkit "
                   "audit-repo` prints the steps. Never rewrite history or force-push without the user's go-ahead.")
    unignored = [str(p) for p in c.get("unignored") or []]
    if unignored:
        out.append(f"Data files that `git add -A` would commit (audit of {when}): {_few(unignored)}. Ignore the folder "
                   "first: `python -m kgkit audit-repo --fix-gitignore`.")
    if not out and c.get("public"):
        try:
            age = (dt.datetime.now(dt.timezone.utc).date() - dt.date.fromisoformat(when)).days
        except ValueError:
            age = STALE_DAYS + 1
        if age > STALE_DAYS:
            out.append(f"The repo is public and its last competition-data audit was {when} ({age} days ago): re-run "
                       "`python -m kgkit audit-repo` to check the commits since.")
    return out
