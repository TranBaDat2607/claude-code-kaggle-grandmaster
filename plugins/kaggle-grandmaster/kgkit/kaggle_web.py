"""Kaggle's internal web API (``www.kaggle.com/api/i/...``): what the public API does not serve.

- ``writeups``: the competition's solution write-ups, ranked by the leaderboard (the private one once it exists,
  else the public one), each with its forum topic; the Writeups tab, not a title search.
- ``topic``: one write-up's source markdown (``rawMarkdown``), not HTML.
- ``versions``: every version of a notebook with the public score of the submission it made, Kaggle's best score
  for the notebook (which tells the score direction), and its fork parent.
- ``source``: the source of one version (``kaggle kernels pull`` only gets the latest).
- ``view``: a notebook's fork parent, the inputs it mounts and the URL of its Docker image's package list.

These endpoints serve the website. They are not Kaggle's public API and can change without notice, so every
caller falls back to the public path when one fails (``discussions solutions`` to its title search, ``kernels
pull`` to the latest version) and says it did. The requests go out through the Kaggle SDK's own session, the one
the CLI builds with whichever credentials it finds (``KAGGLE_API_TOKEN`` file, ``~/.kaggle/access_token``,
``kaggle.json``), plus the XSRF cookie that one GET of www.kaggle.com seeds: kgkit never reads a token.

``run`` starts this module in a child process (``python -m kgkit.kaggle_web``, a JSON request on stdin, one JSON
line on stdout) under the Kaggle CLI's interpreter, with VERBOSE scrubbed from the environment and a hard timeout;
whatever the SDK prints goes to stderr. It must stay standard-library only at import time: that interpreter may have
nothing but the Kaggle CLI installed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BASE = "https://www.kaggle.com"
GET_COMPETITION = "competitions.CompetitionService/GetCompetition"
GET_LEADERBOARD = "competitions.LeaderboardService/GetLeaderboard"
GET_TOPIC = "discussions.DiscussionsService/GetForumTopicById"
VIEW_MODEL = "kernels.LegacyKernelsService/GetKernelViewModel"
LIST_VERSIONS = "kernels.KernelsService/ListKernelVersions"
SOURCE = "kernels.KernelsService/GetKernelSessionSource"
MAX_VERSIONS = 60  # versions scored by `versions` (one request each), newest first
GRACE = 20  # seconds the child gets beyond its budget (start, imports, authentication)
SCORE_RX = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
CODE_URL_RX = re.compile(r"(?:kaggle\.com)?/code/([^/?#\s]+)/([^/?#\s]+)")


class WebError(Exception):
    """A request the internal API refused or answered with something kgkit cannot read."""


def score_num(text) -> float | None:
    """The number in a formatted score ('0.91135', '1.2e-3'); None for '-', '' or None."""
    m = SCORE_RX.search(str(text or ""))
    return float(m.group(0)) if m else None


def code_ref(url: str) -> str | None:
    """``owner/slug`` of a notebook URL or path (``/code/<owner>/<slug>``); None when it is none."""
    m = CODE_URL_RX.search(str(url or ""))
    return f"{m.group(1)}/{m.group(2)}" if m else None


def competition_id(post, slug: str) -> tuple[int, bool | None]:
    """(id, whether a higher score is better, None when Kaggle does not say) of competition ``slug``."""
    comp = post(GET_COMPETITION, {"competitionName": slug})
    comp = comp.get("competition", comp) if isinstance(comp, dict) else {}
    cid = comp.get("id")
    if not cid:
        raise WebError(f"no competition named {slug!r}")
    is_max = (comp.get("evaluationAlgorithm") or {}).get("isMax")
    return int(cid), (bool(is_max) if is_max is not None else None)


def rank_writeups(board: dict, slug: str) -> tuple[list[dict], str]:
    """The teams of a GetLeaderboard answer that posted a write-up, best rank first, and which board ranked them
    ('private', 'public', or 'none' when neither lists the team)."""
    ranks: dict = {}
    by = "none"
    for kind in ("private", "public"):
        rows = board.get(f"{kind}Leaderboard") or []
        for pos, row in enumerate(rows, start=1):
            tid = row.get("teamId")
            if tid is not None and tid not in ranks:
                ranks[tid] = (int(row.get("rank") or pos), row.get("displayScore") or "", row.get("medal") or "")
        if ranks:
            by = kind
            break
    out = []
    for team in board.get("teams") or []:
        path = team.get("solutionWriteUpUrl") or ""
        if not path:
            continue
        rank, score, medal = ranks.get(team.get("teamId"), (None, "", ""))
        out.append({"rank": rank, "team": team.get("teamName") or "", "score": score,
                    "medal": medal.replace("MEDAL_", "").lower() if medal else "",
                    "url": path if path.startswith("http") else BASE + path,
                    "topic_id": team.get("writeUpForumTopicId")})
    out.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0))
    return out, by


def writeups(post, slug: str) -> dict:
    cid, is_max = competition_id(post, slug)
    rows, by = rank_writeups(post(GET_LEADERBOARD, {"competitionId": cid}), slug)
    return {"writeups": rows, "ranked_by": by, "greater_is_better": is_max}


def topic(post, topic_id: int) -> dict:
    t = (post(GET_TOPIC, {"forumTopicId": int(topic_id), "includeComments": False}) or {}).get("forumTopic") or {}
    wu = t.get("writeUp") or {}
    msg = wu.get("message") or {}
    md = (msg.get("rawMarkdown") or msg.get("content") or "").strip()
    if not md and not t:
        raise WebError(f"topic {topic_id} not found")
    url = wu.get("url") or t.get("url") or ""
    return {"title": wu.get("title") or t.get("name") or "", "subtitle": wu.get("subtitle") or "",
            "author": t.get("authorUserName") or "", "posted": wu.get("publishTime") or t.get("postDate") or "",
            "url": url if not url or url.startswith("http") else BASE + url, "markdown": md}


def _view(post, ref: str, version: int | None = None, tab: str = "output") -> dict:
    owner, _, slug = ref.partition("/")
    body = {"authorUserName": owner, "kernelSlug": slug, "tab": tab}
    if version is not None:
        body["versionNumber"] = int(version)
    v = post(VIEW_MODEL, body)
    if not isinstance(v, dict) or not (v.get("kernel") or {}).get("id"):
        raise WebError(f"notebook {ref} not found (private, deleted, or a wrong owner/slug)")
    return v


def fork_parent(view: dict) -> str | None:
    """``owner/slug`` of the notebook this one was forked from, None when it was not forked."""
    fp = (view.get("kernel") or {}).get("forkParent") or {}
    return code_ref(fp.get("url") or "") if fp else None


def view(post, ref: str) -> dict:
    """A notebook's fork parent, the inputs its latest version mounts and its Docker image's package list URL."""
    v = _view(post, ref, tab="input")
    sources = []
    by_mount = {}
    for d in v.get("renderableDataSources") or []:
        if isinstance(d, dict):
            by_mount[str(d.get("name") or "")] = d.get("reference") or ""
    for d in v.get("dataSources") or []:
        kind = str(d.get("sourceType") or "DATA_SOURCE_TYPE_COMPETITION").replace("DATA_SOURCE_TYPE_", "").lower()
        sources.append({"kind": kind, "mount": d.get("mountSlug") or "", "id": d.get("sourceId")})
    return {"fork_parent": fork_parent(v), "sources": sources, "references": by_mount,
            "requirements_url": v.get("requirementsFileDownloadUrl") or "",
            "current_version": v.get("currentVersionNumber"), "title": (v.get("kernel") or {}).get("title") or ""}


def versions(post, ref: str, max_versions: int = MAX_VERSIONS, budget: float = 90.0) -> dict:
    """Every version of ``ref`` (newest ``max_versions`` scored) with the public score of its submission."""
    end = time.monotonic() + budget
    first = _view(post, ref)
    kid = int(first["kernel"]["id"])
    total = int(first.get("totalVersionCount") or 0)
    items = (post(LIST_VERSIONS, {"kernelId": kid, "sortOption": "VERSION_ID",
                                  "pageSize": max(total, 200)}) or {}).get("items") or []
    rows = []
    for it in items:
        ver, run_ = it.get("version") or {}, it.get("run") or {}
        if ver.get("versionNumber") and run_.get("id"):
            rows.append({"version": int(ver["versionNumber"]), "name": ver.get("versionName") or "",
                         "status": str(run_.get("status") or "").lower(), "date": run_.get("dateCreated") or "",
                         "session_id": int(run_["id"]), "score": None, "scored": False})
    rows.sort(key=lambda r: -r["version"])
    for r in rows[:max_versions]:
        if time.monotonic() >= end:
            break
        try:
            sub = (_view(post, ref, r["version"]).get("submission") or {})
        except WebError:
            continue
        r["score"], r["scored"] = sub.get("scoreFormatted") or None, True
    best = (first.get("bestSubmissionScore") or {})
    return {"ref": ref, "title": first["kernel"].get("title") or "", "total": total or len(rows),
            "best": best.get("scoreFormatted") or None, "best_version": best.get("kernelVersionNumber"),
            "fork_parent": fork_parent(first), "versions": rows}


def direction(rows: list[dict], best: str | None) -> bool | None:
    """Whether a higher score is better, read off Kaggle's best score for the notebook: True when it is the highest
    version score, False when it is the lowest, None when that cannot be told (fewer than two distinct scores)."""
    scores = [score_num(r.get("score")) for r in rows]
    scores = [s for s in scores if s is not None]
    b = score_num(best)
    if b is None or len(set(scores)) < 2:
        return None
    lo, hi = min(scores), max(scores)
    tol = 1e-9 * max(1.0, abs(b))
    if abs(b - hi) <= tol:
        return True
    if abs(b - lo) <= tol:
        return False
    return None


def pick_best(rows: list[dict], greater_is_better: bool) -> dict | None:
    """The version with the best public score; the newest of equal ones. None when no version has a score."""
    scored = [r for r in rows if score_num(r.get("score")) is not None]
    if not scored:
        return None
    return max(scored, key=lambda r: ((score_num(r["score"]) if greater_is_better else -score_num(r["score"])),
                                      r["version"]))


def handle(req: dict, post, post_text) -> dict:
    op = req.get("op")
    if op == "writeups":
        return writeups(post, str(req["competition"]))
    if op == "topic":
        return topic(post, int(req["topic_id"]))
    if op == "versions":
        return versions(post, str(req["ref"]), int(req.get("max_versions") or MAX_VERSIONS),
                        float(req.get("budget") or 90))
    if op == "view":
        return view(post, str(req["ref"]))
    if op == "source":
        text = post_text(SOURCE, {"kernelSessionId": int(req["session_id"]), "includeOutputIfAvailable": False})
        return {"source": text}
    raise WebError(f"unknown op {op!r}")


def _session(kc):
    """The SDK client's requests session, with the XSRF cookie seeded, and a function posting to /api/i."""
    http = getattr(kc, "_http_client", None)
    s = getattr(http, "_session", None)
    if s is None:
        raise WebError("this Kaggle SDK keeps no session kgkit can use (pip install -U kaggle)")
    r = s.get(BASE, timeout=30)
    xsrf = s.cookies.get("XSRF-TOKEN")
    if not xsrf:
        raise WebError(f"Kaggle set no XSRF cookie (HTTP {r.status_code})")

    def send(method: str, body: dict):
        for attempt in range(3):
            r = s.post(f"{BASE}/api/i/{method}", json=body, headers={"X-XSRF-TOKEN": xsrf}, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code in (401, 403):
                raise WebError(f"{method.split('/')[-1]}: HTTP {r.status_code} (not signed in, or no access)")
            if r.status_code == 404:
                raise WebError(f"{method.split('/')[-1]}: not found (HTTP 404)")
            if not r.ok:
                raise WebError(f"{method.split('/')[-1]}: HTTP {r.status_code}")
            return r
        raise WebError(f"{method.split('/')[-1]}: Kaggle kept answering HTTP {r.status_code}")

    def post(method: str, body: dict):
        try:
            return send(method, body).json()
        except ValueError:
            raise WebError(f"{method.split('/')[-1]}: the answer was not JSON (the endpoint changed?)") from None

    return post, (lambda method, body: send(method, body).text)


def main() -> int:
    out, sys.stdout = sys.stdout, sys.stderr  # the SDK's prints must not corrupt the JSON line
    try:
        req = json.loads(sys.stdin.read())
        from .kaggle_search import _client

        with _client() as kc:
            post, post_text = _session(kc)
            res = handle(req, post, post_text)
    except ImportError as e:
        res = {"error": f"this Python has no Kaggle SDK ({e}; pip install -U kaggle)"}
    except SystemExit:  # authenticate() exits when no credentials are found
        res = {"error": "not authenticated with the Kaggle API (kaggle auth login; see skill kaggle-cli)"}
    except WebError as e:
        res = {"error": str(e)[:300]}
    except BaseException as e:  # report it, never raise
        res = {"error": f"{type(e).__name__}: {e}"[:300]}
    finally:
        sys.stdout = out
    out.write(json.dumps(res) + "\n")
    out.flush()
    return 0


def run(req: dict, timeout: float = 120) -> dict:
    """Answer ``req`` in a child process under the Kaggle CLI's interpreter. Never raises: ``{"error": why}`` when
    the internal API could not answer (offline, not signed in, an endpoint changed)."""
    if os.environ.get("KGKIT_OFFLINE"):
        return {"error": "KGKIT_OFFLINE is set"}
    from . import gpu, safety

    try:
        r = subprocess.run([gpu.kaggle_python() or sys.executable, "-m", "kgkit.kaggle_web"], input=json.dumps(req),
                           capture_output=True, encoding="utf-8", errors="replace", timeout=timeout + GRACE,
                           cwd=str(Path(__file__).resolve().parents[1]), env=safety.kaggle_env())
    except subprocess.TimeoutExpired:
        return {"error": "Kaggle's web API timed out"}
    except OSError as e:
        return {"error": f"the web API lookup did not start: {e}"}
    try:
        res = json.loads((r.stdout or "").strip().splitlines()[-1])
        if not isinstance(res, dict):
            raise ValueError
    except (ValueError, IndexError):
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        return {"error": "the web API lookup failed" + (f": {safety.clean(tail[-1])[:300]}" if tail
                                                        else f" (exit {r.returncode})")}
    if res.get("error"):
        res["error"] = safety.clean(str(res["error"]))[:300]
    return res


if __name__ == "__main__":
    sys.exit(main())
