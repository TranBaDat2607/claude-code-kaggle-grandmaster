"""The team's submissions in a competition, from Kaggle's API: every scored one, the ones selected as finals and, on
request, where the team stands on the leaderboard.

``kaggle competitions submissions`` has no option for the selected group (``SubmissionGroup.SUBMISSION_GROUP_SELECTED``
is in the SDK), and its API helper drops the page token, so this lists the groups through the SDK client, as the
website does. Kaggle has no call that *selects* finals; only the website does.

The leaderboard (``"leaderboard": true`` in the request) is the listing the website shows: the private one once the
competition has ended, ranked in its order. ``kaggle competitions leaderboard --download`` is no substitute: it always
returns the public leaderboard. The team is found by the team name its submissions carry.

``kgkit.finals.live`` runs this module in a child process (``python -m kgkit.kaggle_submissions``, a JSON request on
stdin, one JSON line on stdout) under the Kaggle CLI's interpreter, with VERBOSE scrubbed from the environment and a
hard timeout; whatever the SDK prints goes to stderr. It must stay standard-library only at import time: that
interpreter may have nothing but the Kaggle CLI installed.
"""

from __future__ import annotations

import json
import sys

from .kaggle_search import _client  # an authenticated KaggleClient, built as the Kaggle CLI builds it

PAGE = 200  # submissions per page (the API's maximum)
LB_PAGES = 150  # leaderboard pages of PAGE teams: 30,000 teams


def _score(v) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _digits(*values) -> int | None:
    """The decimals Kaggle shows, from the strings the API returns ('0.930' -> 3): a float drops trailing zeros."""
    for v in values:
        s = str(v if v is not None else "").strip()
        if "." in s and "e" not in s.lower():
            return len(s.split(".")[1])
    return None


def row(s) -> dict:
    """A submission as plain JSON: ref, date, description, file, public and private score (None when unscored, and
    the private one until Kaggle reveals it), the decimals Kaggle shows them with, status, team name."""
    date = getattr(s, "date", None)
    status = getattr(s, "status", None)
    public, private = getattr(s, "public_score", None), getattr(s, "private_score", None)
    return {"ref": getattr(s, "ref", None), "date": date.isoformat() if hasattr(date, "isoformat") else str(date or ""),
            "description": str(getattr(s, "description", "") or ""), "file": str(getattr(s, "file_name", "") or ""),
            "public_score": _score(public), "private_score": _score(private), "digits": _digits(public, private),
            "status": getattr(status, "name", str(status or "")), "team_name": str(getattr(s, "team_name", "") or "")}


def _board(kc):
    from kagglesdk.competitions.types.competition_api_service import ApiGetLeaderboardRequest

    def board(competition: str, team: str, public: bool = False, stop_at_team: bool = False) -> dict:
        """Scores in rank order, the ranks of the rows named ``team`` and whether the listing was read to its end.
        ``public``: the public leaderboard even after the competition has ended."""
        scores: list = []
        ranks: list[int] = []
        token, complete = "", False
        for _ in range(LB_PAGES):
            r = ApiGetLeaderboardRequest()
            r.competition_name = competition
            r.page_size = PAGE
            if public:
                r.override_public = True
            if token:
                r.page_token = token
            resp = kc.competitions.competition_api_client.get_leaderboard(r)
            for s in resp.submissions or []:
                scores.append(_score(getattr(s, "score", None)))
                if team and str(getattr(s, "team_name", "") or "") == team:
                    ranks.append(len(scores))
            token = resp.next_page_token
            if not token:
                complete = True
                break
            if stop_at_team and ranks:
                break
        return {"scores": scores, "ranks": ranks, "complete": complete}
    return board


def standing(kc, competition: str, scored: list[dict]) -> dict:
    """The team's place on the leaderboard the website shows (the private one once revealed: then the public rank
    too), found by the team name of its submissions."""
    team = next((r["team_name"] for r in scored if r.get("team_name")), "")
    board = _board(kc)
    out = {"team": team, "shown": board(competition, team), "public": None}
    if team and any(r.get("private_score") is not None for r in scored):
        out["public"] = board(competition, team, public=True, stop_at_team=True)
    return out


def _lister(kc):
    from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
    from kagglesdk.competitions.types.competition_enums import SubmissionGroup, SubmissionSortBy

    def list_group(competition: str, group: str, max_pages: int) -> list[dict]:
        rows, token = [], ""
        for _ in range(max_pages):
            r = ApiListSubmissionsRequest()
            r.competition_name = competition
            r.group = getattr(SubmissionGroup, group)
            r.sort_by = SubmissionSortBy.SUBMISSION_SORT_BY_DATE
            r.page_size = PAGE
            if token:
                r.page_token = token
            else:
                r.page = 1
            resp = kc.competitions.competition_api_client.list_submissions(r)
            rows += [row(s) for s in resp.submissions or []]
            token = resp.next_page_token
            if not token:
                break
        return rows
    return list_group


def main() -> int:
    out, sys.stdout = sys.stdout, sys.stderr  # the SDK's prints must not corrupt the JSON line
    res = {"selected": [], "scored": [], "error": None}
    try:
        req = json.loads(sys.stdin.read())
        with _client() as kc:
            list_group = _lister(kc)
            res["selected"] = list_group(req["competition"], "SUBMISSION_GROUP_SELECTED", 1)
            res["scored"] = list_group(req["competition"], "SUBMISSION_GROUP_SUCCESSFUL", int(req.get("max_pages") or 5))
            if req.get("leaderboard"):
                try:  # the submissions stand on their own when the leaderboard cannot be read
                    res["leaderboard"] = standing(kc, req["competition"], res["scored"])
                except Exception as e:
                    res["leaderboard_error"] = f"{type(e).__name__}: {e}"[:300]
    except ImportError as e:
        res["error"] = f"this Python has no Kaggle SDK ({e}; pip install -U kaggle)"
    except SystemExit:  # authenticate() exits when no credentials are found
        res["error"] = "not authenticated with the Kaggle API (kaggle auth login; see skill kaggle-cli)"
    except BaseException as e:  # report it, never raise
        res["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        sys.stdout = out
    out.write(json.dumps(res) + "\n")
    out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
