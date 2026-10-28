"""The team's submissions in a competition, from Kaggle's API: every scored one, and the ones selected as finals.

``kaggle competitions submissions`` has no option for the selected group (``SubmissionGroup.SUBMISSION_GROUP_SELECTED``
is in the SDK), and its API helper drops the page token, so this lists the groups through the SDK client, as the
website does. Kaggle has no call that *selects* finals; only the website does.

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


def row(s) -> dict:
    """A submission as plain JSON: ref, date, description, file, public score (None when unscored), status."""
    score = getattr(s, "public_score", None)
    try:
        score = float(score) if score not in (None, "") else None
    except (TypeError, ValueError):
        score = None
    date = getattr(s, "date", None)
    status = getattr(s, "status", None)
    return {"ref": getattr(s, "ref", None), "date": date.isoformat() if hasattr(date, "isoformat") else str(date or ""),
            "description": str(getattr(s, "description", "") or ""), "file": str(getattr(s, "file_name", "") or ""),
            "public_score": score, "status": getattr(status, "name", str(status or ""))}


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
