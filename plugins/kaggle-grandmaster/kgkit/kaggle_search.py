"""Public scores of notebooks from Kaggle's search API (``kernel_document.best_public_score``).

``kaggle kernels list`` has no score field; the search service behind the website has one: the best public
score across a notebook's versions, at the competition's display precision, 0.0 when none is on record. The
service has no competition filter, so notebooks are matched on owner/slug: one query on the competition's name,
then one query on the title of each notebook still missing.

``kgkit.kernels.public_scores`` runs this module in a child process (``python -m kgkit.kaggle_search``, a JSON
request on stdin, one JSON line on stdout) under the Kaggle CLI's interpreter, with VERBOSE scrubbed from the
environment and a hard timeout; whatever the SDK prints goes to stderr. It must stay standard-library only at
import time: that interpreter may have nothing but the Kaggle CLI installed.
"""

from __future__ import annotations

import json
import sys
import time

PAGE = 50  # documents per page of the search API


def doc_ref(d) -> str:
    """``owner/slug`` of a search document, lowercased."""
    owner = getattr(getattr(d, "owner_user", None), "user_name", None) \
        or getattr(getattr(d, "owner_organization", None), "slug", None) or ""
    return f"{owner}/{getattr(d, 'slug', None) or ''}".lower()


def lookup(search, items, bulk_query: str = "", bulk_pages: int = 1, budget: float = 30.0) -> dict:
    """Best public score of each ``(ref, title)`` in ``items``. ``search(query, pages, by_votes)`` returns the
    documents of a query. Returns ``{"found": {ref: score, or None when none is on record}, "error": ...}``;
    refs missing from ``found`` were not found (unknown), ``error`` is the last query's failure."""
    want = {str(ref).lower(): ref for ref, _ in items}
    found: dict = {}
    error = None
    end = time.monotonic() + budget

    def run(query, pages, by_votes):
        nonlocal error
        try:
            docs = search(query, pages, by_votes)
        except Exception as e:  # one failed query leaves the others' scores
            error = f"{type(e).__name__}: {e}"[:300]
            return
        for d in docs:
            ref = want.get(doc_ref(d))
            if ref is not None and ref not in found:
                s = getattr(getattr(d, "kernel_document", None), "best_public_score", None)
                found[ref] = float(s) if s else None

    if bulk_query:
        run(bulk_query, bulk_pages, True)
    for ref, title in items:
        if ref in found:
            continue
        if time.monotonic() >= end:
            break
        run(title or str(ref).split("/")[-1].replace("-", " "), 1, False)
    return {"found": found, "error": error}


def _client():
    """An authenticated KaggleClient (a context manager), as the Kaggle CLI builds it."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api.build_kaggle_client()


def _searcher(kc):
    from kagglesdk.search.types.search_api_service import ListEntitiesFilters, ListEntitiesRequest
    from kagglesdk.search.types.search_enums import DocumentType, ListSearchContentOrderBy

    def search(query, pages, by_votes):
        docs, token = [], ""
        for _ in range(pages):
            f = ListEntitiesFilters()
            f.query, f.document_types = query, [DocumentType.KERNEL]
            r = ListEntitiesRequest()
            r.filters, r.page_size = f, PAGE
            if by_votes:
                r.canonical_order_by = ListSearchContentOrderBy.LIST_SEARCH_CONTENT_ORDER_BY_VOTES
            if token:
                r.page_token = token
            resp = kc.search.search_api_client.list_entities(r)
            docs += resp.documents
            token = resp.next_page_token
            if not token:
                break
        return docs
    return search


def main() -> int:
    out, sys.stdout = sys.stdout, sys.stderr  # the SDK's prints must not corrupt the JSON line
    try:
        req = json.loads(sys.stdin.read())
        with _client() as kc:
            res = lookup(_searcher(kc), req.get("items") or [], req.get("bulk_query") or "",
                         int(req.get("bulk_pages") or 1), float(req.get("budget") or 30))
    except ImportError as e:
        res = {"found": {}, "error": f"this Python has no Kaggle search API ({e}; pip install -U kaggle)"}
    except SystemExit:  # authenticate() exits when no credentials are found
        res = {"found": {}, "error": "not authenticated with the Kaggle API (kaggle auth login; see skill kaggle-cli)"}
    except BaseException as e:  # report it, never raise
        res = {"found": {}, "error": f"{type(e).__name__}: {e}"[:300]}
    finally:
        sys.stdout = out
    out.write(json.dumps(res) + "\n")
    out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
