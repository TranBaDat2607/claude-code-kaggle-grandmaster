"""The files a kernel's last session saved, and its log, from Kaggle's API.

``kaggle kernels files`` lists nothing for a session that was stopped (a push's ``-t`` cap, the website's Stop button),
though ``kaggle kernels output`` downloads the files it had written; both are the SDK's ``list_kernel_session_output``,
which also returns the session's log. This asks that one call, without downloading anything, so ``gpu wait`` can say
which files a stopped run kept.

``kgkit.gpu.session_output`` runs this module in a child process (``python -m kgkit.kaggle_outputs``, a JSON request on
stdin, one JSON line on stdout) under the Kaggle CLI's interpreter, with VERBOSE scrubbed from the environment and a
hard timeout; whatever the SDK prints goes to stderr. It must stay standard-library only at import time: that
interpreter may have nothing but the Kaggle CLI installed.
"""

from __future__ import annotations

import json
import sys

from .kaggle_search import _client  # an authenticated KaggleClient, built as the Kaggle CLI builds it

PAGE = 200  # files per page (the API's maximum)


def _lister(kc):
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    def list_output(kernel: str, max_pages: int) -> tuple[list[str], bool, str]:
        """(file names, whether more pages follow, the session's log) of ``owner/slug``'s last session."""
        owner, _, slug = kernel.partition("/")
        names, token, log = [], "", ""
        for page in range(max_pages):
            r = ApiListKernelSessionOutputRequest()
            r.user_name, r.kernel_slug, r.page_size = owner, slug, PAGE
            if token:
                r.page_token = token
            resp = kc.kernels.kernels_api_client.list_kernel_session_output(r)
            names += [str(f.file_name) for f in resp.files or []]
            if page == 0:
                log = resp.log or ""
            token = resp.next_page_token
            if not token:
                break
        return names, bool(token), log
    return list_output


def main() -> int:
    out, sys.stdout = sys.stdout, sys.stderr  # the SDK's prints must not corrupt the JSON line
    res = {"files": [], "more": False, "log": "", "error": None}
    try:
        req = json.loads(sys.stdin.read())
        with _client() as kc:
            res["files"], res["more"], res["log"] = _lister(kc)(str(req["kernel"]), int(req.get("max_pages") or 5))
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
