"""The Kaggle CLI, with Kaggle's reason when it refuses a request.

Kaggle's API answers a refused request with a JSON body such as ``{"error": {"code": 400, "message": "...",
"status": "INVALID_ARGUMENT"}}``. kagglesdk (0.1.37, Kaggle CLI 2.2.4) looks for ``code`` and ``message`` at the top
level of that body only, so the CLI prints requests' bare ``400 Client Error: Bad Request for url: ...`` and the reason
is lost. ``kgkit.gpu.run_kaggle`` runs the CLI through this file, under the CLI's own interpreter: such an error gets a
second line, ``Kaggle's reason: <message> (<status>)``, and nothing else changes.

It runs as a script (``python <path>/kaggle_cli.py <kaggle args>``) and must stay standard-library only at import
time: that interpreter may have nothing but the Kaggle CLI installed. When it cannot import the Kaggle CLI it exits
with ``NO_CLI`` before doing anything, and the caller runs the plain CLI instead.
"""

from __future__ import annotations

import os
import sys

NO_CLI = 87  # exit status when this interpreter has no Kaggle CLI: nothing was run
REASON = "Kaggle's reason:"


def reason(response) -> str | None:
    """Kaggle's message in an error response's JSON body (under ``error``, or at the top level), None without one."""
    try:
        body = response.json()
    except Exception:  # not JSON: an HTML error page, an empty body
        return None
    err = body.get("error", body) if isinstance(body, dict) else None
    if not isinstance(err, dict) or not err.get("message"):
        return None
    msg = " ".join(str(err["message"]).split())[:500]
    status = str(err.get("status") or "")
    return f"{msg} ({status})" if status and status not in msg else msg


def add_reasons(requests) -> None:
    """Give ``requests``' HTTP errors Kaggle's reason. kagglesdk raises them with ``Response.raise_for_status`` when
    it finds no top-level ``code`` in the body; the CLI prints them."""
    plain = requests.Response.raise_for_status

    def raise_for_status(self):
        try:
            return plain(self)
        except requests.HTTPError as e:
            why = reason(self)
            if not why or why in str(e):
                raise
            raise requests.HTTPError(f"{e}\n{REASON} {why}", request=e.request, response=self) from None

    requests.Response.raise_for_status = raise_for_status


def main(argv: list[str]) -> int:
    # Run as a script, this folder is sys.path[0]: kgkit's modules (cli, datasets, metrics, ...) must not shadow
    # anything the CLI imports.
    here = os.path.dirname(os.path.realpath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.realpath(p or os.curdir) != here]
    try:
        from kaggle.cli import main as kaggle_main
    except ImportError:
        return NO_CLI
    try:
        import requests

        add_reasons(requests)
    except Exception:  # an unexpected requests: the CLI's own messages
        pass
    sys.argv = ["kaggle", *argv]
    return kaggle_main()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
