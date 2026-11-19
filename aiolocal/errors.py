"""Keep internal exception detail out of HTTP responses.

Exception text and type names can reveal file paths, SQL, library versions and
the internal structure of the app. CodeQL flags that as
``py/stack-trace-exposure`` (CWE-209). Endpoints should log the full exception
and return a short, curated message that the UI can show safely.

Usage::

    except OSError as exc:
        return _err(500, safe_failure(exc, "writing the config", "The config could not be written."), "config_error")

The first argument is the exception (for the log), the second names the action
and the third is the message the user sees.
"""

from __future__ import annotations

import logging

log = logging.getLogger("aiolocal.errors")


def safe_failure(exc: BaseException, action: str, message: str) -> str:
    """Log ``exc`` with a traceback and return ``message`` unchanged.

    The return value never contains exception text, so it is safe to place in an
    HTTP error response. ``message`` should be written for the user and should
    say what to check next.
    """
    try:
        log.error(
            "%s failed",
            action,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    except Exception:  # pragma: no cover - logging must never mask the original error
        pass
    return message
