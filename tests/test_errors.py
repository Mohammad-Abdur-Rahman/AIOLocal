"""Error responses must never carry exception text (CWE-209)."""

from __future__ import annotations

import logging

from aiolocal.errors import safe_failure


def test_safe_failure_returns_only_the_curated_message(caplog):
    exc = OSError(r"C:\Users\someone\secret\models\graph.xml is missing")
    with caplog.at_level(logging.ERROR, logger="aiolocal.errors"):
        message = safe_failure(exc, "loading the model", "The model could not be loaded.")

    assert message == "The model could not be loaded."
    assert "secret" not in message
    assert "graph.xml" not in message
    # The detail is still available to maintainers in the log.
    assert "graph.xml" in caplog.text
