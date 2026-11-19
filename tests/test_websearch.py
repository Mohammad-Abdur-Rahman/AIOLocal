"""DuckDuckGo result unwrapping must only trust the real DuckDuckGo host."""

from __future__ import annotations

from aiolocal.websearch import _DuckParser


def _unwrap(url: str) -> str:
    parser = _DuckParser()
    parser.results = [{"title": "t", "url": url, "snippet": ""}]
    parser.close()
    return parser.results[0]["url"]


def test_duckduckgo_redirect_is_unwrapped():
    target = "https://example.com/article"
    wrapped = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Farticle"
    assert _unwrap(wrapped) == target
    assert _unwrap(
        "https://html.duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Farticle"
    ) == target


def test_lookalike_hosts_are_not_unwrapped():
    # These used to pass a netloc.endswith("duckduckgo.com") check.
    for host in ("evilduckduckgo.com", "duckduckgo.com.evil.example"):
        url = f"https://{host}/l/?uddg=https%3A%2F%2Fexample.com%2Farticle"
        assert _unwrap(url) == url


def test_protocol_relative_urls_become_https():
    assert _unwrap("//example.com/page") == "https://example.com/page"
