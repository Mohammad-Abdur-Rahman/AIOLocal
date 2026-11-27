from __future__ import annotations

import time
import types

from aiolocal.gateway import (
    _extract_topics,
    _unreachable,
    _upstream_message,
    classify,
)
from aiolocal.llm import UpstreamError


def _body(**extra):
    return {"messages": [{"role": "user", "content": "hello"}], **extra}


def test_small_max_tokens_is_still_a_chat_request():
    # OpenCode subagents legitimately ask for short answers; that must not skip
    # the memory pipeline anymore.
    assert classify(_body(max_tokens=8, stream=False), {}, _body(max_tokens=8)["messages"]) == "chat"


def test_explicit_side_header_is_side():
    messages = [{"role": "user", "content": "hello"}]
    assert classify(_body(), {"x-aio-side": "1"}, messages) == "side"


def test_empty_messages_is_a_probe():
    assert classify({}, {}, []) == "probe"


def test_document_mode_is_explicit():
    messages = [{"role": "user", "content": "hello"}]
    assert classify(_body(), {"x-aio-mode": "document"}, messages) == "doc"


def test_tool_loop_is_classified():
    messages = [{"role": "tool", "content": "result"}]
    assert classify({}, {}, messages) == "tool_loop"


def test_mediapipe_missing_graph_names_the_selected_model():
    exc = UpstreamError(
        404,
        '{"error":"Mediapipe graph definition with requested name is not found"}',
    )

    message = _upstream_message(exc, "qwen3.5-9b")

    assert "qwen3.5-9b is not loaded" in message
    assert "Chat dropdown" in message


def test_known_upstream_categories_are_described():
    exc = UpstreamError(500, "out of memory")

    assert "out of memory" in _upstream_message(exc, "qwen3.5-9b")


def test_unknown_upstream_detail_is_logged_not_echoed():
    # Raw server replies can name internal files; they go to the log only.
    exc = UpstreamError(500, "Traceback: /home/user/secret.py line 12")

    message = _upstream_message(exc, "qwen3.5-9b")

    assert "secret.py" not in message
    assert "rejected the request" in message


def test_unreachable_message_hides_the_exception():
    ctx = types.SimpleNamespace(
        settings=types.SimpleNamespace(upstream_base_url="http://127.0.0.1:8000/v3")
    )
    exc = OSError(r"C:\Users\someone\secret\ovms.exe is missing")

    message = _unreachable(ctx, exc)

    assert "secret" not in message
    assert "127.0.0.1:8000" in message


def test_topic_list_is_extracted_from_bullets_and_numbers():
    text = (
        "Classify every page into these topics:\n"
        "- Surds & Indices\n"
        "* Exponential Equations\n"
        "1. Equations & Inequalities\n"
        "2) Graphs & Kinematics\n"
        "\u2022 Trigonometry\n"
    )

    assert _extract_topics(text) == [
        "Surds & Indices",
        "Exponential Equations",
        "Equations & Inequalities",
        "Graphs & Kinematics",
        "Trigonometry",
    ]


def test_topic_parsing_stays_fast_on_long_space_runs():
    # Pathological input from the CodeQL report: a bullet followed by a long run
    # of spaces used to make the pattern backtrack polynomially.
    text = "* " + " " * 20000

    start = time.perf_counter()
    assert _extract_topics(text) == []
    assert time.perf_counter() - start < 1.0


def test_topic_parsing_ignores_very_long_lines():
    assert _extract_topics("- " + "a " * 5000) == []
