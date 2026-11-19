from __future__ import annotations

from aiolocal.gateway import _upstream_message, classify
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


def test_other_upstream_errors_keep_their_detail():
    exc = UpstreamError(500, "out of memory")

    assert "out of memory" in _upstream_message(exc, "qwen3.5-9b")
