"""Pure unit tests for DDIBuffer and S2SPDispatcher.

These tests exercise the agent-side dispatcher with real DDIBuffer /
S2SPDispatcher instances and have zero framework dependencies.
"""

from __future__ import annotations

import json

import pytest

from mcp_sd.agent import DDIBuffer, S2SPDispatcher


SAMPLE_BODY = [
    {"_row_id": "r1", "name": "Alice", "salary": 100000},
    {"_row_id": "r2", "name": "Bob", "salary": 80000},
]


# ---------------------------------------------------------------------------
# DDIBuffer
# ---------------------------------------------------------------------------

def test_buffer_put_returns_ddi_handle():
    buf = DDIBuffer()
    handle = buf.put(SAMPLE_BODY)
    assert handle.startswith("ddi://")
    assert DDIBuffer.is_handle(handle)
    assert len(buf) == 1
    assert handle in buf


def test_buffer_take_is_single_use():
    buf = DDIBuffer()
    handle = buf.put(SAMPLE_BODY)
    assert buf.take(handle) == SAMPLE_BODY
    # Second take returns empty list, not the stored rows.
    assert buf.take(handle) == []
    assert handle not in buf
    assert len(buf) == 0


def test_buffer_peek_is_idempotent():
    buf = DDIBuffer()
    handle = buf.put(SAMPLE_BODY)
    assert buf.peek(handle) == SAMPLE_BODY
    assert buf.peek(handle) == SAMPLE_BODY
    assert len(buf) == 1  # peek does not consume


def test_buffer_peek_missing_returns_empty_list():
    buf = DDIBuffer()
    assert buf.peek("ddi://missing") == []


def test_buffer_is_handle_classmethod():
    assert DDIBuffer.is_handle("ddi://abc")
    assert not DDIBuffer.is_handle("http://example.com")
    assert not DDIBuffer.is_handle("")
    assert not DDIBuffer.is_handle(None)
    assert not DDIBuffer.is_handle(42)
    assert not DDIBuffer.is_handle({"not": "a string"})


def test_buffer_clear_wipes_all_entries():
    buf = DDIBuffer()
    h1 = buf.put(SAMPLE_BODY)
    h2 = buf.put([{"x": 1}])
    assert len(buf) == 2
    buf.clear()
    assert len(buf) == 0
    assert h1 not in buf
    assert h2 not in buf


def test_buffer_len_and_contains():
    buf = DDIBuffer()
    assert len(buf) == 0
    handle = buf.put(SAMPLE_BODY)
    assert len(buf) == 1
    assert handle in buf
    assert "ddi://never-existed" not in buf


def test_buffer_put_generates_distinct_handles():
    buf = DDIBuffer()
    handles = {buf.put([{"i": i}]) for i in range(25)}
    assert len(handles) == 25  # all unique


# ---------------------------------------------------------------------------
# S2SPDispatcher.on_tool_result
# ---------------------------------------------------------------------------

def test_on_tool_result_sync_mode_strips_body():
    dispatcher = S2SPDispatcher()
    payload = json.dumps({
        "abstract": {"count": 2, "domains": ["name"]},
        "body": SAMPLE_BODY,
    })
    rewritten = dispatcher.on_tool_result("get_people", payload)
    data = json.loads(rewritten)

    assert "body" not in data
    assert data["abstract"] == {"count": 2, "domains": ["name"]}
    assert DDIBuffer.is_handle(data["resource_url"])
    # The body is now parked in the buffer, reachable by the new handle.
    assert dispatcher.pending == 1
    assert dispatcher.buffer.peek(data["resource_url"]) == SAMPLE_BODY


def test_on_tool_result_async_mode_passthrough():
    dispatcher = S2SPDispatcher()
    payload = json.dumps({
        "abstract": {"count": 2},
        "resource_url": "http://data.example.com/r/abc",
    })
    rewritten = dispatcher.on_tool_result("get_people", payload)
    assert rewritten == payload
    assert dispatcher.pending == 0


def test_on_tool_result_non_json_string_passthrough():
    dispatcher = S2SPDispatcher()
    out = dispatcher.on_tool_result("tool", "hello world")
    assert out == "hello world"
    assert dispatcher.pending == 0


def test_on_tool_result_json_without_body_passthrough():
    dispatcher = S2SPDispatcher()
    payload = json.dumps({"abstract": {"count": 0}})
    out = dispatcher.on_tool_result("tool", payload)
    assert out == payload
    assert dispatcher.pending == 0


@pytest.mark.parametrize("value", [42, None, {"already": "a dict"}, [1, 2, 3], 3.14])
def test_on_tool_result_non_string_passthrough(value):
    dispatcher = S2SPDispatcher()
    assert dispatcher.on_tool_result("tool", value) is value
    assert dispatcher.pending == 0


def test_on_tool_result_bytes_decoded_and_processed():
    dispatcher = S2SPDispatcher()
    payload = json.dumps({
        "abstract": {"count": 2},
        "body": SAMPLE_BODY,
    }).encode("utf-8")
    rewritten = dispatcher.on_tool_result("get_people", payload)
    assert isinstance(rewritten, str)
    data = json.loads(rewritten)
    assert DDIBuffer.is_handle(data["resource_url"])
    assert dispatcher.pending == 1


# ---------------------------------------------------------------------------
# S2SPDispatcher.on_tool_call
# ---------------------------------------------------------------------------

def test_on_tool_call_resolves_ddi_handle():
    dispatcher = S2SPDispatcher()
    handle = dispatcher.buffer.put(SAMPLE_BODY)
    args = {"resource_url": handle, "chart_kind": "bar"}

    new_args = dispatcher.on_tool_call("draw_chart", args)

    assert "resource_url" not in new_args
    assert new_args["chart_kind"] == "bar"
    assert json.loads(new_args["body_data"]) == SAMPLE_BODY
    # Handle was consumed from the buffer.
    assert dispatcher.pending == 0
    # Original args were not mutated.
    assert args["resource_url"] == handle


def test_on_tool_call_http_url_passthrough():
    dispatcher = S2SPDispatcher()
    args = {"resource_url": "http://data.example.com/r/xyz", "other": 1}
    out = dispatcher.on_tool_call("draw_chart", args)
    # HTTP URLs are left for the consumer server to fetch directly.
    assert out is args or out == args


def test_on_tool_call_no_resource_url_passthrough():
    dispatcher = S2SPDispatcher()
    args = {"chart_kind": "bar", "data": [1, 2, 3]}
    out = dispatcher.on_tool_call("draw_chart", args)
    assert out is args or out == args


def test_on_tool_call_empty_args_passthrough():
    dispatcher = S2SPDispatcher()
    assert dispatcher.on_tool_call("tool", {}) == {}
    assert dispatcher.on_tool_call("tool", None) is None


def test_on_tool_call_stale_handle_yields_empty_body():
    # A DDI handle that no longer exists in the buffer still produces
    # an inlined body_data (an empty JSON array), matching an expired async URL.
    dispatcher = S2SPDispatcher()
    args = {"resource_url": "ddi://never-issued"}
    new_args = dispatcher.on_tool_call("draw_chart", args)
    assert new_args["body_data"] == "[]"
    assert "resource_url" not in new_args


# ---------------------------------------------------------------------------
# Round-trip + lifecycle
# ---------------------------------------------------------------------------

def test_full_round_trip_sync_mode():
    """Simulate: resource tool returns sync body, consumer tool resolves it."""
    dispatcher = S2SPDispatcher()

    # Resource server response (sync mode).
    resource_response = json.dumps({
        "abstract": {"count": 2, "domains": ["name"]},
        "body": SAMPLE_BODY,
    })

    # Dispatcher rewrites on the result path.
    llm_view = dispatcher.on_tool_result("get_people", resource_response)
    llm_data = json.loads(llm_view)
    handle = llm_data["resource_url"]
    assert "body" not in llm_data
    assert DDIBuffer.is_handle(handle)

    # LLM now calls consumer with the handle it sees.
    consumer_args = {"resource_url": handle, "chart_kind": "bar"}
    resolved = dispatcher.on_tool_call("draw_chart", consumer_args)
    assert json.loads(resolved["body_data"]) == SAMPLE_BODY
    assert "resource_url" not in resolved
    assert dispatcher.pending == 0


def test_pending_and_reset():
    dispatcher = S2SPDispatcher()
    assert dispatcher.pending == 0
    dispatcher.buffer.put(SAMPLE_BODY)
    dispatcher.buffer.put([{"x": 1}])
    assert dispatcher.pending == 2

    dispatcher.reset()
    assert dispatcher.pending == 0
    assert len(dispatcher.buffer) == 0


def test_injected_buffer_is_shared():
    """An empty DDIBuffer injected into the dispatcher must be preserved,
    not silently replaced. Regression test for the ``buffer or DDIBuffer()``
    bug that truthiness-tested an empty buffer as falsy.
    """
    shared = DDIBuffer()
    d1 = S2SPDispatcher(buffer=shared)
    d2 = S2SPDispatcher(buffer=shared)

    handle = d1.buffer.put(SAMPLE_BODY)
    # d2 must be able to take what d1 stored because they share the buffer.
    assert d2.buffer.take(handle) == SAMPLE_BODY


def test_injected_non_empty_buffer_is_kept():
    """When the injected buffer is non-empty, truthiness works out by accident."""
    shared = DDIBuffer()
    shared.put([{"seed": True}])  # non-empty makes `buffer or ...` take `buffer`
    d1 = S2SPDispatcher(buffer=shared)
    d2 = S2SPDispatcher(buffer=shared)

    handle = d1.buffer.put(SAMPLE_BODY)
    # Works here only because `shared` is truthy by len.
    assert d2.buffer.take(handle) == SAMPLE_BODY
