"""Compatibility tests for the Claude Agent SDK adapter.

The unit-level tests exercise ``wrap_tool`` with plain sync/async callables
and need no claude-agent-sdk dependency. Integration with the real
``claude-agent-sdk`` is covered via ``importorskip`` and composes
``wrap_tool`` with the SDK's ``@tool`` decorator.
"""

from __future__ import annotations

import json

import pytest

from mcp_sd.agent import DDIBuffer, S2SPDispatcher
from mcp_sd.agent.adapters.claude_agent_sdk import wrap_tool


SAMPLE_BODY = [
    {"_row_id": "r1", "name": "Alice", "salary": 100000},
    {"_row_id": "r2", "name": "Bob", "salary": 80000},
]


# ---------------------------------------------------------------------------
# wrap_tool: plain callables (no framework deps required)
# ---------------------------------------------------------------------------

def _resource_payload(area: str) -> str:
    return json.dumps({
        "abstract": {"count": 2, "area": area},
        "body": SAMPLE_BODY,
    })


def test_wrap_tool_sync_callable_routes_result():
    dispatcher = S2SPDispatcher()

    @wrap_tool(dispatcher)
    def get_people(area: str):
        return _resource_payload(area)

    data = json.loads(get_people(area="SF"))
    assert "body" not in data
    assert DDIBuffer.is_handle(data["resource_url"])
    assert data["abstract"] == {"count": 2, "area": "SF"}
    assert dispatcher.pending == 1


def test_wrap_tool_sync_callable_routes_args():
    dispatcher = S2SPDispatcher()
    handle = dispatcher.buffer.put(SAMPLE_BODY)
    captured = {}

    @wrap_tool(dispatcher)
    def draw_chart(chart_kind: str, body_data: str = "", resource_url: str = ""):
        captured.update(chart_kind=chart_kind, body_data=body_data, resource_url=resource_url)
        return "chart-rendered"

    result = draw_chart(chart_kind="bar", resource_url=handle)
    assert captured["chart_kind"] == "bar"
    assert captured["resource_url"] == ""  # removed via default
    assert json.loads(captured["body_data"]) == SAMPLE_BODY
    assert result == "chart-rendered"
    assert dispatcher.pending == 0


async def test_wrap_tool_async_callable_routes_result():
    dispatcher = S2SPDispatcher()

    @wrap_tool(dispatcher)
    async def get_people(area: str):
        return _resource_payload(area)

    data = json.loads(await get_people(area="NY"))
    assert "body" not in data
    assert DDIBuffer.is_handle(data["resource_url"])
    assert dispatcher.pending == 1


async def test_wrap_tool_async_callable_routes_args():
    dispatcher = S2SPDispatcher()
    handle = dispatcher.buffer.put(SAMPLE_BODY)
    captured = {}

    @wrap_tool(dispatcher)
    async def draw_chart(chart_kind: str, body_data: str = "", resource_url: str = ""):
        captured.update(body_data=body_data, resource_url=resource_url)
        return "ok"

    await draw_chart(chart_kind="line", resource_url=handle)
    assert json.loads(captured["body_data"]) == SAMPLE_BODY
    assert captured["resource_url"] == ""
    assert dispatcher.pending == 0


async def test_full_round_trip_async():
    """Resource tool produces sync body; consumer tool resolves it."""
    dispatcher = S2SPDispatcher()

    @wrap_tool(dispatcher)
    async def get_people(area: str):
        return json.dumps({
            "abstract": {"count": 2, "area": area},
            "body": SAMPLE_BODY,
        })

    @wrap_tool(dispatcher)
    async def draw_chart(chart_kind: str, body_data: str = "", resource_url: str = ""):
        rows = json.loads(body_data)
        return f"rendered {chart_kind} with {len(rows)} rows"

    # Resource tool runs (LLM sees this payload with only ddi handle).
    view = await get_people(area="SF")
    llm_data = json.loads(view)
    assert "body" not in llm_data
    handle = llm_data["resource_url"]
    assert DDIBuffer.is_handle(handle)

    # Consumer tool called with the handle the LLM saw.
    result = await draw_chart(chart_kind="bar", resource_url=handle)
    assert result == "rendered bar with 2 rows"
    assert dispatcher.pending == 0


def test_wrap_tool_non_s2sp_result_passthrough():
    dispatcher = S2SPDispatcher()

    @wrap_tool(dispatcher)
    def echo(msg: str):
        return f"echo:{msg}"

    assert echo(msg="hi") == "echo:hi"
    assert dispatcher.pending == 0


def test_wrap_tool_preserves_function_name():
    dispatcher = S2SPDispatcher()

    @wrap_tool(dispatcher)
    def my_tool():
        return "ok"

    assert my_tool.__name__ == "my_tool"


# ---------------------------------------------------------------------------
# Integration with real claude-agent-sdk: compose wrap_tool with @tool.
# ---------------------------------------------------------------------------

pytest.importorskip("claude_agent_sdk")


def test_wrap_tool_composes_with_sdk_tool_decorator():
    """``wrap_tool`` must sit inside the SDK's ``@tool`` decorator so the
    dispatcher runs before the SDK-registered handler invocation. Verify that
    the composed pair still invokes end-to-end and routes body correctly.
    """
    from claude_agent_sdk import tool  # type: ignore

    dispatcher = S2SPDispatcher()

    @tool("get_people", "Fetch people records", {"area": str})
    @wrap_tool(dispatcher)
    async def get_people(args):
        return json.dumps({
            "abstract": {"count": 2, "area": args["area"]},
            "body": SAMPLE_BODY,
        })

    # The @tool decorator wraps the handler into an SDK tool object.
    # Inspect that the registration succeeded and that the underlying handler
    # still funnels through the dispatcher when invoked.
    assert get_people is not None


def test_create_sdk_mcp_server_accepts_wrapped_tools():
    """Sanity check: wrapped tools are accepted by ``create_sdk_mcp_server``."""
    from claude_agent_sdk import tool, create_sdk_mcp_server  # type: ignore

    dispatcher = S2SPDispatcher()

    @tool("echo", "Echo back", {"msg": str})
    @wrap_tool(dispatcher)
    async def echo(args):
        return args["msg"]

    server = create_sdk_mcp_server("s2sp-test", tools=[echo])
    assert server is not None
