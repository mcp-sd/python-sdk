"""Compatibility tests for the LangGraph adapter.

Unit-level tests exercise ``wrap_tool`` with plain callables and don't need
LangGraph/LangChain. Tests that touch ``BaseTool`` or ``ToolNode`` are
guarded with ``pytest.importorskip``.
"""

from __future__ import annotations

import json

import pytest

from mcp_sd.agent import DDIBuffer, S2SPDispatcher
from mcp_sd.agent.adapters.langgraph import (
    make_sd_tool_node,
    restore_buffer,
    snapshot_buffer,
    wrap_tool,
)


SAMPLE_BODY = [
    {"_row_id": "r1", "name": "Alice", "salary": 100000},
    {"_row_id": "r2", "name": "Bob", "salary": 80000},
]


# ---------------------------------------------------------------------------
# wrap_tool on plain callables (no framework deps)
# ---------------------------------------------------------------------------

def test_wrap_tool_sync_callable():
    dispatcher = S2SPDispatcher()

    def get_people(area: str):
        return json.dumps({
            "abstract": {"count": 2, "area": area},
            "body": SAMPLE_BODY,
        })

    wrapped = wrap_tool(dispatcher, get_people)
    out = wrapped(area="SF")
    data = json.loads(out)
    assert "body" not in data
    assert DDIBuffer.is_handle(data["resource_url"])
    assert dispatcher.pending == 1
    assert wrapped.__name__ == "get_people"


async def test_wrap_tool_async_callable():
    dispatcher = S2SPDispatcher()

    async def get_people(area: str):
        return json.dumps({
            "abstract": {"count": 2, "area": area},
            "body": SAMPLE_BODY,
        })

    wrapped = wrap_tool(dispatcher, get_people)
    out = await wrapped(area="SF")
    data = json.loads(out)
    assert DDIBuffer.is_handle(data["resource_url"])
    assert dispatcher.pending == 1


def test_wrap_tool_resolves_handle_on_call():
    dispatcher = S2SPDispatcher()
    handle = dispatcher.buffer.put(SAMPLE_BODY)

    seen = {}

    def draw_chart(chart_kind: str, body_data: str = "", resource_url: str = ""):
        seen["body_data"] = body_data
        seen["resource_url"] = resource_url
        return "ok"

    wrapped = wrap_tool(dispatcher, draw_chart)
    wrapped(chart_kind="bar", resource_url=handle)
    assert json.loads(seen["body_data"]) == SAMPLE_BODY
    assert seen["resource_url"] == ""
    assert dispatcher.pending == 0


def test_wrap_tool_raises_for_unsupported_type():
    dispatcher = S2SPDispatcher()
    with pytest.raises(TypeError):
        wrap_tool(dispatcher, 42)
    with pytest.raises(TypeError):
        wrap_tool(dispatcher, "not callable")


# ---------------------------------------------------------------------------
# snapshot / restore (no framework deps)
# ---------------------------------------------------------------------------

def test_snapshot_and_restore_buffer_round_trip():
    dispatcher = S2SPDispatcher()
    handle = dispatcher.buffer.put(SAMPLE_BODY)
    assert dispatcher.pending == 1

    snap = snapshot_buffer(dispatcher)
    # Snapshot must be a plain dict we can serialize independently.
    assert isinstance(snap, dict)
    assert handle in snap
    assert snap[handle] == SAMPLE_BODY

    # Simulate crash / restart: wipe the dispatcher and recreate it.
    dispatcher.reset()
    assert dispatcher.pending == 0
    assert handle not in dispatcher.buffer

    fresh = S2SPDispatcher()
    restore_buffer(fresh, snap)
    assert fresh.pending == 1
    assert fresh.buffer.take(handle) == SAMPLE_BODY


def test_snapshot_is_decoupled_from_live_buffer():
    """Mutating the live buffer after snapshotting should not change the snapshot."""
    dispatcher = S2SPDispatcher()
    dispatcher.buffer.put(SAMPLE_BODY)
    snap = snapshot_buffer(dispatcher)

    dispatcher.reset()
    dispatcher.buffer.put([{"new": "rows"}])

    # Snapshot still has one row with the original content.
    assert len(snap) == 1
    assert list(snap.values())[0] == SAMPLE_BODY


# ---------------------------------------------------------------------------
# LangChain BaseTool path
# ---------------------------------------------------------------------------

def _make_echo_tool_class():
    """Build a trivial BaseTool subclass. Deferred so importorskip can gate it."""
    from langchain_core.tools import BaseTool  # type: ignore
    from pydantic import BaseModel

    class EchoArgs(BaseModel):
        chart_kind: str = "bar"
        body_data: str = ""
        resource_url: str = ""

    class EchoTool(BaseTool):
        name: str = "echo_tool"
        description: str = "testing tool"
        args_schema: type = EchoArgs
        captured: dict = {}

        def _run(self, chart_kind="bar", body_data="", resource_url=""):
            self.captured["sync"] = {"body_data": body_data, "resource_url": resource_url}
            return "sync-ok"

        async def _arun(self, chart_kind="bar", body_data="", resource_url=""):
            self.captured["async"] = {"body_data": body_data, "resource_url": resource_url}
            return "async-ok"

    return EchoTool


def test_wrap_tool_basetool_rewrites_run_and_arun():
    pytest.importorskip("langchain_core")
    EchoTool = _make_echo_tool_class()

    dispatcher = S2SPDispatcher()
    tool = EchoTool()
    returned = wrap_tool(dispatcher, tool)
    assert returned is tool  # in-place rewrite

    handle = dispatcher.buffer.put(SAMPLE_BODY)
    tool._run(chart_kind="bar", resource_url=handle)
    assert json.loads(tool.captured["sync"]["body_data"]) == SAMPLE_BODY
    assert tool.captured["sync"]["resource_url"] == ""
    assert dispatcher.pending == 0


async def test_wrap_tool_basetool_async_path():
    pytest.importorskip("langchain_core")
    EchoTool = _make_echo_tool_class()

    dispatcher = S2SPDispatcher()
    tool = wrap_tool(dispatcher, EchoTool())
    handle = dispatcher.buffer.put(SAMPLE_BODY)
    out = await tool._arun(chart_kind="line", resource_url=handle)
    assert out == "async-ok"
    assert json.loads(tool.captured["async"]["body_data"]) == SAMPLE_BODY
    assert dispatcher.pending == 0


# ---------------------------------------------------------------------------
# make_sd_tool_node
# ---------------------------------------------------------------------------

def test_make_sd_tool_node_wires_up():
    pytest.importorskip("langgraph.prebuilt")
    from langchain_core.tools import tool as lc_tool  # type: ignore

    @lc_tool
    def plain_echo(msg: str) -> str:
        """Return the input."""
        return msg

    # Explicit dispatcher: returned verbatim.
    dispatcher = S2SPDispatcher()
    node, returned = make_sd_tool_node([plain_echo], dispatcher=dispatcher)
    assert returned is dispatcher
    assert node is not None

    # Omitted dispatcher: factory builds one.
    node2, disp2 = make_sd_tool_node([plain_echo])
    assert isinstance(disp2, S2SPDispatcher)
    assert disp2.pending == 0
    assert node2 is not None
