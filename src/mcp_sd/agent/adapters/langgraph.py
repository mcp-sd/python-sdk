"""LangGraph adapter for S2SP.

Wires :class:`~mcp_sd.agent.S2SPDispatcher` into a LangGraph state graph
so any tool executed via the standard ``ToolNode`` pattern participates in
MCP-SD's agent-side body routing.

The primary helpers are:

* :func:`wrap_tool` — decorate a plain callable or a LangChain ``BaseTool``
  subclass so its I/O flows through the dispatcher.
* :func:`make_sd_tool_node` — build a ``ToolNode`` drop-in whose tools
  are all wrapped against a shared dispatcher.

Usage::

    from langgraph.graph import StateGraph
    from mcp_sd.agent import S2SPDispatcher
    from mcp_sd.agent.adapters.langgraph import make_sd_tool_node

    dispatcher = S2SPDispatcher()

    tools = [get_alerts, draw_chart]          # your tool functions
    tool_node = make_sd_tool_node(tools, dispatcher=dispatcher)

    graph = StateGraph(...)
    graph.add_node("tools", tool_node)
    ...

Checkpointing
-------------
The DDI buffer lives in agent process memory. If your LangGraph runner
persists graph state through a checkpointer (``MemorySaver``,
``SqliteSaver``, etc.) and you expect S2SP flows to survive recovery,
serialize the buffer alongside the graph state via
:func:`snapshot_buffer` / :func:`restore_buffer`. Otherwise any unreleased
body rows are lost on restart — the same semantics as an expired async
``resource_url``.

This module only imports LangGraph lazily.  Install the extra with::

    pip install mcp-sd[langgraph]
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable, Iterable, List, Optional

from mcp_sd.agent.buffer import DDIBuffer
from mcp_sd.agent.dispatch import S2SPDispatcher


# ---------------------------------------------------------------------------
# Tool-level wrapping
# ---------------------------------------------------------------------------

def wrap_tool(dispatcher: S2SPDispatcher, tool: Any) -> Any:
    """Return a LangGraph-compatible wrapper around ``tool``.

    Accepts either:

    * a plain Python callable (sync or async) — a wrapped callable is
      returned with the same signature;
    * a LangChain ``BaseTool`` instance — its ``_run`` / ``_arun`` are
      wrapped in-place and the same tool object is returned (so it keeps
      its name, description, and args schema).
    """
    try:
        from langchain_core.tools import BaseTool  # type: ignore
    except ImportError:
        BaseTool = None  # type: ignore[assignment]

    if BaseTool is not None and isinstance(tool, BaseTool):
        return _wrap_basetool(dispatcher, tool)
    if callable(tool):
        return _wrap_callable(dispatcher, tool)
    raise TypeError(
        f"Unsupported tool type {type(tool)!r}; expected a callable or "
        "langchain_core.tools.BaseTool instance."
    )


def _wrap_callable(dispatcher: S2SPDispatcher, func: Callable[..., Any]) -> Callable[..., Any]:
    tool_name = getattr(func, "__name__", "tool")

    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            kwargs = dispatcher.on_tool_call(tool_name, kwargs)
            result = await func(*args, **kwargs)
            return dispatcher.on_tool_result(tool_name, result)
        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        kwargs = dispatcher.on_tool_call(tool_name, kwargs)
        result = func(*args, **kwargs)
        return dispatcher.on_tool_result(tool_name, result)
    return sync_wrapper


def _wrap_basetool(dispatcher: S2SPDispatcher, tool: Any) -> Any:
    """In-place wrap of a LangChain BaseTool's _run / _arun methods."""
    tool_name = getattr(tool, "name", tool.__class__.__name__)

    original_run = getattr(tool, "_run", None)
    original_arun = getattr(tool, "_arun", None)

    if original_run is not None:
        @wraps(original_run)
        def wrapped_run(*args: Any, **kwargs: Any) -> Any:
            kwargs = dispatcher.on_tool_call(tool_name, kwargs)
            result = original_run(*args, **kwargs)
            return dispatcher.on_tool_result(tool_name, result)
        tool._run = wrapped_run  # type: ignore[attr-defined]

    if original_arun is not None:
        @wraps(original_arun)
        async def wrapped_arun(*args: Any, **kwargs: Any) -> Any:
            kwargs = dispatcher.on_tool_call(tool_name, kwargs)
            result = await original_arun(*args, **kwargs)
            return dispatcher.on_tool_result(tool_name, result)
        tool._arun = wrapped_arun  # type: ignore[attr-defined]

    return tool


# ---------------------------------------------------------------------------
# ToolNode factory
# ---------------------------------------------------------------------------

def make_sd_tool_node(
    tools: Iterable[Any],
    *,
    dispatcher: Optional[S2SPDispatcher] = None,
    **tool_node_kwargs: Any,
):
    """Build a LangGraph ``ToolNode`` whose tools all go through a dispatcher.

    Returns a tuple ``(tool_node, dispatcher)``. Pass the same dispatcher
    back in later calls to reuse its DDI buffer across graph invocations.
    """
    try:
        from langgraph.prebuilt import ToolNode  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "langgraph is not installed. "
            "Install with: pip install mcp-sd[langgraph]"
        ) from e

    dispatcher = dispatcher or S2SPDispatcher()
    wrapped = [wrap_tool(dispatcher, t) for t in tools]
    node = ToolNode(wrapped, **tool_node_kwargs)
    return node, dispatcher


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def snapshot_buffer(dispatcher: S2SPDispatcher) -> dict:
    """Serialize the DDI buffer to a plain dict suitable for checkpointing."""
    # DDIBuffer._store is a dict[str, list[dict]] of JSON-safe values.
    return dict(dispatcher.buffer._store)  # noqa: SLF001


def restore_buffer(dispatcher: S2SPDispatcher, snapshot: dict) -> None:
    """Rehydrate a dispatcher's DDI buffer from a snapshot dict."""
    dispatcher.buffer._store = dict(snapshot)  # noqa: SLF001


__all__ = [
    "wrap_tool",
    "make_sd_tool_node",
    "snapshot_buffer",
    "restore_buffer",
]
