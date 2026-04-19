"""Claude Agent SDK adapter for S2SP.

Wires :class:`~mcp_sd.agent.S2SPDispatcher` into the tool-call lifecycle
of Anthropic's `claude-agent-sdk`_. The adapter exposes a single helper:

* :func:`wrap_tool` — decorate a tool handler so its incoming arguments
  and outgoing result pass through the dispatcher. Compose it with the
  SDK's own ``@tool`` decorator to register the wrapped handler.

Usage with ``claude-agent-sdk`` 0.1.x::

    from claude_agent_sdk import (
        ClaudeSDKClient, ClaudeAgentOptions,
        tool, create_sdk_mcp_server,
    )
    from mcp_sd.agent import S2SPDispatcher
    from mcp_sd.agent.adapters.claude_agent_sdk import wrap_tool

    dispatcher = S2SPDispatcher()

    @tool("get_alerts", "Fetch weather alerts", {"area": str, "abstract_domains": str})
    @wrap_tool(dispatcher)
    async def get_alerts(args):
        # Calls an S2SP MCP server; returns a JSON string whose sync-mode
        # body is hidden by the dispatcher before the LLM sees it.
        ...

    @tool("draw_chart", "Generate chart", {"abstract_data": str, "resource_url": str})
    @wrap_tool(dispatcher)
    async def draw_chart(args):
        # Receives either an HTTP resource_url (async) or an inlined
        # body_data (sync, resolved from ddi:// by the dispatcher).
        ...

    server = create_sdk_mcp_server("s2sp-demo", tools=[get_alerts, draw_chart])
    client = ClaudeSDKClient(options=ClaudeAgentOptions(mcp_servers={"demo": server}))

.. _claude-agent-sdk: https://github.com/anthropics/claude-agent-sdk-python

This module only imports from :mod:`claude_agent_sdk` lazily (and only for
optional type hints), so importing it does not require the dependency to
be installed. Install the extra with::

    pip install mcp-sd[claude-agent]
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable, TypeVar

from mcp_sd.agent.dispatch import S2SPDispatcher

F = TypeVar("F", bound=Callable[..., Any])


def wrap_tool(dispatcher: S2SPDispatcher) -> Callable[[F], F]:
    """Return a decorator that routes a tool's I/O through ``dispatcher``.

    The wrapped tool's invocation goes through
    :meth:`S2SPDispatcher.on_tool_call` on the way in (resolving any
    ``ddi://`` handle into inline ``body_data``) and through
    :meth:`S2SPDispatcher.on_tool_result` on the way out (stashing any
    sync-mode ``body`` into the DDI buffer and replacing it with an opaque
    handle).

    Works on both sync and async tool handlers. Place ``@wrap_tool()``
    closest to the function definition (innermost decorator) so it runs
    before the SDK's registration decorator sees the handler.
    """

    def decorator(func: F) -> F:
        tool_name = getattr(func, "__name__", "tool")

        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                kwargs = dispatcher.on_tool_call(tool_name, kwargs)
                result = await func(*args, **kwargs)
                return dispatcher.on_tool_result(tool_name, result)
            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            kwargs = dispatcher.on_tool_call(tool_name, kwargs)
            result = func(*args, **kwargs)
            return dispatcher.on_tool_result(tool_name, result)
        return sync_wrapper  # type: ignore[return-value]

    return decorator


__all__ = ["wrap_tool"]
