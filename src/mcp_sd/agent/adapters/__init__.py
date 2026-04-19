"""Framework-specific adapters for the S2SP agent dispatcher.

Each adapter is a thin layer that wires :class:`~mcp_sd.agent.S2SPDispatcher`
into the tool-call lifecycle of a particular agent framework. Adapters live
in separate modules so their upstream dependencies (langgraph, langchain,
claude-agent-sdk, ...) stay optional.

Import the adapter you need; the rest remain unimported and impose no
dependency cost::

    from mcp_sd.agent.adapters.claude_agent_sdk import wrap_tool
    from mcp_sd.agent.adapters.langgraph import make_sd_tool_node
"""

__all__: list[str] = []
