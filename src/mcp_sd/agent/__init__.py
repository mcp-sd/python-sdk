"""Agent-side primitives for S2SP.

This subpackage provides the platform-agnostic logic an agent needs to
participate in MCP-SD / S2SP flows:

* :class:`DDIBuffer` — session-scoped in-process cache for sync-mode body rows
* :class:`S2SPDispatcher` — hooks that rewrite tool I/O so body data never
  reaches the LLM context

Thin platform adapters in :mod:`mcp_sd.agent.adapters` wire these hooks
into the tool-call lifecycle of specific agent frameworks (Claude Agent SDK,
LangGraph, etc.). The dispatcher itself has no framework dependencies.
"""

from mcp_sd.agent.buffer import DDIBuffer
from mcp_sd.agent.dispatch import S2SPDispatcher

__all__ = ["DDIBuffer", "S2SPDispatcher"]
