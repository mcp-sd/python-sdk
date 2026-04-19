"""S2SP agent-side dispatcher.

The dispatcher exposes two hooks that an agent framework should call at
the edges of its tool-call lifecycle:

* :meth:`S2SPDispatcher.on_tool_result` — run after a tool returns, before
  the result is appended to the LLM conversation. For sync-mode S2SP
  responses it stashes the inline ``body`` in the DDI buffer and rewrites
  the payload so only ``abstract`` + an opaque handle reach the LLM.

* :meth:`S2SPDispatcher.on_tool_call` — run before a tool is invoked,
  after the LLM has produced its arguments. If the args carry a
  ``ddi://`` handle (typically in ``resource_url``) the dispatcher pops
  the matching body from the buffer and inlines it as ``body_data``.
  Async-mode HTTP URLs pass through untouched for the consumer server to
  fetch directly.

After these two rewrites, both sync and async MCP-SD flows look identical
from the LLM's perspective: it sees an abstract view plus a short handle,
and the consumer tool always receives a concrete ``body_data`` argument.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from mcp_sd.agent.buffer import DDIBuffer


class S2SPDispatcher:
    """Coordinates agent-side body routing for S2SP tool calls."""

    def __init__(self, buffer: Optional[DDIBuffer] = None) -> None:
        # Use `is None` rather than truthiness: a freshly constructed DDIBuffer
        # has __len__ == 0 and would be replaced by `or`, silently discarding
        # a shared buffer passed in by a caller.
        self.buffer = buffer if buffer is not None else DDIBuffer()

    # ------------------------------------------------------------------
    # Tool-result path (resource server -> agent)
    # ------------------------------------------------------------------
    def on_tool_result(self, tool_name: str, result: Any) -> Any:
        """Rewrite a tool result so body rows never reach the LLM.

        If ``result`` is a JSON string describing a sync-mode S2SP response
        (contains both ``abstract`` and ``body`` keys), the body is moved
        to the DDI buffer and replaced with a ``ddi://`` handle placed
        under ``resource_url``. All other results pass through unchanged.
        """
        if not isinstance(result, (str, bytes)):
            return result

        payload = result.decode("utf-8") if isinstance(result, bytes) else result
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return result

        if not (isinstance(data, dict) and "abstract" in data and "body" in data):
            return result

        body = data.pop("body")
        handle = self.buffer.put(body)
        data["resource_url"] = handle
        return json.dumps(data)

    # ------------------------------------------------------------------
    # Tool-call path (agent -> consumer server)
    # ------------------------------------------------------------------
    def on_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Rewrite tool arguments so ``ddi://`` handles become inline body.

        If ``args`` contains a ``resource_url`` pointing at a DDI handle,
        the matching body is popped from the buffer and placed under
        ``body_data`` (as a JSON string). An HTTP ``resource_url`` is
        left untouched so the consumer server can fetch it over S2SP's
        data-plane HTTP endpoint.
        """
        if not args:
            return args

        url = args.get("resource_url")
        if not url or not DDIBuffer.is_handle(url):
            return args

        rows = self.buffer.take(url)
        new_args = dict(args)
        new_args.pop("resource_url", None)
        new_args["body_data"] = json.dumps(rows)
        return new_args

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Drop any stashed body rows (e.g., at end of a conversation)."""
        self.buffer.clear()

    @property
    def pending(self) -> int:
        """Number of unreleased body entries in the DDI buffer."""
        return len(self.buffer)
