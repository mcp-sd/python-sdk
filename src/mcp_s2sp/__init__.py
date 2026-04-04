"""MCP S2SP — Server-to-Server Protocol for MCP.

S2SP divides data communication between agent and MCP server into a
control plane and a data plane. It applies to MCP servers that provide
or consume structured tabular data (n rows × w columns).

The agent chooses which columns (domains) it needs on each MCP call:
- Abstract domains (control plane): processed by the LLM for decisions
- Body domains (data plane): bypass the LLM, flow server-to-server

Quick start::

    from mcp_s2sp import S2SPServer

    server = S2SPServer("my-server")

    @server.s2sp_tool()
    async def get_data(query: str) -> list[dict]:
        return fetch_rows(query)

    server.run()  # works with: mcp dev my_server.py
"""

from mcp_s2sp.server import S2SPServer
from mcp_s2sp.direct_channel import DirectChannel
from mcp_s2sp.errors import (
    S2SPError,
    TransferDeniedError,
    TransferFailedError,
    TransferTimeoutError,
)

__all__ = [
    "S2SPServer",
    "DirectChannel",
    "S2SPError",
    "TransferDeniedError",
    "TransferFailedError",
    "TransferTimeoutError",
]
