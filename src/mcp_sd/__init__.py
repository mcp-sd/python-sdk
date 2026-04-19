"""mcp-sd - reference implementation of MCP-SD.

This package is *S2SP* (Server-to-Server Protocol), the reference
implementation of *MCP-SD* (Selective Disclosure for MCP).

MCP-SD is the protocol pattern: an MCP extension that lets the agent
name the attributes it needs (abstract_domains) within a single tool
call; the remaining attributes are withheld from the LLM context.
The pattern separates a control plane (agent <-> server, carrying only
agent-selected columns) from a data plane (the transport by which
withheld columns are delivered to non-LLM consumers).

S2SP realizes the MCP-SD data plane as a dedicated HTTP channel we call
the *Direct Data Interface (DDI)*. Withheld columns flow over the DDI
either directly between MCP servers (async mode) or through the agent
process out-of-band from the LLM (sync mode).

Quick start::

    from mcp_sd import S2SPServer

    server = S2SPServer("my-server")

    @server.sd_resource_tool()
    async def get_data(query: str) -> list[dict]:
        return fetch_rows(query)

    server.run()  # works with: mcp dev my_server.py
"""

from mcp_sd.server import S2SPServer
from mcp_sd.direct_channel import DirectChannel
from mcp_sd.agent import DDIBuffer, S2SPDispatcher
from mcp_sd.errors import (
    S2SPError,
    TransferDeniedError,
    TransferFailedError,
    TransferTimeoutError,
    InvalidTokenError,
    InvalidStateTransitionError,
)

__all__ = [
    # Server side
    "S2SPServer",
    "DirectChannel",
    # Agent side (platform-agnostic)
    "DDIBuffer",
    "S2SPDispatcher",
    # Errors
    "S2SPError",
    "TransferDeniedError",
    "TransferFailedError",
    "TransferTimeoutError",
    "InvalidTokenError",
    "InvalidStateTransitionError",
]
