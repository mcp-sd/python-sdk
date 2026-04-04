"""S2SPServer — MCP server with S2SP data-plane support.

An S2SPServer embeds a FastMCP instance and adds the ``@s2sp_tool()``
decorator for splitting tool responses into abstract (control plane)
and body (data plane) domains.

Usage::

    from mcp_s2sp import S2SPServer

    server = S2SPServer("my-server")

    @server.s2sp_tool()
    async def get_data(query: str) -> list[dict]:
        return fetch_rows(query)

    server.run()  # works with: mcp dev my_server.py
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from mcp_s2sp.direct_channel import DirectChannel

logger = logging.getLogger(__name__)


class S2SPServer:
    """MCP server with S2SP data-plane support.

    Wraps FastMCP and adds:
    - ``@server.s2sp_tool()``: decorator for domain projection
    - ``server.mcp``: the embedded FastMCP instance for custom tools/resources
    - ``server.run()``: run as MCP server (stdio, for ``mcp dev``)
    - HTTP data-plane endpoint at ``/s2sp/data/{token}``
    """

    def __init__(
        self,
        name: str,
        s2sp_port: int = 0,
        s2sp_host: str = "127.0.0.1",
        cache_ttl: int = 600,
    ) -> None:
        """
        Args:
            name: Server name.
            s2sp_port: Port for the data-plane HTTP server (0 = auto).
            s2sp_host: Host to bind the data-plane server.
            cache_ttl: Seconds before cached data expires (default 600 = 10 min).
        """
        self.name = name
        self._cache_ttl = cache_ttl
        self._direct_channel = DirectChannel(
            port=s2sp_port,
            host=s2sp_host,
        )

        # Data-plane cache: token → {data, tool, timestamp, domains...}
        self._data_cache: dict[str, dict[str, Any]] = {}
        self._direct_channel._data_cache = self._data_cache
        self._direct_channel._cache_ttl = cache_ttl

        # Embedded MCP server
        self._mcp = self._create_mcp()

    # ── Properties ────────────────────────────────────────────────

    @property
    def mcp(self) -> FastMCP:
        """The embedded FastMCP instance.

        Add custom tools, resources, and prompts::

            @server.mcp.tool()
            async def my_tool(query: str) -> str:
                return "result"
        """
        return self._mcp

    @property
    def s2sp_endpoint(self) -> str:
        """This server's HTTP endpoint URL (for data-plane access)."""
        return self._direct_channel.endpoint_url

    # ── Lifecycle ─────────────────────────────────────────────────

    def _create_mcp(self) -> FastMCP:
        server_ref = self

        @asynccontextmanager
        async def lifespan(mcp_server: FastMCP):
            await server_ref.start()
            try:
                yield
            finally:
                await server_ref.stop()

        return FastMCP(f"{self.name} (S2SP)", lifespan=lifespan)

    async def start(self) -> None:
        """Start the S2SP data-plane HTTP server."""
        await self._direct_channel.start()
        logger.info(f"S2SP server '{self.name}' started on {self.s2sp_endpoint}")

    async def stop(self) -> None:
        """Stop the S2SP data-plane HTTP server."""
        await self._direct_channel.stop()
        logger.info(f"S2SP server '{self.name}' stopped")

    def run(self) -> None:
        """Run as a standalone MCP server (stdio transport).

        Use for ``mcp dev`` or direct MCP Inspector connection.
        """
        self._mcp.run()

    # ── @s2sp_resource_tool() / @s2sp_tool() ────────────────────────

    def s2sp_resource_tool(self, name: str | None = None, description: str | None = None):
        """Decorator: register a resource tool with S2SP domain projection.

        The decorated function returns ``list[dict]`` (tabular data).
        The decorator adds two optional parameters:

        - ``abstract_domains``: comma-separated column names for the
          control plane. Only these columns (+ ``_row_id``) are returned.
          Remaining columns are cached as body domains.
        - ``mode``: ``"async"`` (default) caches body, returns a
          ``resource_url`` (presigned URL for data-plane fetch).
          ``"sync"`` returns body inline.

        Without ``abstract_domains``, the tool behaves as standard MCP.

        Example::

            @server.s2sp_resource_tool()
            async def get_alerts(area: str) -> list[dict]:
                return await fetch_from_api(area)

            # Standard call: get_alerts(area="CA") → all columns
            # S2SP call: get_alerts(area="CA", abstract_domains="event,severity")
            #   → only event + severity + _row_id returned
        """
        def decorator(fn):
            tool_name = name or fn.__name__
            tool_desc = description or fn.__doc__ or ""
            server_ref = self

            # Build wrapper with original params + abstract_domains + mode
            orig_sig = inspect.signature(fn)
            orig_params = list(orig_sig.parameters.values())
            new_params = orig_params + [
                inspect.Parameter("abstract_domains", inspect.Parameter.KEYWORD_ONLY,
                                  default="", annotation=str),
                inspect.Parameter("mode", inspect.Parameter.KEYWORD_ONLY,
                                  default="async", annotation=str),
            ]
            new_sig = orig_sig.replace(parameters=new_params)

            @functools.wraps(fn)
            async def wrapper(*args, **kwargs) -> str:
                abstract_str = kwargs.pop("abstract_domains", "")
                mode = kwargs.pop("mode", "async")

                result = await fn(*args, **kwargs)

                if not abstract_str:
                    return json.dumps(result, indent=2, default=str)

                # Parse abstract_domains (comma-separated or JSON array)
                raw = abstract_str.strip()
                if raw.startswith("["):
                    try:
                        fields = [str(f).strip() for f in json.loads(raw)]
                    except json.JSONDecodeError:
                        fields = [f.strip().strip('"\'') for f in raw.strip("[]").split(",")]
                else:
                    fields = [f.strip() for f in raw.split(",") if f.strip()]

                if not isinstance(result, list):
                    result = [result]

                # Discover all columns
                all_cols = list(dict.fromkeys(c for row in result for c in row))
                body_fields = [c for c in all_cols if c not in fields]

                # Add _row_id
                for i, row in enumerate(result):
                    row["_row_id"] = i

                # Project abstract
                abstract = [
                    {"_row_id": row["_row_id"], **{f: row[f] for f in fields if f in row}}
                    for row in result
                ]

                if mode == "sync":
                    # Body returned inline, no caching
                    body = [
                        {"_row_id": row["_row_id"], **{f: row[f] for f in body_fields if f in row}}
                        for row in result
                    ]
                    return json.dumps({
                        "total_rows": len(result),
                        "columns": all_cols,
                        "abstract_domains": fields,
                        "body_domains": body_fields,
                        "abstract": abstract,
                        "body": body,
                    }, indent=2, default=str)
                else:
                    # Cache body, return presigned resource_url
                    token = secrets.token_urlsafe(32)  # 256-bit unguessable
                    server_ref._data_cache[token] = {
                        "data": result,
                        "tool": tool_name,
                        "timestamp": time.time(),
                        "abstract_domains": fields,
                        "body_domains": body_fields,
                        "columns": all_cols,
                    }
                    resource_url = f"{server_ref.s2sp_endpoint}/s2sp/data/{token}"
                    return json.dumps({
                        "total_rows": len(result),
                        "columns": all_cols,
                        "abstract_domains": fields,
                        "body_domains": body_fields,
                        "resource_url": resource_url,
                        "abstract": abstract,
                    }, indent=2, default=str)

            # Set signature for FastMCP introspection
            wrapper.__signature__ = new_sig.replace(return_annotation=str)
            wrapper.__annotations__ = {
                k: v for k, v in fn.__annotations__.items() if k != "return"
            }
            wrapper.__annotations__.update(abstract_domains=str, mode=str, **{"return": str})

            # Register on FastMCP
            enhanced_desc = (
                tool_desc.strip() +
                "\n\nS2SP: set abstract_domains to ONLY the columns you need for "
                "filtering/reasoning. Pick as few as possible — more columns = more "
                "tokens. Remaining columns are cached on the data plane for the "
                "consumer server. Accepts comma-separated or JSON array. "
                "mode='async' (default) or 'sync' (body inline)."
            )
            self._mcp.add_tool(wrapper, name=tool_name, description=enhanced_desc)
            return fn

        return decorator

    # Backward-compatible alias
    s2sp_tool = s2sp_resource_tool

    # ── @s2sp_consumer_tool() ─────────────────────────────────────

    def s2sp_consumer_tool(self, name: str | None = None, description: str | None = None):
        """Decorator: register a consumer tool with automatic S2SP resolution.

        The decorated function receives ``rows: list[dict]`` — the merged
        result of abstract + body data, with column mapping applied.
        The decorator adds four parameters to the MCP tool:

        - ``abstract_data``: JSON array of abstract rows from the agent
        - ``resource_url``: presigned URL for async body fetch (optional)
        - ``body_data``: inline body for sync mode (optional)
        - ``column_mapping``: JSON dict to rename columns (optional)

        The decorator calls ``DirectChannel.resolve()`` to handle parsing,
        fetching, remapping, and merging — then passes the merged rows
        to your function.

        Example::

            @server.s2sp_consumer_tool()
            async def draw_chart(rows: list[dict]) -> str:
                return generate_chart(rows)

            # Agent calls:
            #   draw_chart(abstract_data='[...]', resource_url='http://...')
            # Your function receives: merged rows with all columns
        """
        def decorator(fn):
            tool_name = name or fn.__name__
            tool_desc = description or fn.__doc__ or ""

            # Build a clean wrapper with explicit signature (no @wraps)
            async def wrapper(
                abstract_data: str,
                resource_url: str = "",
                body_data: str = "",
                column_mapping: str = "",
            ):
                rows = await DirectChannel.resolve(
                    abstract_data, resource_url, body_data, column_mapping)
                return await fn(rows)

            wrapper.__name__ = fn.__name__
            wrapper.__doc__ = fn.__doc__

            enhanced_desc = (
                tool_desc.strip() +
                "\n\nS2SP consumer: pass abstract_data (JSON) + resource_url "
                "(async) or body_data (sync). Optional column_mapping renames "
                "columns. The tool receives merged rows automatically."
            )
            self._mcp.add_tool(wrapper, name=tool_name, description=enhanced_desc)
            return fn

        return decorator

