"""HTTP data-plane channel for S2SP.

Runs a lightweight HTTP server (Starlette + uvicorn) that serves cached
data to other servers via ``POST /s2sp/data/{token}``.

Also provides ``DirectChannel.fetch_data(resource_url)`` — the static
method that consumer tools use to fetch body data from resource servers.
The ``resource_url`` is a presigned URL containing the token.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

logger = logging.getLogger(__name__)


class DirectChannel:
    """HTTP data-plane channel.

    Each S2SPServer runs a DirectChannel that:
    - Serves ``POST /s2sp/data/{resource_id}`` for data-plane fetch
    - Provides ``fetch_data()`` static method for client-side calls
    """

    def __init__(self, port: int = 0, host: str = "127.0.0.1") -> None:
        self._port = port
        self._host = host
        self._server: Optional[uvicorn.Server] = None
        self._server_task: Optional[asyncio.Task] = None
        self._actual_port: Optional[int] = None

        # Set by S2SPServer — points to the shared data cache
        self._data_cache: Optional[dict] = None
        # Set by S2SPServer — cache TTL in seconds
        self._cache_ttl: int = 600

        self._app = Starlette(routes=[
            Route("/s2sp/data/{resource_id}", self._handle_data_fetch, methods=["POST"]),
            Route("/s2sp/health", self._handle_health, methods=["GET"]),
        ])

    @property
    def port(self) -> int:
        return self._actual_port or self._port

    @property
    def endpoint_url(self) -> str:
        return f"http://{self._host}:{self.port}"

    # ── Lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        config = uvicorn.Config(self._app, host=self._host, port=self._port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())

        for _ in range(50):
            if self._server.started:
                break
            await asyncio.sleep(0.1)

        if self._server and self._server.started and self._server.servers:
            for srv in self._server.servers:
                for sock in srv.sockets:
                    addr = sock.getsockname()
                    if isinstance(addr, tuple):
                        self._actual_port = addr[1]
                        break
            logger.info(f"S2SP data plane listening on {self.endpoint_url}")

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._server_task:
            try:
                await asyncio.wait_for(self._server_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._server_task.cancel()
            self._server_task = None

    # ── Data-plane endpoint ───────────────────────────────────────

    async def _handle_data_fetch(self, request: Request) -> Response:
        """Serve cached data to another server.

        POST /s2sp/data/{resource_id}
        Body: {"row_ids": [0, 1, 5], "columns": ["description"]}

        row_ids: select rows by _row_id (omit for all rows)
        columns: project to specific columns (omit for all columns)
        _row_id is always included in the response.

        After serving, the cache entry is deleted (single-use).
        Expired entries (older than cache_ttl) are also rejected.
        """
        token = request.path_params["resource_id"]

        if self._data_cache is None:
            return JSONResponse({"error": "No data cache"}, status_code=500)

        # Clean expired entries
        import time
        now = time.time()
        expired = [k for k, v in self._data_cache.items()
                   if now - v.get("timestamp", 0) > self._cache_ttl]
        for k in expired:
            del self._data_cache[k]

        entry = self._data_cache.get(token)
        if entry is None:
            return JSONResponse({"error": f"Unknown or expired resource"}, status_code=404)

        try:
            body = await request.json()
        except Exception:
            body = {}

        data = entry["data"]
        req_row_ids = body.get("row_ids", [])
        req_columns = body.get("columns", [])

        # Filter by _row_id
        if req_row_ids:
            id_set = set(int(r) for r in req_row_ids)
            rows = [row for row in data if row.get("_row_id") in id_set]
        else:
            rows = list(data)

        # Column projection
        if req_columns:
            cols = set(req_columns) | {"_row_id"}
            rows = [{k: v for k, v in row.items() if k in cols} for row in rows]

        # Delete after serving (single-use)
        del self._data_cache[token]
        logger.info(f"Served and deleted cache entry {token}")

        return JSONResponse({
            "body": rows,
            "total_rows": len(rows),
        })

    async def _handle_health(self, request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    # ── Client-side: fetch from another server ────────────────────

    @staticmethod
    async def fetch_data(
        resource_url: str,
        row_ids: list[int] | None = None,
        columns: list[str] | None = None,
        column_mapping: dict[str, str] | None = None,
    ) -> list[dict]:
        """Fetch data from a resource server's data-plane cache.

        This is the call a consumer tool makes to the resource server
        to get body data, bypassing the agent.

        Args:
            resource_url: The full presigned URL from the s2sp_tool
                response (e.g. ``http://host:port/s2sp/data/TOKEN``).
            row_ids: Optional _row_id values to select.
            columns: Optional column names to project to (uses resource
                server column names, before mapping).
            column_mapping: Optional dict mapping resource server column
                names to consumer column names. E.g.
                ``{"event": "alert_type", "areaDesc": "location"}``.
                Applied after fetching. Unmapped columns keep their
                original names.

        Returns:
            List of row dicts (with columns renamed if mapping provided).
        """
        payload: dict[str, Any] = {}
        if row_ids:
            payload["row_ids"] = row_ids
        if columns:
            payload["columns"] = columns

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.post(resource_url, json=payload)
            response.raise_for_status()
            rows = response.json().get("body", [])

        if column_mapping:
            rows = [
                {column_mapping.get(k, k): v for k, v in row.items()}
                for row in rows
            ]

        return rows

    @staticmethod
    def remap_columns(
        rows: list[dict],
        column_mapping: dict[str, str],
    ) -> list[dict]:
        """Rename columns in a list of row dicts.

        Args:
            rows: List of row dicts.
            column_mapping: Dict mapping old names to new names.
                Unmapped columns keep their original names.
        """
        return [
            {column_mapping.get(k, k): v for k, v in row.items()}
            for row in rows
        ]

    @staticmethod
    async def resolve(
        abstract_data: str,
        resource_url: str = "",
        body_data: str = "",
        column_mapping: str = "",
    ) -> list[dict]:
        """Resolve abstract + body into merged rows. One-call consumer helper.

        Handles the full consumer workflow:
        1. Parse abstract_data JSON
        2. Parse column_mapping (if provided)
        3. Fetch body from resource_url (async) or parse body_data (sync)
        4. Remap columns on both abstract and body
        5. Merge abstract + body by _row_id

        Use this in consumer tools instead of manually calling fetch_data,
        remap_columns, and merging::

            @server.mcp.tool()
            async def draw_chart(abstract_data: str, resource_url: str = "",
                                 body_data: str = "", column_mapping: str = ""):
                rows = await DirectChannel.resolve(
                    abstract_data, resource_url, body_data, column_mapping)
                return generate_chart(rows)

        Args:
            abstract_data: JSON array of abstract rows (from the agent).
            resource_url: Presigned URL for async fetch (empty for sync).
            body_data: JSON array of body rows for sync mode (empty for async).
            column_mapping: Optional JSON dict to rename columns.
                E.g. '{"event": "alert_type", "areaDesc": "location"}'.

        Returns:
            Merged list of row dicts (abstract + body joined by _row_id,
            columns renamed if mapping provided).
        """
        import json

        # Parse inputs
        abstract_rows = json.loads(abstract_data)
        if not abstract_rows:
            return []

        mapping = json.loads(column_mapping) if column_mapping.strip() else None
        row_ids = [r["_row_id"] for r in abstract_rows if "_row_id" in r]

        # Get body rows
        if resource_url.strip():
            body_rows = await DirectChannel.fetch_data(
                resource_url, row_ids, column_mapping=mapping)
        elif body_data.strip():
            body_rows = json.loads(body_data)
            if mapping:
                body_rows = DirectChannel.remap_columns(body_rows, mapping)
        else:
            body_rows = []

        # Remap abstract columns
        if mapping:
            abstract_rows = DirectChannel.remap_columns(abstract_rows, mapping)

        # Merge by _row_id
        body_map = {r["_row_id"]: r for r in body_rows if "_row_id" in r}
        merged = []
        for row in abstract_rows:
            rid = row.get("_row_id")
            if rid is None:
                continue
            m = dict(row)
            for k, v in body_map.get(rid, {}).items():
                if k not in m:
                    m[k] = v
            merged.append(m)

        return merged
