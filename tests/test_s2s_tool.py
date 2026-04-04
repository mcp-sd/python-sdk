"""Tests for S2SP protocol features: s2sp_tool, data caching, data-plane endpoints, and DirectChannel."""

import json

import httpx
import pytest

from mcp_s2sp.direct_channel import DirectChannel
from mcp_s2sp.server import S2SPServer

SAMPLE_DATA = [
    {"name": "Alice", "age": 30, "email": "alice@example.com", "salary": 100000},
    {"name": "Bob", "age": 25, "email": "bob@example.com", "salary": 80000},
    {"name": "Charlie", "age": 35, "email": "charlie@example.com", "salary": 120000},
]


def _make_server_with_tool(name: str = "test-server", port: int = 0) -> S2SPServer:
    """Create an S2SPServer with a sample s2sp_tool registered."""
    server = S2SPServer(name, s2sp_port=port)

    @server.s2sp_tool()
    async def get_people(area: str) -> list[dict]:
        """Return sample people data."""
        # Return a deep copy so mutations (_row_id injection) don't leak between tests
        return [dict(row) for row in SAMPLE_DATA]

    return server


@pytest.fixture
async def server():
    srv = _make_server_with_tool()
    await srv.start()
    yield srv
    await srv.stop()


@pytest.fixture
async def source_server():
    srv = _make_server_with_tool("source-server")
    await srv.start()
    yield srv
    await srv.stop()


@pytest.fixture
async def consumer_server():
    srv = S2SPServer("consumer-server", s2sp_port=0)
    await srv.start()
    yield srv
    await srv.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _call_tool(server: S2SPServer, tool_name: str, arguments: dict) -> str:
    """Invoke an MCP tool on the server's embedded FastMCP and return the text."""
    result = await server.mcp.call_tool(tool_name, arguments)
    # call_tool returns (content_list, meta_dict); extract text from first content block
    content_list = result[0] if isinstance(result, tuple) else result
    return content_list[0].text


# ===========================================================================
# 1. Without abstract_domains
# ===========================================================================

class TestNoAbstractDomains:
    async def test_returns_all_data_as_json(self, server):
        """Without abstract_domains the tool returns all rows as plain JSON."""
        text = await _call_tool(server, "get_people", {"area": "US"})
        data = json.loads(text)

        assert isinstance(data, list)
        assert len(data) == 3
        # All original columns present, no _row_id injected
        assert set(data[0].keys()) == {"name", "age", "email", "salary"}
        assert data[0]["name"] == "Alice"

    async def test_no_cache_entry_created(self, server):
        """Without abstract_domains nothing is cached."""
        await _call_tool(server, "get_people", {"area": "US"})
        assert len(server._data_cache) == 0


# ===========================================================================
# 2. Async mode (abstract_domains set)
# ===========================================================================

class TestAsyncMode:
    async def test_rows_have_row_id(self, server):
        text = await _call_tool(
            server, "get_people",
            {"area": "US", "abstract_domains": "name,age"},
        )
        resp = json.loads(text)
        for row in resp["abstract"]:
            assert "_row_id" in row

    async def test_only_abstract_columns_and_row_id(self, server):
        text = await _call_tool(
            server, "get_people",
            {"area": "US", "abstract_domains": "name,age"},
        )
        resp = json.loads(text)
        for row in resp["abstract"]:
            assert set(row.keys()) == {"_row_id", "name", "age"}

    async def test_no_body_in_async(self, server):
        """Async mode does not include body — only resource_url for later fetch."""
        text = await _call_tool(
            server, "get_people",
            {"area": "US", "abstract_domains": "name,age"},
        )
        resp = json.loads(text)
        assert "body" not in resp
        assert "resource_url" in resp

    async def test_resource_url_is_presigned(self, server):
        text = await _call_tool(
            server, "get_people",
            {"area": "US", "abstract_domains": "name,age"},
        )
        resp = json.loads(text)
        url = resp["resource_url"]
        assert url.startswith("http")
        assert "/s2sp/data/" in url  # presigned URL contains path + token
        assert "resource_id" not in resp  # no separate resource_id

    async def test_domains_computed_correctly(self, server):
        text = await _call_tool(
            server, "get_people",
            {"area": "US", "abstract_domains": "name,age"},
        )
        resp = json.loads(text)
        assert resp["abstract_domains"] == ["name", "age"]
        assert set(resp["body_domains"]) == {"email", "salary"}

    async def test_data_cached(self, server):
        text = await _call_tool(
            server, "get_people",
            {"area": "US", "abstract_domains": "name"},
        )
        resp = json.loads(text)
        # Extract token from presigned URL
        token = resp["resource_url"].split("/s2sp/data/")[-1]

        assert token in server._data_cache
        entry = server._data_cache[token]
        assert len(entry["data"]) == 3
        assert entry["tool"] == "get_people"
        assert "_row_id" in entry["data"][0]


# ===========================================================================
# 3. Sync mode
# ===========================================================================

class TestSyncMode:
    async def test_body_not_null(self, server):
        text = await _call_tool(
            server, "get_people",
            {"area": "US", "abstract_domains": "name,age", "mode": "sync"},
        )
        resp = json.loads(text)
        assert resp["body"] is not None
        assert isinstance(resp["body"], list)
        assert len(resp["body"]) == 3

    async def test_body_contains_body_columns_and_row_id(self, server):
        text = await _call_tool(
            server, "get_people",
            {"area": "US", "abstract_domains": "name,age", "mode": "sync"},
        )
        resp = json.loads(text)
        for body_row in resp["body"]:
            assert "_row_id" in body_row
            # Body columns = email, salary (everything NOT in abstract_domains)
            assert "email" in body_row
            assert "salary" in body_row
            # Abstract columns should NOT be in body rows
            assert "name" not in body_row
            assert "age" not in body_row

    async def test_abstract_rows_only_abstract_columns(self, server):
        text = await _call_tool(
            server, "get_people",
            {"area": "US", "abstract_domains": "name,age", "mode": "sync"},
        )
        resp = json.loads(text)
        for row in resp["abstract"]:
            assert set(row.keys()) == {"_row_id", "name", "age"}

    async def test_no_resource_id_in_sync(self, server):
        """Sync mode returns body inline — no caching, no resource_id."""
        text = await _call_tool(
            server, "get_people",
            {"area": "US", "abstract_domains": "name", "mode": "sync"},
        )
        resp = json.loads(text)
        assert "resource_id" not in resp
        assert "resource_url" not in resp


# ===========================================================================
# 4. Data-plane endpoint (presigned resource_url)
# ===========================================================================

class TestDataPlaneEndpoint:
    async def _get_resource_url(self, server):
        text = await _call_tool(
            server, "get_people",
            {"area": "US", "abstract_domains": "name"},
        )
        return json.loads(text)["resource_url"]

    async def test_fetch_all_rows(self, server):
        url = await self._get_resource_url(server)
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json={})
        assert r.status_code == 200
        body = r.json()
        assert body["total_rows"] == 3
        assert len(body["body"]) == 3

    async def test_filter_by_row_ids(self, server):
        url = await self._get_resource_url(server)
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json={"row_ids": [0, 2]})
        body = r.json()
        assert body["total_rows"] == 2
        ids = {row["_row_id"] for row in body["body"]}
        assert ids == {0, 2}

    async def test_filter_by_columns(self, server):
        url = await self._get_resource_url(server)
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json={"columns": ["email"]})
        body = r.json()
        for row in body["body"]:
            assert "_row_id" in row
            assert "email" in row
            assert "salary" not in row

    async def test_row_id_always_included(self, server):
        url = await self._get_resource_url(server)
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json={"columns": ["salary"]})
        body = r.json()
        for row in body["body"]:
            assert "_row_id" in row

    async def test_unknown_resource_returns_404(self, server):
        url = f"{server.s2sp_endpoint}/s2sp/data/bad-token"
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json={})
        assert r.status_code == 404


# ===========================================================================
# 5. DirectChannel.fetch_data (cross-server)
# ===========================================================================

class TestDirectChannelFetchData:
    async def test_fetch_from_source(self, source_server, consumer_server):
        """Consumer fetches data from source via presigned resource_url."""
        text = await _call_tool(
            source_server, "get_people",
            {"area": "US", "abstract_domains": "name"},
        )
        resp = json.loads(text)

        rows = await DirectChannel.fetch_data(
            resource_url=resp["resource_url"],
            row_ids=[1],
            columns=["name", "salary"],
        )
        assert len(rows) == 1
        assert rows[0]["name"] == "Bob"
        assert rows[0]["salary"] == 80000
        assert "_row_id" in rows[0]

    async def test_fetch_all_from_source(self, source_server, consumer_server):
        text = await _call_tool(
            source_server, "get_people",
            {"area": "US", "abstract_domains": "name"},
        )
        resp = json.loads(text)
        rows = await DirectChannel.fetch_data(resp["resource_url"])
        assert len(rows) == 3


# ===========================================================================
# 7. MCP Inspector compatibility (list_tools + call via MCP client)
# ===========================================================================

class TestMCPInspectorCompat:
    async def test_list_tools(self, server):
        """Only user-defined s2sp_tools should be listed (no s2sp_fetch clutter)."""
        tools = await server.mcp.list_tools()
        tool_names = {t.name for t in tools}
        assert "get_people" in tool_names
        assert "s2sp_fetch" not in tool_names  # removed — agent never fetches body

    async def test_call_tool_without_abstract(self, server):
        text = await _call_tool(server, "get_people", {"area": "US"})
        data = json.loads(text)
        assert isinstance(data, list)
        assert len(data) == 3

    async def test_call_tool_with_abstract(self, server):
        text = await _call_tool(
            server, "get_people",
            {"area": "US", "abstract_domains": "name,age"},
        )
        resp = json.loads(text)
        assert "abstract" in resp
        assert "resource_url" in resp
        assert "resource_id" not in resp  # merged into resource_url
        assert "body" not in resp  # async mode — no body inline
        for row in resp["abstract"]:
            assert set(row.keys()) == {"_row_id", "name", "age"}

    async def test_response_parses_as_valid_json(self, server):
        """All responses should be valid JSON strings."""
        t1 = await _call_tool(server, "get_people", {"area": "US"})
        json.loads(t1)  # should not raise

        t2 = await _call_tool(
            server, "get_people",
            {"area": "US", "abstract_domains": "name"},
        )
        json.loads(t2)  # should not raise
