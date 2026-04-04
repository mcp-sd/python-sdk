"""Weather MCP Server with S2SP.

A standard MCP weather server. The @s2sp_tool decorator adds
abstract_domains support — the agent picks which columns it sees.

    mcp dev demos/weather_agent/weather_server.py
"""

import json, sys
from typing import Any
import httpx

sys.path.insert(0, "src")
from mcp_s2sp import S2SPServer

server = S2SPServer("weather-server")

NWS_API = "https://api.weather.gov"
USER_AGENT = "S2SP-Weather-Demo/1.0"


async def _nws_request(url: str) -> dict | None:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(url, headers=headers, timeout=30.0)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None


@server.s2sp_resource_tool()
async def get_alerts(area: str) -> list[dict]:
    """Get active weather alerts for a US state.

    Returns ~30 fields per alert. Use abstract_domains to pick ONLY the
    columns you need to filter. Fewer columns = fewer tokens = faster.

    Common choices:
      - "event" alone is enough to filter by alert type (Wind, Flood, etc.)
      - "event,severity" adds severity level
      - "event,severity,areaDesc" adds affected area

    All 30 fields: id, status, event, urgency, certainty, severity,
    headline, areaDesc, description, instruction, response, sent,
    effective, onset, expires, ends, sender, senderName, messageType,
    category, scope, parameters, geocode, affectedZones, etc.

    Args:
        area: Two-letter US state code (e.g. CA, TX, FL, NY).
    """
    data = await _nws_request(f"{NWS_API}/alerts/active?area={area}")
    if not data or "features" not in data:
        return []
    return [
        {k: v for k, v in f["properties"].items() if not k.startswith("@")}
        for f in data["features"]
    ]


@server.s2sp_resource_tool()
async def get_forecast(latitude: float, longitude: float) -> list[dict]:
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location.
        longitude: Longitude of the location.
    """
    points = await _nws_request(f"{NWS_API}/points/{latitude},{longitude}")
    if not points:
        return []
    forecast = await _nws_request(points["properties"]["forecast"])
    return forecast["properties"]["periods"] if forecast else []


@server.mcp.resource("weather://cache")
async def cache_status() -> str:
    """Data cached on the server (data plane)."""
    return json.dumps({
        rid: {"rows": len(e["data"]), "tool": e["tool"]}
        for rid, e in server._data_cache.items()
    }, indent=2)


@server.mcp.prompt()
def workflow_guide() -> str:
    """How to use get_alerts with abstract_domains."""
    return (
        "1. Call get_alerts(area, abstract_domains='event,severity') "
        "— pick columns based on what you need to filter.\n"
        "2. Read abstract rows, filter for what the user wants.\n"
        "3. Call draw_chart(abstract_data=<filtered>, resource_url=<from response>) "
        "— Stats Server fetches body directly.\n"
        "Keep abstract_domains minimal. You never see the body."
    )


# ── Entry point ──────────────────────────────────────────────────

mcp = server.mcp  # for: mcp dev weather_server.py

if __name__ == "__main__":
    server.run()
