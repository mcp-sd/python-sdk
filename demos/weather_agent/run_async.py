#!/usr/bin/env python3
"""Scripted demo — Async mode.

Calls get_alerts through MCP with abstract_domains, filters the abstract,
then calls draw_chart which fetches body server-to-server.

    python demos/weather_agent/run_async.py [--area CA] [--event Wind]
"""

import argparse, asyncio, json, sys
sys.path.insert(0, "src")

from weather_server import server as weather
from stats_server import server as stats


async def call_tool(server, name, args):
    result = await server.mcp.call_tool(name, args)
    content = result[0] if isinstance(result, tuple) else result
    for block in content:
        if hasattr(block, "text"):
            return block.text
    return ""


async def run(area="CA", event_keyword="Wind"):
    await weather.start()
    await stats.start()

    print(f"Weather: {weather.s2sp_endpoint}  Stats: {stats.s2sp_endpoint}\n")

    # Step 1: Get abstract
    print(f"[1] get_alerts(area={area}, abstract_domains='event,severity,urgency,status,headline,areaDesc')")
    resp = json.loads(await call_tool(weather, "get_alerts", {
        "area": area,
        "abstract_domains": "event,severity,urgency,status,headline,areaDesc",
    }))
    print(f"    {resp['total_rows']} alerts, {len(resp['abstract_domains'])} abstract columns")
    print(f"    resource_url={resp['resource_url']}")

    # Step 2: Filter
    print(f"\n[2] Filter: event contains '{event_keyword}'")
    selected = [r for r in resp["abstract"]
                if event_keyword.lower() in (r.get("event") or "").lower()]
    print(f"    {len(selected)} of {resp['total_rows']} matched")
    for r in selected[:3]:
        print(f"    _row_id={r['_row_id']}: [{r.get('severity')}] {r.get('event')}")

    if not selected:
        events = sorted(set(r.get("event", "?") for r in resp["abstract"]))
        print(f"    Available: {', '.join(events)}")
        return

    # Step 3: Draw chart
    print(f"\n[3] draw_chart({len(selected)} rows)")
    chart_resp = json.loads(await call_tool(stats, "draw_chart", {
        "abstract_data": json.dumps(selected, default=str),
        "resource_url": resp["resource_url"],
    }))
    print(f"    analyzed={chart_resp.get('alerts_analyzed', '?')}")
    print(f"    events: {chart_resp.get('event_breakdown')}")

    # Token savings
    abstract_tokens = len(json.dumps(resp, default=str)) // 4
    print(f"\n    Tokens: agent saw ~{abstract_tokens:,} (body fetched server-to-server)")

    await weather.stop()
    await stats.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--area", default="CA")
    p.add_argument("--event", default="Wind")
    a = p.parse_args()
    asyncio.run(run(a.area, a.event))
