#!/usr/bin/env python3
"""Scripted demo — Sync mode.

Calls get_alerts with mode=sync to get abstract + body inline.
Filters abstract, matches body by _row_id, passes both to draw_chart.

    python demos/weather_agent/run_sync.py [--area CA] [--event Wind]
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

    # Step 1: Get abstract + body inline
    print(f"[1] get_alerts(area={area}, mode=sync)")
    resp = json.loads(await call_tool(weather, "get_alerts", {
        "area": area,
        "abstract_domains": "event,severity,urgency,status,headline,areaDesc",
        "mode": "sync",
    }))
    print(f"    {resp['total_rows']} alerts")
    print(f"    abstract: {len(resp['abstract'])} rows, body: {len(resp['body'])} rows")

    # Step 2: Filter abstract
    print(f"\n[2] Filter: event contains '{event_keyword}'")
    selected = [r for r in resp["abstract"]
                if event_keyword.lower() in (r.get("event") or "").lower()]
    print(f"    {len(selected)} of {resp['total_rows']} matched")

    if not selected:
        events = sorted(set(r.get("event", "?") for r in resp["abstract"]))
        print(f"    Available: {', '.join(events)}")
        return

    # Step 3: Match body by _row_id
    selected_ids = {r["_row_id"] for r in selected}
    matched_body = [b for b in resp["body"] if b.get("_row_id") in selected_ids]
    print(f"\n[3] Matched {len(matched_body)} body rows by _row_id")

    # Step 4: Draw chart (no server-to-server fetch)
    print(f"\n[4] draw_chart({len(selected)} abstract + {len(matched_body)} body)")
    chart_resp = json.loads(await call_tool(stats, "draw_chart", {
        "abstract_data": json.dumps(selected, default=str),
        "body_data": json.dumps(matched_body, default=str),
    }))
    print(f"    analyzed={chart_resp.get('alerts_analyzed', '?')}")
    print(f"    events: {chart_resp.get('event_breakdown')}")
    print(f"    No server-to-server fetch needed (sync mode)")

    await weather.stop()
    await stats.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--area", default="CA")
    p.add_argument("--event", default="Wind")
    a = p.parse_args()
    asyncio.run(run(a.area, a.event))
