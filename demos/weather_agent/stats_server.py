"""Statistics MCP Server with S2SP.

Receives abstract rows from the agent, fetches body data directly
from the Weather Server (async mode) or inline (sync mode), merges
by _row_id, and generates an 8-panel chart.

    mcp dev demos/weather_agent/stats_server.py
"""

import io, json, os, sys
from collections import Counter
from typing import Any

sys.path.insert(0, "src")
from mcp_s2sp import S2SPServer, DirectChannel

server = S2SPServer("stats-server")
_results: dict[str, Any] = {}


# ── Chart generation ─────────────────────────────────────────────

def _generate_chart(rows: list[dict]) -> bytes:
    """Generate an 8-panel bar chart from alert data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        (Counter(r.get("event", "?") for r in rows), "Event Type"),
        (Counter(r.get("severity", "?") for r in rows), "Severity"),
        (Counter(r.get("urgency", "?") for r in rows), "Urgency"),
        (Counter(r.get("certainty", "?") for r in rows), "Certainty"),
        (dict(Counter(r.get("senderName", "?") for r in rows).most_common(8)), "NWS Office"),
        (Counter(r.get("messageType", "?") for r in rows), "Message Type"),
        (Counter(r.get("category", "?") for r in rows), "Category"),
        (dict(Counter(
            z.strip() for r in rows
            for z in (r.get("areaDesc") or "").split(";") if z.strip()
        ).most_common(8)), "Affected Area"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(28, 12))
    fig.suptitle(f"Weather Alert Statistics ({len(rows)} alerts)",
                 fontsize=18, fontweight="bold", y=0.98)
    cmaps = [plt.cm.Set2, plt.cm.Set1, plt.cm.Pastel1, plt.cm.tab10,
             plt.cm.Paired, plt.cm.Accent, plt.cm.Dark2, plt.cm.Set3]

    for (ax, (counts, title), cmap) in zip(axes.flat, panels, cmaps):
        labels, values = list(counts.keys()), list(counts.values())
        if not values:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color="gray", transform=ax.transAxes)
            ax.set_title(title, fontsize=11, fontweight="bold")
            continue
        short = [(l[:20] + "…") if len(l) > 20 else l for l in labels]
        colors = cmap([i / max(len(labels), 1) for i in range(len(labels))])
        bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="gray")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=11, fontweight="bold")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    str(v), ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Consumer tool ─────────────────────────────────────────────────
# The @s2sp_consumer_tool decorator handles:
#   - parsing abstract_data, resource_url, body_data, column_mapping
#   - fetching body from resource server (async) or parsing inline (sync)
#   - remapping columns if column_mapping provided
#   - merging abstract + body by _row_id
# Our function just receives the merged rows.

@server.s2sp_consumer_tool()
async def draw_chart(rows: list[dict]) -> str:
    """Generate an 8-panel statistics chart from weather alerts.

    Receives merged rows (abstract + body joined by _row_id).
    The S2SP decorator handles fetching and column remapping automatically.
    """
    if not rows:
        return json.dumps({"error": "No rows to chart"})

    chart_png = _generate_chart(rows)
    chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weather_alerts_chart.png")
    with open(chart_path, "wb") as f:
        f.write(chart_png)

    events = dict(Counter(r.get("event", "?") for r in rows))
    _results.update(last_chart=chart_png, last_count=len(rows), last_events=events)

    from mcp.server.fastmcp.utilities.types import Image
    summary = json.dumps({
        "chart_generated": True,
        "alerts_analyzed": len(rows),
        "chart_size_bytes": len(chart_png),
        "chart_saved_to": chart_path,
        "event_breakdown": events,
    }, indent=2, default=str)
    return [summary, Image(data=chart_png, format="png")]


@server.mcp.resource("stats://results")
async def results() -> str:
    """Last chart results."""
    return json.dumps({
        "count": _results.get("last_count", 0),
        "events": _results.get("last_events", {}),
    }, indent=2)


# ── Entry point ──────────────────────────────────────────────────

mcp = server.mcp

if __name__ == "__main__":
    server.run()
