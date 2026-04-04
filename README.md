# MCP S2SP Protocol

Server-to-Server (S2SP) data transfer protocol for MCP (Model Context Protocol).

## Overview

When AI agents orchestrate multiple MCP servers, data transfers between servers currently must pass through the agent's context window — wasting tokens, adding latency, and saturating context.

The S2SP protocol separates the **control plane** (agent ↔ server via MCP tool calls) from the **data plane** (server ↔ server via HTTP). The agent sees only the columns it needs for decision-making (abstract domains); full data flows directly between servers without entering the LLM context.

### How It Works

S2SP treats tool responses as **tabular data** (like a pandas DataFrame): n rows × w columns. The agent chooses which columns (domains) it needs:

- **Abstract domains** (control plane): Columns the agent/LLM reasons over — event type, severity, status, etc.
- **Body domains** (data plane): Remaining columns — full descriptions, parameters, raw payloads, etc. These never enter the LLM context.

```
┌─────────────────────────────────────────────────────────────────┐
│  MCP Tool returns 30 columns × 100 rows                        │
│                                                                 │
│  Agent requests: abstract_domains="event,severity,status"       │
│                                                                 │
│  ┌──────────────────┐     ┌────────────────────────────────┐    │
│  │ Control Plane     │     │ Data Plane                      │    │
│  │ (→ LLM context)  │     │ (cached on server)              │    │
│  │                  │     │                                  │    │
│  │ _row_id, event,  │     │ _row_id, description,           │    │
│  │ severity, status │     │ instruction, parameters, ...    │    │
│  │ (~600 tokens)    │     │ (~9,000 tokens saved)           │    │
│  └──────────────────┘     └────────────────────────────────┘    │
│                                                                 │
│  Agent filters → picks row IDs → tells another server to fetch  │
│  from the data plane directly (server-to-server, no LLM)        │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

S2SP has two types of MCP tools:

| | Resource tool | Consumer tool |
|---|---|---|
| **Decorator** | `@server.s2sp_resource_tool()` | `@server.s2sp_consumer_tool()` |
| **You write** | `async def get_data(...) -> list[dict]` | `async def process(rows: list[dict]) -> str` |
| **S2SP adds** | `abstract_domains`, `mode`, `_row_id`, `resource_url` | `abstract_data`, `resource_url`, `body_data`, `column_mapping` |
| **Data plane** | Caches body, serves via `/s2sp/data/` | Fetches + remaps + merges automatically |

### Resource Tool (`@server.s2sp_resource_tool()`)

Use this for tools that return tabular data. S2SP automatically adds
`abstract_domains` and `mode` parameters so the agent can choose which
columns it sees.

```python
from mcp_s2sp import S2SPServer

server = S2SPServer("weather-server")

@server.s2sp_resource_tool()                          # ← S2SP decorator
async def get_alerts(area: str) -> list[dict]:
    """Get weather alerts — returns ~30 columns per alert."""
    data = await fetch_from_nws(area)
    return [feature["properties"] for feature in data["features"]]

server.run()  # MCP Inspector compatible: mcp dev weather_server.py
```

### Agent Calls the Tool

```python
# Standard mode — all columns returned (traditional MCP)
get_alerts(area="CA")

# S2SP mode — agent chooses which columns it needs
get_alerts(area="CA", abstract_domains="event,severity,urgency,status")
# → Only those columns + _row_id returned (control plane, ~600 tokens)
# → Full data cached on server with resource_id (data plane, ~9,000 tokens saved)
# → Agent filters, passes abstract rows + resource_id to consumer
```

### Consumer Tool (`@server.s2sp_consumer_tool()`)

The consumer decorator handles all S2SP plumbing — your function just
receives merged rows:

```python
from mcp_s2sp import S2SPServer

server = S2SPServer("stats-server")

@server.s2sp_consumer_tool()                 # ← S2SP consumer decorator
async def draw_chart(rows: list[dict]) -> str:
    """Draw a chart from merged rows (abstract + body)."""
    return generate_chart(rows)

server.run()
```

The decorator automatically adds `abstract_data`, `resource_url`, `body_data`,
and `column_mapping` parameters to the MCP tool. It calls `DirectChannel.resolve()`
internally to parse, fetch, remap, and merge — then passes the result to your function.

### Column Mapping (optional)

When the consumer uses different column names than the resource server:

```python
draw_chart(
    abstract_data=...,
    resource_url=...,
    column_mapping='{"event": "alert_type", "areaDesc": "location"}'
)
# Resource returns: {"event": "Wind Advisory", "areaDesc": "LA"}
# Consumer sees:    {"alert_type": "Wind Advisory", "location": "LA"}
```

## Running Tests

```bash
pytest tests/
```

## Running Demos

### Interactive Agent (recommended)

```bash
pip install -e ".[demos]"
export ANTHROPIC_API_KEY=sk-ant-...
python demos/weather_agent/agent.py
```

Then ask:
- "Show me weather alerts for California"
- "Filter for wind advisories"
- "Generate a chart of the severe alerts"

### Scripted Demos

```bash
# Async mode — body stays on source server, fetched via data plane
python demos/weather_agent/run_async.py [--area CA] [--event Wind]

# Sync mode — body returned inline, no data-plane fetch
python demos/weather_agent/run_sync.py [--area CA] [--event Wind]
```

### Debug with MCP Inspector

```bash
pip install -e ".[inspector]"
mcp dev demos/weather_agent/weather_server.py
mcp dev demos/weather_agent/stats_server.py
```

## Protocol Design

See [website](https://s2sp-protocol.github.io/index.html) for full documentation, or [paper/s2s_protocol.tex](paper/) for the academic paper.
