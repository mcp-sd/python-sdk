# mcp-sd — Reference implementation of MCP-SD

This package is **S2SP** (Server-to-Server Protocol), the reference implementation of **MCP-SD** (Selective Disclosure for MCP).

- **MCP-SD** is the *protocol pattern*: an MCP extension that lets an agent select which attributes of a tool result enter the LLM (via an `abstract_domains` parameter), within a single tool call. The remainder is withheld.
- **S2SP** is this *reference implementation*: MCP-SD plus a dedicated **Direct Data Interface (DDI)** — an HTTP data plane over which withheld columns flow directly between MCP servers, or through the agent process out-of-band from the LLM.

## Overview

When AI agents orchestrate multiple MCP servers, data transfers between servers currently must pass through the agent's context window — wasting tokens, adding latency, and saturating context.

MCP-SD separates the **control plane** (agent ↔ server via MCP tool calls, carrying only agent-selected columns) from the **data plane** (the transport by which withheld columns are delivered). The agent sees only the columns it asks for; full data stays on the data plane. S2SP realizes the data plane as a dedicated DDI running over HTTP, either directly server-to-server (async mode) or through the agent process out-of-band (sync mode).

### How It Works

MCP-SD treats tool responses as **tabular data** (like a pandas DataFrame): n rows × w columns. The agent chooses which columns (domains) it needs:

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
│  │ (small abstract) │     │ (withheld full columns)         │    │
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

Optional extras:

```bash
pip install -e ".[dev]"            # tests
pip install -e ".[demos]"          # matplotlib + anthropic + openai for demos
pip install -e ".[claude-agent]"   # agent-side adapter for Claude Agent SDK
pip install -e ".[langgraph]"      # agent-side adapter for LangGraph
pip install -e ".[agents]"         # all agent adapters
```

## Agent-side integrations

Sync mode requires the agent itself to split tool responses so body rows
never reach the LLM. This package ships platform-agnostic primitives
(`DDIBuffer`, `S2SPDispatcher`) plus thin adapters for mainstream agent
frameworks under `mcp_sd.agent`:

```python
from mcp_sd.agent import S2SPDispatcher

# Platform-agnostic: use with any agent loop
dispatcher = S2SPDispatcher()
rewritten = dispatcher.on_tool_result("get_alerts", raw_response)  # hides body
resolved = dispatcher.on_tool_call("draw_chart", args)             # injects body

# Claude Agent SDK
from mcp_sd.agent.adapters.claude_agent_sdk import wrap_tool

# LangGraph
from mcp_sd.agent.adapters.langgraph import make_sd_tool_node
```

Both adapters share a single `S2SPDispatcher` instance across all tools in
a session. In async mode the dispatcher passes through untouched; in sync
mode it stashes body rows in an in-process DDI buffer keyed by opaque
`ddi://...` handles and resolves them when the agent calls a consumer
tool. The LLM sees the same short handle regardless of mode, so both
flows stay transparent to the model.

## Quick Start

S2SP has two types of MCP tools:

| | Resource tool | Consumer tool |
|---|---|---|
| **Decorator** | `@server.sd_resource_tool()` | `@server.sd_consumer_tool()` |
| **You write** | `async def get_data(...) -> list[dict]` | `async def process(rows: list[dict]) -> str` |
| **S2SP adds** | `abstract_domains`, `mode`, `_row_id`, `resource_url` | `abstract_data`, `resource_url`, `body_data`, `column_mapping` |
| **Data plane** | Caches body, serves via `/s2sp/data/` | Fetches + remaps + merges automatically |

### Resource Tool (`@server.sd_resource_tool()`)

Use this for tools that return tabular data. S2SP automatically adds
`abstract_domains` and `mode` parameters so the agent can choose which
columns it sees.

```python
from mcp_sd import S2SPServer

server = S2SPServer("weather-server")

@server.sd_resource_tool()                          # ← S2SP decorator
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
# → Only those columns + _row_id returned on the control plane
# → Full data cached on server behind resource_url
# → Agent filters, passes abstract rows + resource_url to consumer
```

### Consumer Tool (`@server.sd_consumer_tool()`)

The consumer decorator handles all S2SP plumbing — your function just
receives merged rows:

```python
from mcp_sd import S2SPServer

server = S2SPServer("stats-server")

@server.sd_consumer_tool()                 # ← S2SP consumer decorator
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
export ANTHROPIC_API_KEY=<your-anthropic-api-key>
python demos/weather_agent/agent_async.py   # or agent_sync.py
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

See [website](https://mcp-sd.github.io/index.html) for full documentation.
