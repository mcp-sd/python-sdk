#!/usr/bin/env python3
"""Weather Agent — Sync Mode on the Claude Agent SDK.

What this demo proves
---------------------
The LLM sees only ``abstract`` + a ``ddi://...`` handle. The ``body`` rows
transit the ``DDIBuffer`` in the agent process but never reach the LLM
context — the pattern MCP-SD's Direct Data Interface (DDI) was designed
for. The resource tool still returns sync-mode JSON (``abstract`` + ``body``
inline); the agent-side ``S2SPDispatcher`` rewrites the payload before it
is handed back to the model, and later re-inlines the body when the LLM
calls the consumer tool with the handle.

Run::

    export ANTHROPIC_API_KEY=<your-anthropic-api-key>
    python demos/weather_agent/agent_sync_claude.py
"""

from __future__ import annotations

import asyncio, json, os, sys
from functools import wraps

sys.path.insert(0, "src")
os.environ.setdefault("NO_PROXY", "api.anthropic.com,api.weather.gov")

from weather_server import server as weather_server  # noqa: E402
from stats_server import server as stats_server  # noqa: E402

try:
    from claude_agent_sdk import (  # noqa: E402
        ClaudeAgentOptions, ClaudeSDKClient, create_sdk_mcp_server, tool,
    )
except ImportError:
    print("claude-agent-sdk not installed: pip install 'claude-agent-sdk>=0.1.0'")
    raise SystemExit(0)

from mcp_sd.agent import S2SPDispatcher  # noqa: E402
from mcp_sd.agent.adapters.claude_agent_sdk import wrap_tool  # noqa: E402


MODEL = "claude-sonnet-4-6"  # matches agent_async.py


async def call_mcp_tool(server, name, args):
    """Invoke an in-process S2SPServer tool; return the first text block."""
    result = await server.mcp.call_tool(name, args)
    content = result[0] if isinstance(result, tuple) else result
    for block in content:
        if hasattr(block, "text"):
            return block.text
    return str(content)


def as_sdk_content(func):
    """Adapt a ``str``-returning handler to SDK ``{"content": [...]}`` format.

    The dispatcher rewrites the string form; ``@tool`` expects a dict. This
    shim sits between them so we can keep the canonical @tool/@wrap_tool order.
    """
    @wraps(func)
    async def inner(args):
        text = await func(args)
        return {"content": [{"type": "text", "text": text}]}
    return inner


# ── Shared dispatcher + tool definitions ──────────────────────────
# A single dispatcher lives for the life of the agent. Its DDIBuffer is
# where withheld body rows temporarily rest — in-process only, never in
# the LLM transcript.

dispatcher = S2SPDispatcher()


@tool(
    "get_alerts",
    "Fetch weather alerts for a US state. Returns abstract rows + ddi:// handle.",
    {"area": str, "abstract_domains": str},
)
@as_sdk_content
@wrap_tool(dispatcher)
async def get_alerts(args):
    # Force sync mode: the resource server inlines body alongside abstract.
    # dispatcher.on_tool_result (via @wrap_tool) then intercepts the JSON
    # string, moves `body` into the DDIBuffer, and returns a payload whose
    # only reference to the body is a ddi://... handle under resource_url.
    sync_args = {**args, "mode": "sync"}
    return await call_mcp_tool(weather_server, "get_alerts", sync_args)


@tool(
    "draw_chart",
    "Generate an 8-panel weather statistics chart. Pass abstract_data + resource_url from get_alerts.",
    {"abstract_data": str, "resource_url": str},
)
@as_sdk_content
@wrap_tool(dispatcher)
async def draw_chart(args):
    # dispatcher.on_tool_call (via @wrap_tool) sees the ddi:// handle in
    # args["resource_url"], pops the stashed body from the DDIBuffer, and
    # swaps it in as args["body_data"] before the consumer tool runs.
    return await call_mcp_tool(stats_server, "draw_chart", args)


SYSTEM = (
    "You are a weather agent. Two tools are available:\n"
    "1) get_alerts(area, abstract_domains) — returns abstract rows + a resource_url handle.\n"
    "2) draw_chart(abstract_data, resource_url) — forward the filtered rows + the handle.\n"
    "Filter abstract yourself. Keep abstract_domains minimal. You never see body rows; "
    "pass resource_url through untouched."
)


# ── Main REPL ─────────────────────────────────────────────────────

async def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run this demo.")
        return

    await weather_server.start()
    await stats_server.start()

    s2sp_server = create_sdk_mcp_server(
        "s2sp-weather", version="1.0.0", tools=[get_alerts, draw_chart]
    )
    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt=SYSTEM,
        mcp_servers={"s2sp": s2sp_server},
        allowed_tools=["mcp__s2sp__get_alerts", "mcp__s2sp__draw_chart"],
        permission_mode="bypassPermissions",
    )

    print(f"S2SP Weather Agent (sync / claude-agent-sdk) — {MODEL}")
    print(f"Weather: {weather_server.s2sp_endpoint}  Stats: {stats_server.s2sp_endpoint}")
    print('Try: "Get wind alert statistics for California"\n')

    try:
        async with ClaudeSDKClient(options=options) as client:
            while True:
                try:
                    msg = input("you> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not msg or msg.lower() in ("quit", "exit", "q"):
                    break
                await client.query(msg)
                parts = []
                async for event in client.receive_response():
                    for b in getattr(event, "content", None) or []:
                        if getattr(b, "text", None):
                            parts.append(b.text)
                        if getattr(b, "name", None):
                            snippet = json.dumps(getattr(b, "input", {}), default=str)[:100]
                            print(f"  [tool] {b.name}({snippet})")
                if parts:
                    print(f"\nagent> {''.join(parts)}\n")
    finally:
        dispatcher.reset()  # drop unreleased DDIBuffer rows
        await weather_server.stop()
        await stats_server.stop()


if __name__ == "__main__":
    asyncio.run(main())
