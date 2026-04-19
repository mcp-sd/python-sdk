#!/usr/bin/env python3
"""Weather Agent — Sync Mode

The agent calls get_alerts with mode="sync" to get abstract + body inline.
It filters abstract rows, matches body by _row_id, passes both to draw_chart.
No server-to-server fetch needed.

Usage:
    export OPENAI_API_KEY=sk-...    # or ANTHROPIC_API_KEY
    python demos/weather_agent/agent_sync.py
"""

import asyncio, json, os, sys
sys.path.insert(0, "src")
os.environ.setdefault("NO_PROXY", "api.anthropic.com,api.openai.com,api.weather.gov")

from weather_server import server as weather_server
from stats_server import server as stats_server


async def call_tool(server, name, args):
    """Call an MCP tool and return the text response."""
    result = await server.mcp.call_tool(name, args)
    content = result[0] if isinstance(result, tuple) else result
    for block in content:
        if hasattr(block, "text"):
            return block.text
    return str(content)


TOOLS = [
    {
        "name": "get_alerts",
        "description": (
            "Fetch weather alerts with mode=sync. Response includes both "
            "abstract rows and body rows inline. Filter abstract yourself, "
            "match body by _row_id, pass both to draw_chart."
        ),
        "input_schema": {
            "type": "object", "required": ["area"],
            "properties": {
                "area": {"type": "string", "description": "US state (CA, TX, NY)"},
                "abstract_domains": {
                    "type": "string",
                    "default": "event,severity,urgency,status,headline,areaDesc",
                    "description": "Columns you need. Keep minimal.",
                },
            },
        },
    },
    {
        "name": "draw_chart",
        "description": (
            "Generate statistics chart. Pass selected abstract rows + "
            "matching body rows as JSON. No server-to-server fetch."
        ),
        "input_schema": {
            "type": "object", "required": ["abstract_data"],
            "properties": {
                "abstract_data": {"type": "string", "description": "JSON array of selected abstract rows"},
                "body_data": {"type": "string", "default": "", "description": "JSON array of matching body rows"},
            },
        },
    },
]

SYSTEM = """\
You are a weather agent (sync mode). Two tools:
1. get_alerts — returns abstract + body inline. Filter abstract yourself.
2. draw_chart — pass filtered abstract + matching body (matched by _row_id).

Filter rows yourself. Match body by _row_id. Keep abstract_domains minimal.\
"""


async def run_tool(name, args):
    if name == "get_alerts":
        # Force sync mode
        args["mode"] = "sync"
        return await call_tool(weather_server, "get_alerts", args)
    elif name == "draw_chart":
        return await call_tool(stats_server, "draw_chart", args)
    return json.dumps({"error": f"Unknown: {name}"})


async def chat_openai(client, model, messages, user_msg):
    messages.append({"role": "user", "content": user_msg})
    oai_tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in TOOLS]
    oai_msgs = [{"role": "system", "content": SYSTEM}] + [m for m in messages if isinstance(m.get("content"), str)]

    while True:
        resp = client.chat.completions.create(model=model, messages=oai_msgs, tools=oai_tools, tool_choice="auto")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            return msg.content or ""
        oai_msgs.append(msg.model_dump())
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"  [tool] {tc.function.name}({json.dumps(args, default=str)[:100]})")
            output = await run_tool(tc.function.name, args)
            oai_msgs.append({"role": "tool", "tool_call_id": tc.id, "content": output})


async def chat_anthropic(client, model, messages, user_msg):
    messages.append({"role": "user", "content": user_msg})
    while True:
        resp = client.messages.create(model=model, max_tokens=4096, system=SYSTEM, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return "\n".join(b.text for b in resp.content if b.type == "text")
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                print(f"  [tool] {b.name}({json.dumps(b.input, default=str)[:100]})")
                output = await run_tool(b.name, b.input)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": output})
        messages.append({"role": "user", "content": results})


async def main():
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client, model, provider = OpenAI(), "gpt-4o", "openai"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client, model, provider = anthropic.Anthropic(), "claude-sonnet-4-6", "anthropic"
    else:
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY"); return

    await weather_server.start()
    await stats_server.start()
    print(f"S2SP Weather Agent (sync) — {provider}/{model}")
    print(f"Weather: {weather_server.s2sp_endpoint}  Stats: {stats_server.s2sp_endpoint}")
    print('Try: "Get wind alert statistics for California"\n')

    messages = []
    chat = chat_openai if provider == "openai" else chat_anthropic
    try:
        while True:
            msg = input("you> ").strip()
            if not msg or msg.lower() in ("quit", "exit", "q"): break
            print(f"\nagent> {await chat(client, model, messages, msg)}\n")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        await weather_server.stop()
        await stats_server.stop()


if __name__ == "__main__":
    asyncio.run(main())
