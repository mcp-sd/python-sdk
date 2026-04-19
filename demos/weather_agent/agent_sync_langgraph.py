#!/usr/bin/env python3
"""Weather Agent — Sync Mode on LangGraph.

What this demo proves
---------------------
The LLM sees only ``abstract`` + a ``ddi://...`` handle. The ``body`` rows
transit the ``DDIBuffer`` in the agent process but never reach the LLM
context. Each tool is a LangChain ``BaseTool`` wrapped by
``mcp_sd.agent.adapters.langgraph.wrap_tool``, so its I/O runs through the
shared ``S2SPDispatcher``. ``make_sd_tool_node`` gives a drop-in
``ToolNode`` for a classic LangGraph ReAct loop.

Run::

    export ANTHROPIC_API_KEY=<your-anthropic-api-key>  # or OPENAI_API_KEY
    python demos/weather_agent/agent_sync_langgraph.py
"""

from __future__ import annotations

import asyncio, json, os, sys

sys.path.insert(0, "src")
os.environ.setdefault("NO_PROXY", "api.anthropic.com,api.openai.com,api.weather.gov")

from weather_server import server as weather_server  # noqa: E402
from stats_server import server as stats_server  # noqa: E402

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
    from langchain_core.tools import tool as lc_tool  # noqa: E402
    from langgraph.graph import END, MessagesState, StateGraph  # noqa: E402
except ImportError:
    print("Install: pip install 'langgraph>=0.2' 'langchain-core>=0.3'")
    raise SystemExit(0)

from mcp_sd.agent import S2SPDispatcher  # noqa: E402
from mcp_sd.agent.adapters.langgraph import make_sd_tool_node, wrap_tool  # noqa: E402


async def call_mcp_tool(server, name, args):
    """Invoke a FastMCP tool; return the first text block."""
    result = await server.mcp.call_tool(name, args)
    content = result[0] if isinstance(result, tuple) else result
    for block in content:
        if hasattr(block, "text"):
            return block.text
    return str(content)


def pick_llm():
    """Return a tool-calling chat model or (None, None) if unavailable."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore
            return ChatAnthropic(model="claude-sonnet-4-6", temperature=0), "anthropic"
        except ImportError:
            pass
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
            return ChatOpenAI(model="gpt-4o", temperature=0), "openai"
        except ImportError:
            pass
    return None, None


# ── Dispatcher + tool definitions ─────────────────────────────────
# Shared dispatcher -> its DDIBuffer is the only place body rows live.

dispatcher = S2SPDispatcher()


@lc_tool
async def get_alerts(area: str, abstract_domains: str = "event,severity,urgency,status,headline,areaDesc") -> str:
    """Fetch weather alerts for a US state (e.g. CA, TX, NY).

    Returns abstract rows + a resource_url handle. Keep abstract_domains minimal.
    """
    # mode="sync" makes weather_server return abstract + body inline. The
    # agent-side dispatcher then splits them: body -> DDIBuffer, the
    # payload the LLM receives carries only abstract + ddi:// handle.
    return await call_mcp_tool(
        weather_server,
        "get_alerts",
        {"area": area, "abstract_domains": abstract_domains, "mode": "sync"},
    )


@lc_tool
async def draw_chart(abstract_data: str, resource_url: str = "") -> str:
    """Generate an 8-panel weather statistics chart from filtered abstract rows.

    Pass ``abstract_data`` (JSON array) and the ``resource_url`` handle
    returned by ``get_alerts``.
    """
    # Dispatcher intercepts the ddi:// resource_url, pops the cached body
    # from DDIBuffer, and hands body_data to the consumer server inline.
    return await call_mcp_tool(
        stats_server,
        "draw_chart",
        {"abstract_data": abstract_data, "resource_url": resource_url},
    )


# Wrap each BaseTool so its _arun routes through the dispatcher. wrap_tool
# mutates the tool in place, preserving its name/description/schema.
wrapped_tools = [wrap_tool(dispatcher, get_alerts), wrap_tool(dispatcher, draw_chart)]

# ToolNode drop-in; dispatcher is shared so both tools see the same DDIBuffer.
tool_node, dispatcher = make_sd_tool_node(wrapped_tools, dispatcher=dispatcher)


SYSTEM = (
    "You are a weather agent using MCP-SD sync mode. Tools:\n"
    "1) get_alerts(area, abstract_domains) — abstract rows + a resource_url handle.\n"
    "2) draw_chart(abstract_data, resource_url) — pass the filtered rows + handle through.\n"
    "Filter abstract rows yourself. You never see body rows; pass resource_url untouched."
)


# ── Graph construction ────────────────────────────────────────────

def build_graph(llm):
    model = llm.bind_tools(wrapped_tools)

    async def agent_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM)] + state["messages"]
        response = await model.ainvoke(messages)
        return {"messages": [response]}

    def route(state: MessagesState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# ── Main REPL ─────────────────────────────────────────────────────

async def main():
    llm, provider = pick_llm()
    if llm is None:
        print("Install langchain-anthropic (+ANTHROPIC_API_KEY) or langchain-openai (+OPENAI_API_KEY).")
        return

    await weather_server.start()
    await stats_server.start()
    app = build_graph(llm)

    print(f"S2SP Weather Agent (sync / langgraph) — {provider}")
    print(f"Weather: {weather_server.s2sp_endpoint}  Stats: {stats_server.s2sp_endpoint}")
    print('Try: "Get wind alert statistics for California"\n')

    history = []
    try:
        while True:
            try:
                msg = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not msg or msg.lower() in ("quit", "exit", "q"):
                break
            history.append(HumanMessage(content=msg))
            out = await app.ainvoke({"messages": history})
            history = out["messages"]
            for m in history[-5:]:
                for tc in getattr(m, "tool_calls", []) or []:
                    snippet = json.dumps(tc.get("args", {}), default=str)[:100]
                    print(f"  [tool] {tc.get('name')}({snippet})")
            text = getattr(history[-1], "content", "") or ""
            if isinstance(text, list):
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
            print(f"\nagent> {text}\n")
    finally:
        dispatcher.reset()  # drop unreleased DDIBuffer rows
        await weather_server.stop()
        await stats_server.stop()


if __name__ == "__main__":
    asyncio.run(main())
