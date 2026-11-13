r"""LangGraph agent with a human approval gate for write actions.

Graph:

    START -> agent --(read tool calls)--> tools -----> agent
               |  \--(any write call)---> approval --> agent
               \--(no tool calls)-------> END

- agent:    the LLM, with all tools bound. It plans and decides which tool to call.
- tools:    LangGraph's ToolNode runs read-only tools.
- approval: pauses the graph with interrupt() and waits for a human decision.
            Approved writes go through ToolContext.apply_write(); rejected writes
            return a "rejected" result to the LLM. Read calls in the same turn
            run after the decision, so nothing runs twice when the graph resumes.

A checkpointer stores the graph state between the pause and the resume, keyed by
thread_id. That is what lets a CLI, a web UI or a test approve later.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from .tools import WRITE_TOOLS, ToolContext, build_tools

SYSTEM_PROMPT = """You are a diagnostics agent for embedded and industrial control systems.

Rules:
1. Never guess from memory. Get data with tools (capture_step_test or load_log_file),
   then run analyze_control_loop before you diagnose anything.
2. Every claim in your diagnosis must cite evidence returned by a tool: signal name,
   time window and metric value. Copy the evidence strings.
3. To change a device use set_pid_gains. The system will ask a human operator first.
   Only say a change was made if the tool result has status "applied".
4. After an applied change, capture a new step test, analyze it, and call
   compare_datasets to show before vs after.
5. Be short and technical. End with: Diagnosis, Evidence, Action taken, Result.
"""


def build_graph(llm: BaseChatModel, ctx: ToolContext, checkpointer: Any | None = None):
    tools = build_tools(ctx)
    by_name = {t.name: t for t in tools}
    read_tools = [t for t in tools if t.name not in WRITE_TOOLS]
    llm_t = llm.bind_tools(tools)

    def agent(state: MessagesState) -> dict:
        msgs = state["messages"]
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(SYSTEM_PROMPT), *msgs]
        return {"messages": [llm_t.invoke(msgs)]}

    def route(state: MessagesState) -> str:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        if not calls:
            return END
        return "approval" if any(c["name"] in WRITE_TOOLS for c in calls) else "tools"

    def approval(state: MessagesState) -> dict:
        last: AIMessage = state["messages"][-1]
        writes = [c for c in last.tool_calls if c["name"] in WRITE_TOOLS]
        # One interrupt for all writes in this turn. On resume this node restarts,
        # interrupt() returns the human's answer, and execution continues below.
        decision = interrupt({
            "type": "approval_request",
            "actions": [{"id": c["id"], "tool": c["name"], "args": c["args"]} for c in writes],
        })
        decisions = _normalize_decision(decision, writes)

        out: list[ToolMessage] = []
        for c in last.tool_calls:
            if c["name"] in WRITE_TOOLS:
                d = decisions[c["id"]]
                if d["approved"]:
                    try:
                        res = ctx.apply_write(c["name"], c["args"], approved_by=d.get("by", "operator"))
                    except Exception as e:  # range checks, device errors
                        res = {"status": "error", "error": str(e)}
                else:
                    ctx.audit.record(event="write_rejected", tool=c["name"], args=c["args"],
                                     note=d.get("note", ""))
                    res = {"status": "rejected", "note": d.get("note", "operator rejected")}
                out.append(ToolMessage(json.dumps(res, default=float), tool_call_id=c["id"],
                                       name=c["name"]))
            else:
                try:
                    content = by_name[c["name"]].invoke(c["args"])
                except Exception as e:
                    content = json.dumps({"error": str(e)})
                out.append(ToolMessage(content, tool_call_id=c["id"], name=c["name"]))
        return {"messages": out}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(read_tools, handle_tool_errors=True))
    g.add_node("approval", approval)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "approval": "approval", END: END})
    g.add_edge("tools", "agent")
    g.add_edge("approval", "agent")
    return g.compile(checkpointer=checkpointer or MemorySaver())


def _normalize_decision(decision: Any, writes: list[dict]) -> dict[str, dict]:
    """Accept True/False, {"approved": bool, ...}, or {call_id: {...}}."""
    if isinstance(decision, bool):
        return {c["id"]: {"approved": decision} for c in writes}
    if isinstance(decision, dict) and "approved" in decision:
        return {c["id"]: dict(decision) for c in writes}
    if isinstance(decision, dict):
        return {c["id"]: dict(decision.get(c["id"], {"approved": False})) for c in writes}
    return {c["id"]: {"approved": False, "note": "invalid decision"} for c in writes}
