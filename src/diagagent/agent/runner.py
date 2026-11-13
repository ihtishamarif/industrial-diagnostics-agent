"""Drive a graph run to completion, handing approval requests to a callback."""

from __future__ import annotations

import uuid
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

Approver = Callable[[dict[str, Any]], Any]


def cli_approver(request: dict[str, Any]) -> dict[str, Any]:
    print("\n=== APPROVAL REQUIRED ===")
    for a in request["actions"]:
        print(f"  tool: {a['tool']}")
        for k, v in a["args"].items():
            print(f"    {k}: {v}")
    ans = input("Approve this write? [y/N]: ").strip().lower()
    if ans in {"y", "yes"}:
        return {"approved": True, "by": "cli-operator"}
    note = input("Reason for rejection (optional): ").strip()
    return {"approved": False, "note": note or "operator rejected"}


def auto(approve: bool) -> Approver:
    def _a(_req: dict[str, Any]) -> dict[str, Any]:
        return {"approved": approve, "by": "auto" if approve else None,
                "note": "" if approve else "auto-reject"}
    return _a


def run_session(app, user_message: str, approver: Approver,
                thread_id: str | None = None, max_steps: int = 50,
                on_event: Callable[[dict], None] | None = None) -> dict[str, Any]:
    cfg = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())},
           "recursion_limit": max_steps}
    payload: Any = {"messages": [HumanMessage(user_message)]}
    approvals: list[dict[str, Any]] = []
    while True:
        for ev in app.stream(payload, cfg, stream_mode="updates"):
            if on_event:
                on_event(ev)
        state = app.get_state(cfg)
        pending = [i for t in state.tasks for i in t.interrupts]
        if not pending:
            break
        req = pending[0].value
        decision = approver(req)
        approvals.append({"request": req, "decision": decision})
        payload = Command(resume=decision)
    msgs = app.get_state(cfg).values["messages"]
    final = next((m for m in reversed(msgs) if isinstance(m, AIMessage) and not m.tool_calls), None)
    return {"final": final.content if final else "", "messages": msgs, "approvals": approvals,
            "thread_id": cfg["configurable"]["thread_id"]}
