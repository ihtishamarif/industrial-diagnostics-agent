"""Agent graph tests. They use the offline RuleBasedPlanner, so no LLM is needed."""

import json

import pytest

from diagagent.agent.graph import build_graph
from diagagent.agent.llm import get_llm
from diagagent.agent.runner import auto, run_session
from diagagent.agent.tools import ToolContext, build_tools
from diagagent.devices.base import ApprovalRequired
from diagagent.devices.sim_device import SimulatedPIDDevice
from diagagent.sim.pid_sim import LoopConfig, PIDGains, simulate_step

PROMPT = "The loop is unstable. Diagnose and fix it."


def _setup():
    dev = SimulatedPIDDevice.oscillating_demo()
    ctx = ToolContext(devices={dev.device_id: dev})
    return dev, ctx, build_graph(get_llm("offline"), ctx)


def test_graph_pauses_before_any_write():
    dev, ctx, app = _setup()
    cfg = {"configurable": {"thread_id": "t1"}}
    from langchain_core.messages import HumanMessage
    list(app.stream({"messages": [HumanMessage(PROMPT)]}, cfg))
    state = app.get_state(cfg)
    interrupts = [i for t in state.tasks for i in t.interrupts]
    assert len(interrupts) == 1
    req = interrupts[0].value
    assert req["type"] == "approval_request"
    assert req["actions"][0]["tool"] == "set_pid_gains"
    assert dev.read_params()["kp"] == 4.0  # nothing written while paused


def test_rejected_write_leaves_device_unchanged():
    dev, ctx, app = _setup()
    res = run_session(app, PROMPT, auto(False))
    assert dev.read_params()["kp"] == 4.0
    assert len(res["approvals"]) == 1
    assert "rejected" in res["final"]
    assert any(e["event"] == "write_rejected" for e in ctx.audit.entries)


def test_approved_write_fixes_loop_and_is_audited():
    dev, ctx, app = _setup()
    res = run_session(app, PROMPT, auto(True))
    assert dev.read_params()["kp"] < 4.0
    tool_msgs = {m.name: m for m in res["messages"] if m.type == "tool"}
    cmp_ = json.loads(tool_msgs["compare_datasets"].content)
    assert cmp_["before"]["oscillating"] is True
    assert cmp_["after"]["oscillating"] is False
    assert any(e["event"] == "write_applied" for e in dev.audit.entries)
    assert "evidence:" in res["final"]


def test_write_tool_cannot_be_called_directly():
    dev, ctx, _ = _setup()
    tools = {t.name: t for t in build_tools(ctx)}
    with pytest.raises(ApprovalRequired):
        tools["set_pid_gains"].func(device_id=dev.device_id, kp=1, ki=1, kd=0, reason="x")
    assert dev.read_params()["kp"] == 4.0


def test_log_file_analysis_path(tmp_path):
    p = tmp_path / "loop.csv"
    simulate_step(LoopConfig(PIDGains(4.0, 1.5), u_max=5.0)).to_csv(p, index=False)
    ctx = ToolContext(devices={})
    app = build_graph(get_llm("offline"), ctx)
    res = run_session(app, f"Analyze {p}", auto(False))
    assert res["approvals"] == []
    assert "oscillation" in res["final"].lower()
