"""Guardrails added after the first live Ollama run (qwen2.5:7b).

That run asked for a step test at setpoint 100 on a plant that can reach at most
10, then passed made-up actuator limits to the analysis and reported actuator
saturation as instability. These tests pin down that the tools, not the model,
now enforce the physical limits.
"""

import json
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from diagagent.agent.graph import build_graph
from diagagent.agent.runner import auto, run_session
from diagagent.agent.tools import ToolContext, build_tools
from diagagent.devices.sim_device import SimulatedPIDDevice


def _ctx():
    dev = SimulatedPIDDevice.oscillating_demo()
    return dev, ToolContext(devices={dev.device_id: dev})


def test_device_info_exposes_safe_test_range():
    dev, _ = _ctx()
    info = dev.info()
    assert info["actuator_limits"] == {"u_min": 0.0, "u_max": 5.0}
    assert info["step_test"]["allowed_setpoint_range"] == [0.0, 8.0]  # 0.8 * K=2 * u_max=5
    assert info["step_test"]["default_setpoint"] == 1.0


@pytest.mark.parametrize("kw", [{"setpoint": 100.0}, {"setpoint": -1.0}, {"duration": 600.0}])
def test_device_refuses_unreachable_step_tests(kw):
    dev, _ = _ctx()
    with pytest.raises(ValueError, match="outside allowed range"):
        dev.capture(**kw)


def test_capture_tool_uses_device_defaults_and_records_limits():
    dev, ctx = _ctx()
    tools = {t.name: t for t in build_tools(ctx)}
    out = json.loads(tools["capture_step_test"].invoke({"device_id": dev.device_id}))
    assert out["setpoint"] == 1.0 and out["duration_s"] == 30.0
    meta = ctx.store.meta(out["dataset_id"])
    assert (meta["u_min"], meta["u_max"]) == (0.0, 5.0)


def test_device_limits_override_model_supplied_limits():
    dev, ctx = _ctx()
    tools = {t.name: t for t in build_tools(ctx)}
    ds = json.loads(tools["capture_step_test"].invoke({"device_id": dev.device_id}))["dataset_id"]
    honest = json.loads(tools["analyze_control_loop"].invoke({"dataset_id": ds}))
    made_up = json.loads(tools["analyze_control_loop"].invoke(
        {"dataset_id": ds, "u_min": 0, "u_max": 0.5}))
    assert made_up["actuator_limits_from"] == "device"
    assert made_up["anomalies"] == honest["anomalies"]
    assert honest["oscillating"] is True


class ScriptedLLM(BaseChatModel):
    """Replays a fixed list of AI messages, one per agent turn."""
    script: list[Any]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        n_ai = sum(isinstance(m, AIMessage) for m in messages)
        return ChatResult(generations=[ChatGeneration(message=self.script[n_ai])])


def _call(i: int, name: str, args: dict) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"c{i}", "type": "tool_call"}])


def test_replay_of_bad_live_run_gets_error_then_recovers():
    dev, ctx = _ctx()
    llm = ScriptedLLM(script=[
        _call(0, "list_devices", {}),
        _call(1, "capture_step_test", {"device_id": "sim_pid_1", "duration": 60, "setpoint": 100}),
        _call(2, "capture_step_test", {"device_id": "sim_pid_1"}),
        _call(3, "analyze_control_loop", {"dataset_id": "ds1", "u_min": 0, "u_max": 5}),
        AIMessage("done"),
    ])
    res = run_session(build_graph(llm, ctx), "diagnose", auto(False))
    caps = [m for m in res["messages"] if m.type == "tool" and m.name == "capture_step_test"]
    assert "outside allowed range" in caps[0].content          # bad request refused
    assert json.loads(caps[1].content)["dataset_id"] == "ds1"   # retry with defaults works
    assert len(ctx.store.list()) == 1                           # refused test stored nothing
    analysis = json.loads([m for m in res["messages"] if m.name == "analyze_control_loop"][0].content)
    assert analysis["oscillating"] is True
    assert dev.read_params()["kp"] == 4.0
