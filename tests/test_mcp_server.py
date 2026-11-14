"""MCP server tests: tools are exposed, and writes need the operator's approval."""

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from mcp import types  # noqa: E402
from mcp.client import Client  # noqa: E402

from diagagent.mcp_server import create_server, default_context  # noqa: E402

GAINS = {"device_id": "sim_pid_1", "kp": 1.8, "ki": 0.5, "kd": 0.4, "reason": "retune"}


def _run(coro):
    return asyncio.run(coro)


def _text(result) -> dict:
    return json.loads(result.content[0].text)


def _answer(approve: bool, operator: str = "tester"):
    async def cb(context, params):
        return types.ElicitResult(action="accept", content={"approve": approve, "operator": operator})
    return cb


async def _call(ctx, name, args, cb=None):
    async with Client(create_server(ctx), elicitation_callback=cb, mode="legacy") as c:
        return await c.call_tool(name, args)


def test_tools_are_listed():
    async def go():
        async with Client(create_server(default_context())) as c:
            return {t.name for t in (await c.list_tools()).tools}
    names = _run(go())
    assert {"list_devices", "set_pid_gains"} <= names


def test_read_tool_passes_through():
    out = _text(_run(_call(default_context(), "list_devices", {})))
    assert out[0]["device_id"] == "sim_pid_1"


def test_write_applied_after_operator_approves():
    ctx = default_context()
    out = _text(_run(_call(ctx, "set_pid_gains", GAINS, _answer(True, "ihtisham"))))
    assert out["status"] == "applied"
    assert ctx.device("sim_pid_1").read_params()["kp"] == 1.8
    assert ctx.audit.entries[-1]["approved_by"] == "ihtisham"


def test_write_denied_when_operator_declines():
    ctx = default_context()
    before = ctx.device("sim_pid_1").read_params()
    out = _text(_run(_call(ctx, "set_pid_gains", GAINS, _answer(False))))
    assert out["status"] == "denied"
    assert ctx.device("sim_pid_1").read_params() == before
    assert ctx.audit.entries[-1]["event"] == "write_denied"


def test_write_denied_when_client_cannot_ask():
    ctx = default_context()
    before = ctx.device("sim_pid_1").read_params()
    out = _text(_run(_call(ctx, "set_pid_gains", GAINS, cb=None)))
    assert out["status"] == "denied"
    assert ctx.device("sim_pid_1").read_params() == before


def test_device_range_check_still_applies_after_approval():
    ctx = default_context()
    out = _text(_run(_call(ctx, "set_pid_gains", {**GAINS, "kp": 99.0}, _answer(True))))
    assert out["status"] == "rejected_by_device"
