"""An MCP client writes gains to a serial device, with the operator in the loop."""

import asyncio
import json
import os

import pytest

pytest.importorskip("mcp")
pytest.importorskip("serial")
if not hasattr(os, "openpty"):
    pytest.skip("needs a POSIX pseudo-terminal", allow_module_level=True)

from mcp import types  # noqa: E402
from mcp.client import Client  # noqa: E402

from diagagent.mcp_server import create_server, default_context  # noqa: E402
from diagagent.sim.serial_fw import SimulatedFirmware  # noqa: E402

GAINS = {"device_id": "serial_pid_1", "kp": 1.4, "ki": 0.9, "kd": 0.15, "reason": "retune"}


def _call(ctx, approve: bool):
    async def answer(context, params):
        return types.ElicitResult(action="accept", content={"approve": approve, "operator": "tester"})

    async def go():
        async with Client(create_server(ctx), elicitation_callback=answer, mode="legacy") as c:
            return await c.call_tool("set_pid_gains", GAINS)
    return json.loads(asyncio.run(go()).content[0].text)


def _ctx(fw):
    ctx = default_context(serial_port=fw.port)
    ctx.devices["serial_pid_1"].reset_delay = 0.0
    return ctx


def test_approved_write_reaches_the_firmware():
    with SimulatedFirmware(time_scale=10.0) as fw:
        assert _call(_ctx(fw), approve=True)["status"] == "applied"
        assert fw.gains["kp"] == pytest.approx(1.4)


def test_declined_write_leaves_the_firmware_untouched():
    with SimulatedFirmware(time_scale=10.0) as fw:
        assert _call(_ctx(fw), approve=False)["status"] == "denied"
        assert fw.gains["kp"] == 3.0
