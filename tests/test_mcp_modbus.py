"""An MCP client writes gains to a Modbus device, with the operator in the loop."""

import asyncio
import json

import pytest

pytest.importorskip("mcp")
pytest.importorskip("pymodbus")

from mcp import types  # noqa: E402
from mcp.client import Client  # noqa: E402

from diagagent.devices.modbus_device import ModbusPIDDevice  # noqa: E402
from diagagent.mcp_server import create_server, default_context  # noqa: E402
from diagagent.sim.modbus_plc import SimulatedPLC, free_port  # noqa: E402

GAINS = {"device_id": "modbus_pid_1", "kp": 1.8, "ki": 0.5, "kd": 0.4, "reason": "retune"}


def _call(ctx, approve: bool):
    async def answer(context, params):
        return types.ElicitResult(action="accept", content={"approve": approve, "operator": "tester"})

    async def go():
        async with Client(create_server(ctx), elicitation_callback=answer, mode="legacy") as c:
            return await c.call_tool("set_pid_gains", GAINS)
    return json.loads(asyncio.run(go()).content[0].text)


@pytest.fixture
def plc_port():
    port = free_port()
    with SimulatedPLC(port=port, time_scale=10.0):
        yield port


def test_approved_write_reaches_the_plc_registers(plc_port):
    ctx = default_context(f"127.0.0.1:{plc_port}")
    assert _call(ctx, approve=True)["status"] == "applied"
    check = ModbusPIDDevice(port=plc_port)
    assert check.read_params()["kp"] == pytest.approx(1.8, abs=1e-3)
    check.close()


def test_declined_write_leaves_the_plc_untouched(plc_port):
    ctx = default_context(f"127.0.0.1:{plc_port}")
    assert _call(ctx, approve=False)["status"] == "denied"
    check = ModbusPIDDevice(port=plc_port)
    assert check.read_params()["kp"] == 4.0
    check.close()
