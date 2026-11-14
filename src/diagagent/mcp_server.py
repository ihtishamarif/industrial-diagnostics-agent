"""MCP server that exposes the diagnostics toolbox to any MCP client.

MCP (Model Context Protocol) is an open standard that lets an AI client such as
Claude Desktop, Claude Code or your own agent discover and call tools served by
a separate process. Here the server wraps the same tools the LangGraph agent
uses, so there is one implementation and two ways to reach it.

Read tools are passed through unchanged. The one write tool, set_pid_gains, is
different: before anything touches a device, the server asks the human operator
through the client's own UI (MCP "elicitation"). The model cannot answer that
prompt. If the client cannot show it, or the operator declines, nothing is
written. Every decision goes into the audit log.

Run over stdio (the client starts the process):

    diagagent-mcp [--audit-out audit.json]

stdout carries the protocol, so this module never prints to stdout.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Context, MCPServer
from pydantic import BaseModel, Field

from .agent.tools import WRITE_TOOLS, ToolContext, build_tools
from .devices.sim_device import SimulatedPIDDevice

log = logging.getLogger("diagagent.mcp")


class WriteApproval(BaseModel):
    """What the operator sees and answers in the client."""

    approve: bool = Field(default=False, description="Tick to apply these gains to the device")
    operator: str = Field(default="", description="Your name, recorded in the audit log")


def _j(obj: Any) -> str:
    return json.dumps(obj, default=float)


def create_server(tool_ctx: ToolContext, audit_out: Path | None = None) -> MCPServer:
    server = MCPServer("diagagent")

    def save_audit() -> None:
        if audit_out is not None:
            audit_out.write_text(json.dumps(tool_ctx.audit.entries, indent=2, default=str))

    # Read tools: register the existing functions as they are.
    for t in build_tools(tool_ctx):
        if t.name in WRITE_TOOLS:
            continue
        server.add_tool(t.func, name=t.name, description=t.description)

    async def set_pid_gains(device_id: str, kp: float, ki: float, kd: float,
                            reason: str, ctx: Context) -> str:
        """WRITE ACTION. Change PID gains on a device. The human operator is asked
        to approve in their client before anything is written; if they decline,
        or the client cannot ask, nothing changes. Only call this after you have
        analyzed a dataset and proposed gains with suggest_pid_retune."""
        args = {"device_id": device_id, "kp": kp, "ki": ki, "kd": kd, "reason": reason}
        try:
            current = tool_ctx.device(device_id).read_params()
        except KeyError as e:
            return _j({"status": "error", "error": str(e)})

        def deny(why: str) -> str:
            tool_ctx.audit.record(event="write_denied", tool="set_pid_gains", args=args, why=why)
            save_audit()
            return _j({"status": "denied", "reason": why, "applied": False})

        message = (f"Apply new PID gains to '{device_id}'?\n"
                   f"  current: Kp={current.get('kp')}, Ki={current.get('ki')}, Kd={current.get('kd')}\n"
                   f"  new:     Kp={kp}, Ki={ki}, Kd={kd}\n"
                   f"  reason:  {reason}")
        try:
            answer = await ctx.elicit(message, WriteApproval)
        except Exception as e:  # client does not support elicitation, or it failed
            log.warning("elicitation failed: %s", e)
            return deny("client cannot ask the operator for approval")

        if answer.action != "accept" or not answer.data.approve:
            return deny(f"operator answered '{answer.action}'"
                        if answer.action != "accept" else "operator did not approve")

        operator = answer.data.operator.strip() or "mcp-operator"
        try:
            result = tool_ctx.apply_write("set_pid_gains", args, approved_by=operator)
        except Exception as e:  # the device's own range checks still apply
            tool_ctx.audit.record(event="write_failed", tool="set_pid_gains", args=args,
                                  approved_by=operator, error=str(e))
            save_audit()
            return _j({"status": "rejected_by_device", "error": str(e), "applied": False})
        save_audit()
        return _j(result)

    server.add_tool(set_pid_gains, name="set_pid_gains")
    return server


def default_context(modbus: str | None = None, serial_port: str | None = None) -> ToolContext:
    """The simulator by default, a Modbus TCP device (HOST:PORT) or a serial device."""
    if serial_port:
        from .devices.serial_device import SerialPIDDevice

        dev = SerialPIDDevice(serial_port)
    elif modbus:
        from .devices.modbus_device import ModbusPIDDevice

        host, _, port = modbus.rpartition(":")
        dev = ModbusPIDDevice(host=host or "127.0.0.1", port=int(port))
    else:
        dev = SimulatedPIDDevice.oscillating_demo()
    return ToolContext(devices={dev.device_id: dev})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="diagagent-mcp", description=__doc__.split("\n\n")[0])
    p.add_argument("--audit-out", type=Path, default=None,
                   help="write the audit log (approved and denied writes) to this JSON file")
    p.add_argument("--modbus", metavar="HOST:PORT", default=None,
                   help="use a Modbus TCP device (e.g. 127.0.0.1:5020) instead of the simulator")
    p.add_argument("--serial", metavar="PORT", default=None,
                   help="use a serial PID controller (e.g. /dev/ttyACM0) instead of the simulator")
    a = p.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    create_server(default_context(a.modbus, a.serial), a.audit_out).run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
