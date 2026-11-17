"""Command line entry point.

  diagagent demo     [--backend offline|ollama|anthropic] [--approve ask|yes|no] [--out results]
  diagagent analyze  LOG.csv [--sp setpoint --pv pv --u u --u-min 0 --u-max 5]
"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
from pathlib import Path

from .agent.graph import build_graph
from .agent.llm import get_llm
from .agent.runner import auto, cli_approver, run_session
from .agent.tools import ToolContext
from .devices.sim_device import SimulatedPIDDevice
from .io.log_loader import load_log
from .report import plot_before_after, results_table
from .signals.control import analyze_control_loop

DEMO_PROMPT = ("The loop on the connected PID controller is unstable. Find out why, "
               "propose a fix, apply it if the operator approves, and show before vs after.")


def _print_event(ev: dict) -> None:
    for node, upd in ev.items():
        if node == "__interrupt__" or not isinstance(upd, dict):
            continue
        for m in upd.get("messages", []):
            calls = getattr(m, "tool_calls", None)
            if calls:
                for c in calls:
                    print(f"[{node}] -> {c['name']}({json.dumps(c['args'], default=float)})")
            elif m.type == "tool":
                txt = str(m.content)
                print(f"[{node}] <- {m.name}: {txt[:160]}{'...' if len(txt) > 160 else ''}")


def _make_device(kind: str, serial_port: str | None = None, fw_plant: str = "servo"):
    """Build the device to diagnose. Returns (device, cleanup function)."""
    if kind == "serial":
        from .devices.serial_device import SerialPIDDevice

        if serial_port:  # real hardware: opening the port resets an Arduino
            dev = SerialPIDDevice(serial_port)
            print(f"Using the serial device on {serial_port}")
            return dev, dev.close
        from .sim.serial_fw import SimulatedFirmware

        fw = SimulatedFirmware(time_scale=5.0, plant=fw_plant).start()  # stand-in on a pseudo-terminal
        dev = SerialPIDDevice(fw.port, reset_delay=0.0)
        print(f"Using simulated Arduino firmware ({fw_plant} plant) on {fw.port}")
        return dev, lambda: (dev.close(), fw.stop())
    if kind == "modbus":
        from .devices.modbus_device import ModbusPIDDevice
        from .sim.modbus_plc import SimulatedPLC, free_port

        plc = SimulatedPLC(port=free_port()).start()  # a PLC stand-in, started in-process
        dev = ModbusPIDDevice(port=plc.port)
        print(f"Using a Modbus TCP device at 127.0.0.1:{plc.port}")
        return dev, lambda: (dev.close(), plc.stop())
    return SimulatedPIDDevice.oscillating_demo(), lambda: None


def cmd_demo(a: argparse.Namespace) -> int:
    dev, cleanup = _make_device(a.device, a.serial_port, a.fw_plant)
    atexit.register(cleanup)
    gains_before = dev.read_params()
    ctx = ToolContext(devices={dev.device_id: dev})
    app = build_graph(get_llm(a.backend, a.model), ctx)
    approver = {"ask": cli_approver, "yes": auto(True), "no": auto(False)}[a.approve]
    res = run_session(app, DEMO_PROMPT, approver, on_event=_print_event)
    print("\n================ FINAL ================\n")
    print(res["final"])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_log.json").write_text(json.dumps(ctx.audit.entries + dev.audit.entries,
                                                   indent=2, default=str))
    caps = [d["dataset_id"] for d in ctx.store.list() if d["source"].startswith("step_test")]
    if len(caps) >= 2 and caps[0] in ctx.reports and caps[-1] in ctx.reports:
        b, af = ctx.store.get(caps[0]), ctx.store.get(caps[-1])
        plot_before_after(b, af, out / "pid_before_after.png")

        def m(ds):
            r = ctx.reports[ds]
            step = r.get("step_response").details
            osc = r.get("oscillation")
            return {**step, "oscillating": bool(osc and osc.detected),
                    "n_anomalies": sum(f.detected for f in r.findings)}
        table = results_table(m(caps[0]), m(caps[-1]), gains_before, dev.read_params())
        (out / "results.md").write_text(
            f"# PID oscillation demo\n\nBackend: `{a.backend}`\n\n{table}\n\n"
            f"![before vs after](pid_before_after.png)\n")
        print(f"\nSaved {out/'results.md'}, {out/'pid_before_after.png'}, {out/'audit_log.json'}")
    else:
        print(f"\nNo after-capture (write not applied). Audit log in {out/'audit_log.json'}")
    return 0


def cmd_analyze(a: argparse.Namespace) -> int:
    df = load_log(a.log, time_col=a.time_col)
    r = analyze_control_loop(df, sp_col=a.sp, pv_col=a.pv,
                             u_col=a.u if a.u in df else None, u_min=a.u_min, u_max=a.u_max)
    print(json.dumps(r.to_dict(), indent=2, default=float))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="diagagent")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="simulated PID oscillation diagnosis and retune")
    d.add_argument("--backend", default="ollama", choices=["ollama", "anthropic", "offline"])
    d.add_argument("--model", default=None)
    d.add_argument("--device", default="sim", choices=["sim", "modbus", "serial"],
                   help="sim: in-process simulator; modbus: simulated PLC over Modbus TCP; "
                        "serial: Arduino-style firmware over a serial port (simulated unless --serial-port)")
    d.add_argument("--serial-port", default=None, metavar="PORT",
                   help="with --device serial: a real port such as /dev/ttyACM0")
    d.add_argument("--fw-plant", default="servo", choices=["servo", "led"],
                   help="with simulated serial firmware: servo + pot, or LED + LDR")
    d.add_argument("--approve", default="ask", choices=["ask", "yes", "no"])
    d.add_argument("--out", default="results")
    d.set_defaults(func=cmd_demo)

    z = sub.add_parser("analyze", help="run detectors on a log file (no LLM)")
    z.add_argument("log")
    z.add_argument("--time-col", default="t")
    z.add_argument("--sp", default="setpoint")
    z.add_argument("--pv", default="pv")
    z.add_argument("--u", default="u")
    z.add_argument("--u-min", type=float, default=None)
    z.add_argument("--u-max", type=float, default=None)
    z.set_defaults(func=cmd_analyze)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
