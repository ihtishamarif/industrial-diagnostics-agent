"""Tools the agent can call.

A "tool" here is a Python function with a typed signature and a docstring. The
LLM sees the name, docstring and argument schema, and asks the graph to run it.
Tools return JSON strings so the LLM gets compact, structured results.

Read tools run freely. Write tools (names in WRITE_TOOLS) are never executed
directly: the graph stops at the approval node, asks a human, and only then calls
ToolContext.apply_write(). Calling a write tool function directly raises
ApprovalRequired, so there is no path that skips the human.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool, tool

from ..devices.base import ApprovalRequired, AuditLog, Device, WriteRequest
from ..io.log_loader import DatasetStore, load_log, summarize
from ..signals import control as ctl
from ..signals.control import LoopReport

WRITE_TOOLS = {"set_pid_gains"}


@dataclass
class ToolContext:
    devices: dict[str, Device]
    store: DatasetStore = field(default_factory=DatasetStore)
    audit: AuditLog = field(default_factory=AuditLog)
    reports: dict[str, LoopReport] = field(default_factory=dict)

    def device(self, device_id: str) -> Device:
        if device_id not in self.devices:
            raise KeyError(f"unknown device_id '{device_id}'. Known: {list(self.devices)}")
        return self.devices[device_id]

    def apply_write(self, tool_name: str, args: dict[str, Any], approved_by: str) -> dict[str, Any]:
        """Called by the approval node only, after a human said yes."""
        if tool_name != "set_pid_gains":
            raise ValueError(f"unknown write tool {tool_name}")
        req = WriteRequest(
            device_id=args["device_id"],
            params={k: float(args[k]) for k in ("kp", "ki", "kd") if k in args},
            reason=args.get("reason", ""),
            approved=True, approved_by=approved_by,
        )
        result = self.device(req.device_id).apply(req)
        self.audit.record(event="approved_write", tool=tool_name, args=args, approved_by=approved_by)
        return {"status": "applied", **result}


def _j(obj: Any) -> str:
    return json.dumps(obj, default=float, indent=None)


def _compact_report(r: LoopReport) -> dict[str, Any]:
    out = {"anomalies": [], "hypotheses": [], "metrics": {}}
    for f in r.findings:
        if f.detector == "step_response":
            out["metrics"] = {k: f.details.get(k) for k in
                              ("overshoot_pct", "rise_time_s", "settling_time_s", "settled",
                               "steady_state_error")}
        if f.detected:
            out["anomalies"].append({"detector": f.detector, "severity": f.severity.value,
                                     "summary": f.summary,
                                     "evidence": [e.cite() for e in f.evidence]})
    out["hypotheses"] = [{"cause": h.cause, "confidence": h.confidence,
                          "based_on": list(h.based_on), "suggested_fix": h.suggested_fix}
                         for h in r.hypotheses]
    return out


def build_tools(ctx: ToolContext) -> list[BaseTool]:
    @tool
    def list_devices() -> str:
        """List connected devices and their ids."""
        return _j([{"device_id": d, "type": type(dev).__name__} for d, dev in ctx.devices.items()])

    @tool
    def get_device_info(device_id: str) -> str:
        """Read a device's current PID gains, actuator limits, allowed step-test
        setpoint range and writable parameter ranges. Call this before testing."""
        return _j(ctx.device(device_id).info())

    @tool
    def capture_step_test(device_id: str, setpoint: float | None = None,
                          duration: float | None = None) -> str:
        """Run a setpoint step test on a device and record t, setpoint, pv, u.
        Leave setpoint and duration empty to use the device defaults; the device
        rejects values outside its allowed range. Returns a dataset_id."""
        dev = ctx.device(device_id)
        test = dev.info()["step_test"]
        sp = test["default_setpoint"] if setpoint is None else setpoint
        dur = test["default_duration_s"] if duration is None else duration
        df = dev.capture(setpoint=sp, duration=dur)
        lim = dev.info()["actuator_limits"]
        ds = ctx.store.add(df, source=f"step_test:{device_id}", device_id=device_id,
                           u_min=lim["u_min"], u_max=lim["u_max"])
        return _j({"dataset_id": ds, "setpoint": sp, "duration_s": dur, **summarize(df)})

    @tool
    def load_log_file(path: str, time_col: str = "t") -> str:
        """Load a CSV/TSV/Parquet/JSON log file. Returns a dataset_id and a signal summary."""
        df = load_log(path, time_col=time_col)
        ds = ctx.store.add(df, source=path)
        return _j({"dataset_id": ds, **summarize(df)})

    @tool
    def list_datasets() -> str:
        """List datasets loaded so far."""
        return _j(ctx.store.list())

    @tool
    def analyze_control_loop(dataset_id: str, sp_col: str = "setpoint", pv_col: str = "pv",
                             u_col: str = "u", u_min: float | None = None,
                             u_max: float | None = None) -> str:
        """Run deterministic detectors (step response, oscillation, saturation, noise,
        dropouts) on a closed-loop dataset. Returns anomalies with cited evidence and
        ranked hypotheses. For device step tests the actuator limits come from the
        device; u_min/u_max are only used for log files where the user gave them."""
        df = ctx.store.get(dataset_id)
        meta = ctx.store.meta(dataset_id)
        limits_from = "user"
        if "u_max" in meta:  # captured from a device: its limits win over model input
            u_min, u_max, limits_from = meta["u_min"], meta["u_max"], "device"
        elif u_min is None or u_max is None:
            limits_from = "unknown (saturation check skipped)"
        r = ctl.analyze_control_loop(df, sp_col=sp_col, pv_col=pv_col,
                                 u_col=u_col if u_col in df else None,
                                 u_min=u_min, u_max=u_max)
        ctx.reports[dataset_id] = r
        osc = r.get("oscillation")
        return _j({"dataset_id": dataset_id, "actuator_limits_from": limits_from,
                   "oscillating": bool(osc and osc.detected), **_compact_report(r)})

    @tool
    def suggest_pid_retune(dataset_id: str, device_id: str) -> str:
        """Propose new PID gains from the analysis of dataset_id and the device's current
        gains. Call it whenever analyze_control_loop reports oscillation or a bad step
        response. Does not change the device."""
        if dataset_id not in ctx.reports:
            return _j({"error": f"run analyze_control_loop on {dataset_id} first"})
        p = ctx.device(device_id).read_params()
        prop = ctl.suggest_pid_retune(ctx.reports[dataset_id], p["kp"], p["ki"], p.get("kd", 0.0))
        if prop is None:
            return _j({"proposal": None, "message": "No retune needed."})
        return _j({"current": {k: p[k] for k in ("kp", "ki", "kd")}, "proposal": prop.as_dict()})

    @tool
    def compare_datasets(before_id: str, after_id: str) -> str:
        """Compare step-response metrics of two analyzed datasets (before and after a change)."""
        rows = {}
        for name, ds in (("before", before_id), ("after", after_id)):
            if ds not in ctx.reports:
                return _j({"error": f"run analyze_control_loop on {ds} first"})
            rows[name] = _compact_report(ctx.reports[ds])["metrics"]
            osc = ctx.reports[ds].get("oscillation")
            rows[name]["oscillating"] = bool(osc and osc.detected)
            rows[name]["n_anomalies"] = sum(f.detected for f in ctx.reports[ds].findings)
        return _j(rows)

    @tool
    def set_pid_gains(device_id: str, kp: float, ki: float, kd: float, reason: str) -> str:
        """WRITE ACTION. Change PID gains on a device. Requires human approval; the
        system will pause and ask the operator before anything is written."""
        raise ApprovalRequired("set_pid_gains must go through the approval node")

    return [list_devices, get_device_info, capture_step_test, load_log_file, list_datasets,
            analyze_control_loop, suggest_pid_retune, compare_datasets, set_pid_gains]
