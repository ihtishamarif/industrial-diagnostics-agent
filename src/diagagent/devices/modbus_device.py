"""PID controller reached over Modbus TCP, behind the same Device interface.

The agent code does not change: it still calls info(), capture() and apply().
Differences from the simulated device are real-world ones: every call can fail
on the network, values travel as scaled integers, and a write is verified by
reading it back.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from pymodbus.client import ModbusTcpClient

from . import modbus_regs as R
from .base import ApprovalRequired, AuditLog, WriteRequest

WRITABLE = set(R.GAIN_REGS)
MAX_DURATION = 120.0
T_STEP = 1.0          # the setpoint steps one second into a test, as in the simulator
GRID_DT = 0.05        # captures are resampled to this uniform step (seconds)


class ModbusPIDDevice:
    def __init__(self, host: str = "127.0.0.1", port: int = 5020,
                 device_id: str = "modbus_pid_1", unit: int = 1,
                 audit: AuditLog | None = None, timeout: float = 3.0):
        self.device_id = device_id
        self.host, self.port, self.unit = host, port, unit
        self.audit = audit or AuditLog()
        self._client = ModbusTcpClient(host, port=port, timeout=timeout, retries=1)

    # ---- low-level access -----------------------------------------------------------
    def _connect(self) -> ModbusTcpClient:
        if not self._client.connected and not self._client.connect():
            raise ConnectionError(f"cannot reach Modbus device at {self.host}:{self.port}")
        return self._client

    def _read(self, addr: int, count: int = 1) -> list[int]:
        rr = self._connect().read_holding_registers(addr, count=count, device_id=self.unit)
        if rr.isError():
            raise ConnectionError(f"Modbus read of register {addr} failed: {rr}")
        return list(rr.registers)

    def _write(self, addr: int, value: int) -> None:
        rr = self._connect().write_register(addr, value, device_id=self.unit)
        if rr.isError():
            raise ConnectionError(f"Modbus write of register {addr} failed: {rr}")

    def close(self) -> None:
        self._client.close()

    def _wait_until_stopped(self, timeout: float = 2.0) -> None:
        """Wait for the PLC to confirm the loop is stopped (clock back at zero)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._read(R.R_TIME)[0] == 0:
                return
            time.sleep(0.005)
        raise TimeoutError("the PLC did not stop the loop")

    # ---- Device interface -----------------------------------------------------------
    def _gains(self) -> dict[str, float]:
        regs = self._read(R.R_KP, 3)
        return {k: R.dec_unsigned(v) for k, v in zip(("kp", "ki", "kd"), regs)}

    def _limits(self) -> tuple[float, float, float]:
        u_min, u_max, sp_max = self._read(R.R_UMIN, 3)
        return R.dec_unsigned(u_min), R.dec_unsigned(u_max), R.dec_unsigned(sp_max)

    def read_params(self) -> dict[str, float]:
        u_min, u_max, _ = self._limits()
        return {**self._gains(), "u_min": u_min, "u_max": u_max}

    def info(self) -> dict[str, Any]:
        u_min, u_max, sp_max = self._limits()
        return {
            "device_id": self.device_id,
            "type": type(self).__name__,
            "gains": self._gains(),
            "actuator_limits": {"u_min": u_min, "u_max": u_max},
            "step_test": {
                "default_setpoint": 1.0,
                "allowed_setpoint_range": [0.0, sp_max],
                "default_duration_s": 30.0,
                "max_duration_s": MAX_DURATION,
            },
            "writable": {k: list(v) for k, v in R.GAIN_LIMITS.items()},
            "transport": {"protocol": "modbus-tcp", "host": self.host, "port": self.port,
                          "unit": self.unit},
        }

    def apply(self, req: WriteRequest) -> dict[str, Any]:
        if not req.approved:
            self.audit.record(event="write_blocked", device=self.device_id, params=req.params)
            raise ApprovalRequired("write request was not approved")
        bad = set(req.params) - WRITABLE
        if bad:
            raise ValueError(f"not writable: {sorted(bad)}")
        for k, v in req.params.items():
            lo, hi = R.GAIN_LIMITS[k]
            if not (lo <= float(v) <= hi):
                raise ValueError(f"{k}={v} outside safe range [{lo}, {hi}]")
        before = self._gains()
        for k, v in req.params.items():
            self._write(R.GAIN_REGS[k], R.enc_unsigned(float(v)))
        time.sleep(0.05)  # let the PLC run a scan cycle, so it can reject the value
        after = self._gains()
        for k, v in req.params.items():
            if abs(after[k] - float(v)) > 1.5 / R.SCALE:
                self.audit.record(event="write_failed", device=self.device_id,
                                  wanted=req.params, read_back=after)
                raise RuntimeError(f"read-back mismatch for {k}: wrote {v}, device holds {after[k]}")
        self.audit.record(event="write_applied", device=self.device_id, before=before,
                          after=after, reason=req.reason, approved_by=req.approved_by)
        return {"device_id": self.device_id, "before": before, "after": after}

    def capture(self, setpoint: float = 1.0, duration: float = 30.0, **_: Any) -> pd.DataFrame:
        """Run a step test on the device and return t, setpoint, pv, u on a uniform grid."""
        _, _, sp_max = self._limits()
        if not (0.0 < float(setpoint) <= sp_max):
            raise ValueError(f"setpoint={setpoint} outside allowed range (0, {sp_max}] for "
                             f"{self.device_id}; use the default 1.0 unless told otherwise")
        if not (0.0 < float(duration) <= MAX_DURATION):
            raise ValueError(f"duration={duration} outside allowed range (0, {MAX_DURATION}] s")

        self._write(R.R_RUN, 0)
        self._wait_until_stopped()         # the PLC must have cleared the last run first
        self._write(R.R_SP, R.enc_signed(0.0))
        self._write(R.R_RUN, 1)            # starts a fresh loop with the clock at 0
        rows: list[tuple[float, float, float, float]] = []
        stepped, last_t = False, -1.0
        deadline = time.monotonic() + 3.0 * duration + 10.0
        try:
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError("step test did not finish in time")
                pv, sp, u, t_raw = self._read(R.R_PV, 4)
                t = t_raw / R.TIME_TICKS_PER_S
                if t > last_t:
                    rows.append((t, R.dec_signed(sp), R.dec_signed(pv), R.dec_signed(u)))
                    last_t = t
                if not stepped and t >= T_STEP:
                    self._write(R.R_SP, R.enc_signed(float(setpoint)))
                    stepped = True
                if t >= duration:
                    break
                time.sleep(0.002)
        finally:
            self._write(R.R_RUN, 0)        # always leave the loop stopped

        arr = np.array(rows)
        # Never hand stale or patchy data to the analysis: it would look like a real result.
        if arr[0, 0] > 0.5:
            raise RuntimeError(f"first sample is at t={arr[0, 0]:.2f} s: the PLC did not restart "
                               "the loop (stale registers)")
        if np.max(np.diff(arr[:, 0], prepend=0.0, append=duration)) > 1.0:
            raise RuntimeError("step test data has a gap longer than 1 s (PLC or link stalled)")
        grid = np.arange(0.0, duration + 1e-9, GRID_DT)
        hold = np.clip(np.searchsorted(arr[:, 0], grid, side="right") - 1, 0, len(arr) - 1)
        return pd.DataFrame({
            "t": grid,
            "setpoint": arr[hold, 1],
            "pv": np.interp(grid, arr[:, 0], arr[:, 2]),
            "u": arr[hold, 3],
        })
