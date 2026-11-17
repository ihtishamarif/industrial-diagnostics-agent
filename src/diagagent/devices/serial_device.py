"""PID controller reached over a serial port (Arduino Uno etc.), same Device interface.

Real-world differences from the other devices: opening the port resets an Arduino
(so connecting takes ~2 s), bytes can arrive damaged, and the loop runs in real
time on the microcontroller. Writes are verified by reading the gains back.
"""

from __future__ import annotations

import functools
import threading
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import serial

from . import serial_protocol as P
from .base import ApprovalRequired, AuditLog, WriteRequest

WRITABLE = set(P.GAIN_LIMITS)
MAX_DURATION = 60.0
DEFAULT_DURATION = 10.0
DEFAULT_SETPOINT = 0.6
T_STEP = 1.0
GRID_DT = 0.05
MAX_BAD_FRACTION = 0.05   # more damaged data lines than this and the capture is refused


def _exclusive(method):
    """Hold the device lock for the whole call.

    The agent runs the tool calls of one step in parallel threads, and a serial port is one
    byte stream: two threads reading it at once steal each other's bytes (damaged lines,
    "multiple access on port"). A lock per device makes a call wait for the one before it.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class SerialPIDDevice:
    def __init__(self, port: str, baud: int = P.BAUD, device_id: str = "serial_pid_1",
                 audit: AuditLog | None = None, reset_delay: float = 2.0,
                 timeout: float = 3.0):
        self.device_id = device_id
        self.port, self.baud = port, baud
        self.audit = audit or AuditLog()
        self.reset_delay, self.timeout = reset_delay, timeout
        self._ser: serial.Serial | None = None
        self._buf = b""
        self._lock = threading.RLock()       # re-entrant: public methods call each other

    # ---- low-level access -----------------------------------------------------------
    def _open(self) -> serial.Serial:
        if self._ser is not None and self._ser.is_open:
            return self._ser
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.02)
        except serial.SerialException as e:
            raise ConnectionError(f"cannot open serial port {self.port}: {e}") from e
        self._ser, self._buf = ser, b""
        time.sleep(self.reset_delay)         # an Arduino reboots when the port opens
        ser.reset_input_buffer()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:   # wait until the firmware answers
            try:
                if self._command("PING", timeout=0.5, check=False).startswith("OK diagfw"):
                    return ser
            except (TimeoutError, ConnectionError):
                pass
        ser.close()
        raise ConnectionError(f"no diagfw firmware answered on {self.port}")

    @_exclusive
    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
        self._ser = None

    def _readline(self, timeout: float) -> str | None:
        ser = self._ser
        deadline = time.monotonic() + timeout
        while True:
            i = self._buf.find(b"\n")
            if i >= 0:
                line, self._buf = self._buf[:i], self._buf[i + 1:]
                return line.decode("ascii", errors="replace").strip()
            if time.monotonic() > deadline:
                return None
            self._buf += ser.read(ser.in_waiting or 1)

    def _command(self, cmd: str, timeout: float | None = None,
                 on_data: Callable[[str], None] | None = None, check: bool = True) -> str:
        """Send one command and return the text after OK. Data lines go to on_data."""
        if self._ser is None:
            self._open()
        self._ser.write(cmd.encode("ascii") + b"\n")
        deadline = time.monotonic() + (timeout or self.timeout)
        while time.monotonic() < deadline:
            line = self._readline(max(0.0, deadline - time.monotonic()))
            if line is None:
                break
            if line.startswith("OK"):
                return line
            if line.startswith("ERR"):
                if check:
                    raise RuntimeError(f"device rejected '{cmd}': {line}")
                return line
            if line.startswith("D,") and on_data:
                on_data(line)
        raise TimeoutError(f"no answer to '{cmd}' from {self.port}")

    # ---- Device interface -----------------------------------------------------------
    def _get(self) -> dict[str, float]:
        self._open()
        return P.parse_kv(self._command("GET")[2:])

    def _gains(self) -> dict[str, float]:
        g = self._get()
        return {k: g[k] for k in ("kp", "ki", "kd")}

    @_exclusive
    def read_params(self) -> dict[str, float]:
        g = self._get()
        return {"kp": g["kp"], "ki": g["ki"], "kd": g["kd"], "u_min": g["umin"], "u_max": g["umax"]}

    @staticmethod
    def _units(g: dict[str, float]) -> str:
        if g["umin"] < 0:
            return ("setpoint and pv are normalized servo position (0 to 1); "
                    f"u is a velocity command in [{g['umin']:g}, {g['umax']:g}]")
        return ("setpoint and pv are the normalized sensor reading (0 to 1); "
                f"u is the actuator drive (LED duty) in [{g['umin']:g}, {g['umax']:g}]")

    @_exclusive
    def info(self) -> dict[str, Any]:
        g = self._get()
        return {
            "device_id": self.device_id,
            "type": type(self).__name__,
            "gains": {k: g[k] for k in ("kp", "ki", "kd")},
            "actuator_limits": {"u_min": g["umin"], "u_max": g["umax"]},
            "step_test": {
                "default_setpoint": DEFAULT_SETPOINT,
                "allowed_setpoint_range": [0.0, g["spmax"]],
                "default_duration_s": DEFAULT_DURATION,
                "max_duration_s": MAX_DURATION,
            },
            "writable": {k: list(v) for k, v in P.GAIN_LIMITS.items()},
            "units": self._units(g),
            "transport": {"protocol": "serial-ascii", "port": self.port, "baud": self.baud},
        }

    @_exclusive
    def apply(self, req: WriteRequest) -> dict[str, Any]:
        if not req.approved:
            self.audit.record(event="write_blocked", device=self.device_id, params=req.params)
            raise ApprovalRequired("write request was not approved")
        bad = set(req.params) - WRITABLE
        if bad:
            raise ValueError(f"not writable: {sorted(bad)}")
        for k, v in req.params.items():
            lo, hi = P.GAIN_LIMITS[k]
            if not (lo <= float(v) <= hi):
                raise ValueError(f"{k}={v} outside safe range [{lo}, {hi}]")
        before = self._gains()
        for k, v in req.params.items():
            self._command(f"SET {k} {float(v):.4f}")
        after = self._gains()
        for k, v in req.params.items():
            if abs(after[k] - float(v)) > 0.0015:
                self.audit.record(event="write_failed", device=self.device_id,
                                  wanted=req.params, read_back=after)
                raise RuntimeError(f"read-back mismatch for {k}: wrote {v}, device holds {after[k]}")
        self.audit.record(event="write_applied", device=self.device_id, before=before,
                          after=after, reason=req.reason, approved_by=req.approved_by)
        return {"device_id": self.device_id, "before": before, "after": after}

    @_exclusive
    def capture(self, setpoint: float = DEFAULT_SETPOINT, duration: float = DEFAULT_DURATION,
                **_: Any) -> pd.DataFrame:
        """Home the actuator, run a step test, return t, setpoint, pv, u on a uniform grid."""
        sp_max = self._get()["spmax"]
        if not (P.BASE_SP < float(setpoint) <= sp_max):
            raise ValueError(f"setpoint={setpoint} outside allowed range ({P.BASE_SP}, {sp_max}] for "
                             f"{self.device_id}; use the default {DEFAULT_SETPOINT} unless told otherwise")
        if not (0.0 < float(duration) <= MAX_DURATION):
            raise ValueError(f"duration={duration} outside allowed range (0, {MAX_DURATION}] s")

        rows: list[tuple[float, float, float, float]] = []
        bad = [0]

        def on_data(line: str) -> None:
            rec = P.parse_data(line)
            if rec is None:
                bad[0] += 1
            else:
                rows.append(rec)

        self._command("RUN 0")
        self._command(f"HOME {P.BASE_SP}", timeout=8.0)
        self._command(f"SP {P.BASE_SP}")
        self._buf = b""
        self._ser.reset_input_buffer()
        self._command("RUN 1")
        stepped, t_last = False, -1.0
        deadline = time.monotonic() + 3.0 * duration + 10.0
        try:
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError("step test did not finish in time")
                line = self._readline(0.5)
                if line is None:
                    raise TimeoutError("the device stopped sending data")
                if line.startswith("D,"):
                    on_data(line)
                if not rows or rows[-1][0] <= t_last:
                    continue
                t_last = rows[-1][0]
                if not stepped and t_last >= T_STEP:
                    self._command(f"SP {float(setpoint):.4f}", on_data=on_data)
                    stepped = True
                if t_last >= duration:
                    break
        finally:
            try:
                self._command("RUN 0")     # always leave the loop stopped
            except (TimeoutError, RuntimeError, serial.SerialException):
                pass

        if len(rows) < 10:
            raise RuntimeError("the device sent almost no data")
        if bad[0] > MAX_BAD_FRACTION * (len(rows) + bad[0]):
            raise RuntimeError(f"{bad[0]} of {len(rows) + bad[0]} data lines were damaged: "
                               "check the cable and baud rate")
        arr = np.array(sorted(set(rows)))
        if arr[0, 0] > 0.5:
            raise RuntimeError(f"first sample is at t={arr[0, 0]:.2f} s: the loop did not restart")
        if np.max(np.diff(arr[:, 0], prepend=0.0, append=duration)) > 1.0:
            raise RuntimeError("step test data has a gap longer than 1 s (device or link stalled)")
        grid = np.arange(0.0, duration + 1e-9, GRID_DT)
        hold = np.clip(np.searchsorted(arr[:, 0], grid, side="right") - 1, 0, len(arr) - 1)
        return pd.DataFrame({
            "t": grid,
            "setpoint": arr[hold, 1],
            "pv": np.interp(grid, arr[:, 0], arr[:, 2]),
            "u": arr[hold, 3],
        })
