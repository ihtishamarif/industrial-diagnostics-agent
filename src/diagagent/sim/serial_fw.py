"""A stand-in for the Arduino firmware, served over a pseudo-terminal.

It speaks the protocol in devices/serial_protocol.py, so the host code is exercised
over a real serial device file (opened with pyserial like /dev/ttyACM0). The plant is
an emulation of the hardware setup: a hobby servo whose position is read back by a
potentiometer on its horn. The controller output is a velocity command, the servo
follows with a first-order lag and a speed limit, and the pot reading has noise and
10-bit quantization. Linux and macOS only (it needs os.openpty).

    diagagent-fw --time-scale 5      # prints the port name to give to --serial-port
"""

from __future__ import annotations

import argparse
import math
import os
import select
import threading
import time
import tty

import numpy as np

from ..devices import serial_protocol as P

SERVO_TAU = 0.08        # s, lag between commanded and actual servo position
SERVO_MAX_RATE = 3.0    # position units per second (a fast SG90)
ADC_STEPS = 1023
POT_NOISE = 0.003


class _FwLoop:
    """What the microcontroller does every sample: read pot, run PID, move servo."""

    def __init__(self, gains: dict[str, float], start_pos: float, seed: int):
        self.rng = np.random.default_rng(seed)
        self.g = gains
        self.theta_cmd = self.theta = start_pos
        self.integ = self.d_state = 0.0
        self.y_prev: float | None = None
        self.steps = 0
        self.a = math.exp(-P.SAMPLE_DT / SERVO_TAU)

    def step(self, sp: float) -> tuple[float, float]:
        dt = P.SAMPLE_DT
        y = self.theta + self.rng.normal(0.0, POT_NOISE)
        y = round(min(max(y, 0.0), 1.0) * ADC_STEPS) / ADC_STEPS
        e = sp - y
        kp, ki, kd = self.g["kp"], self.g["ki"], self.g["kd"]
        raw_d = -(y - self.y_prev) / dt if self.y_prev is not None else 0.0
        tf = kd / (kp * 10.0) if (kd > 0 and kp > 0) else 0.0
        self.d_state = self.d_state + dt / (tf + dt) * (raw_d - self.d_state) if tf > 0 else raw_d
        self.y_prev = y
        u_unsat = kp * e + ki * self.integ + kd * self.d_state
        u = min(max(u_unsat, P.U_MIN), P.U_MAX)
        if not ((u_unsat > P.U_MAX and e > 0) or (u_unsat < P.U_MIN and e < 0)):
            self.integ += e * dt
        self.theta_cmd = min(max(self.theta_cmd + u * P.VMAX * dt, 0.0), 1.0)
        target = self.theta + (self.theta_cmd - self.theta) * (1 - self.a)
        self.theta += min(max(target - self.theta, -SERVO_MAX_RATE * dt), SERVO_MAX_RATE * dt)
        self.steps += 1
        return y, u

    @property
    def ms(self) -> int:
        return int(round(self.steps * P.SAMPLE_DT * 1000))


class SimulatedFirmware:
    def __init__(self, gains: dict[str, float] | None = None, time_scale: float = 5.0,
                 seed: int = 0):
        self.gains = dict(gains or {"kp": 3.0, "ki": 50.0, "kd": 0.0})
        self.time_scale, self.seed = time_scale, seed
        self.sp = P.BASE_SP
        self.pos = 0.0           # servo position when no loop runs
        self.port: str = ""
        self._master = self._slave = -1
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._buf = b""
        self._loop: _FwLoop | None = None
        self._t0 = 0.0
        self._runs = 0

    def start(self) -> "SimulatedFirmware":
        self._master, self._slave = os.openpty()
        tty.setraw(self._slave)                  # no echo, no newline translation
        os.set_blocking(self._master, False)
        self.port = os.ttyname(self._slave)
        self._thread = threading.Thread(target=self._run, name="sim-fw", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        for fd in (self._master, self._slave):
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> "SimulatedFirmware":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- firmware main loop -------------------------------------------------------
    def _send(self, text: str) -> None:
        try:
            os.write(self._master, text.encode("ascii") + b"\n")
        except (BlockingIOError, OSError):
            pass                                  # nobody is listening: drop it

    def _run(self) -> None:
        while not self._stop.is_set():
            ready, _, _ = select.select([self._master], [], [], 0.005)
            if ready:
                try:
                    self._buf += os.read(self._master, 4096)
                except (BlockingIOError, OSError):
                    pass
                while b"\n" in self._buf:
                    line, self._buf = self._buf.split(b"\n", 1)
                    self._handle(line.decode("ascii", errors="replace").strip())
            if self._loop is not None:
                target = (time.monotonic() - self._t0) * self.time_scale
                while self._loop.steps * P.SAMPLE_DT + P.SAMPLE_DT <= target:
                    pv, u = self._loop.step(self.sp)
                    self._send(f"D,{self._loop.ms},{self.sp:.3f},{pv:.4f},{u:.3f}")
                    if self._loop.steps * P.SAMPLE_DT >= 600.0:
                        self.pos, self._loop = self._loop.theta, None
                        break

    def _handle(self, cmd: str) -> None:
        w = cmd.split()
        if not w:
            return
        if w[0] == "PING":
            self._send(P.HELLO)
        elif w[0] == "GET":
            g = self.gains
            self._send(f"OK kp={g['kp']:.3f} ki={g['ki']:.3f} kd={g['kd']:.3f} "
                       f"umin={P.U_MIN:.3f} umax={P.U_MAX:.3f} spmax={P.SP_MAX:.3f}")
        elif w[0] == "SET" and len(w) == 3 and w[1] in P.GAIN_LIMITS:
            try:
                v = float(w[2])
            except ValueError:
                return self._send("ERR value")
            lo, hi = P.GAIN_LIMITS[w[1]]
            if not (lo <= v <= hi):
                return self._send("ERR range")   # the firmware protects itself
            self.gains[w[1]] = v
            if self._loop:
                self._loop.g = self.gains
            self._send(f"OK {w[1]}={v:.3f}")
        elif w[0] == "SP" and len(w) == 2:
            try:
                v = float(w[1])
            except ValueError:
                return self._send("ERR value")
            if not (0.0 <= v <= P.SP_MAX):
                return self._send("ERR range")
            self.sp = v
            self._send("OK")
        elif w[0] == "HOME" and len(w) == 2:
            try:
                v = float(w[1])
            except ValueError:
                return self._send("ERR value")
            if self._loop is not None or not (0.0 <= v <= 1.0):
                return self._send("ERR state" if self._loop else "ERR range")
            self.pos = v
            self._send("OK")
        elif w[0] == "RUN" and len(w) == 2 and w[1] in ("0", "1"):
            if w[1] == "1" and self._loop is None:
                self._loop = _FwLoop(self.gains, self.pos, self.seed + self._runs)
                self._runs += 1
                self._t0 = time.monotonic()
            elif w[1] == "0" and self._loop is not None:
                self.pos, self._loop = self._loop.theta, None
            self._send("OK")
        else:
            self._send("ERR unknown")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="diagagent-fw", description="Simulated Arduino PID firmware")
    p.add_argument("--time-scale", type=float, default=1.0)
    a = p.parse_args(argv)
    fw = SimulatedFirmware(time_scale=a.time_scale).start()
    print(f"Simulated firmware on {fw.port} (time x{a.time_scale}). Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        fw.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
