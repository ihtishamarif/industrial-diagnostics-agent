"""A simulated PLC that serves a PID loop over Modbus TCP.

It stands in for a real controller: the agent talks to it with the same protocol
it would use on a factory floor, so the Modbus client code is exercised for real.
The loop math matches sim/pid_sim.py, stepped one sample at a time. Simulated
time can run faster than wall-clock time (time_scale) so a 30 s step test does
not take 30 s.

Run it on its own:

    diagagent-plc --port 5020 --time-scale 5
"""

from __future__ import annotations

import argparse
import asyncio
import socket
import threading
import time
from collections import deque
from copy import deepcopy

import numpy as np
from pymodbus.server import ModbusTcpServer
from pymodbus.simulator import DataType, SimData, SimDevice

from ..devices import modbus_regs as R
from ..devices.sim_device import SimulatedPIDDevice
from .pid_sim import LoopConfig

MAX_RUN_S = 600.0  # the 16-bit clock register holds up to 655 s


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Loop:
    """PID + first-order-plus-dead-time plant, advanced one sample per step()."""

    def __init__(self, cfg: LoopConfig, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.dt = cfg.dt
        n_delay = max(0, int(round(cfg.plant.dead_time / self.dt)))
        self.u_buf = deque([0.0] * (n_delay + 1), maxlen=n_delay + 1)
        self.a = float(np.exp(-self.dt / cfg.plant.tau))
        self.y_true = 0.0
        self.integ = 0.0
        self.d_state = 0.0
        self.y_prev = 0.0
        self.steps = 0
        self.pv = 0.0
        self.u = 0.0

    def step(self, sp: float, kp: float, ki: float, kd: float) -> None:
        c, dt = self.cfg, self.dt
        noise = self.rng.normal(0.0, c.faults.noise_std) if c.faults.noise_std > 0 else 0.0
        y = self.y_true + c.faults.sensor_bias + noise
        e = sp - y
        raw_d = -(y - self.y_prev) / dt if self.steps > 0 else 0.0
        tf = (kd / (kp * c.deriv_filter_n)) if (kd > 0 and kp > 0) else 0.0
        if tf > 0:
            self.d_state += dt / (tf + dt) * (raw_d - self.d_state)
        else:
            self.d_state = raw_d
        self.y_prev = y
        u_unsat = kp * e + ki * self.integ + kd * self.d_state
        u = min(max(u_unsat, c.u_min), c.u_max)
        if not ((u_unsat > c.u_max and e > 0) or (u_unsat < c.u_min and e < 0)):
            self.integ += e * dt
        self.pv, self.u = y, u
        self.u_buf.appendleft(u)
        self.y_true = self.a * self.y_true + (1 - self.a) * c.plant.gain * self.u_buf[-1]
        self.steps += 1

    @property
    def time_s(self) -> float:
        return self.steps * self.dt


class _Regs:
    """Read and write the PLC's own registers (the same ones the Modbus clients see)."""

    def __init__(self, core, unit: int):
        self._core, self._unit = core, unit

    async def get(self, addr: int) -> int:
        return (await self._core.async_getValues(self._unit, 3, addr, 1))[0]

    async def put(self, addr: int, value: int) -> None:
        await self._core.async_setValues(self._unit, 3, addr, [value])


class SimulatedPLC:
    def __init__(self, cfg: LoopConfig | None = None, host: str = "127.0.0.1",
                 port: int = 5020, time_scale: float = 5.0, tick_s: float = 0.01,
                 unit: int = 1, seed: int = 0):
        if cfg is None:
            cfg = SimulatedPIDDevice.oscillating_demo().cfg
        self.cfg = deepcopy(cfg)
        self.host, self.port, self.unit = host, port, unit
        self.time_scale, self.tick_s, self.seed = time_scale, tick_s, seed
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._error: BaseException | None = None

    # ---- lifecycle ----------------------------------------------------------------
    def start(self, timeout: float = 5.0) -> "SimulatedPLC":
        self._thread = threading.Thread(target=self._run, name="sim-plc", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._error:
                raise RuntimeError(f"PLC failed to start: {self._error}")
            try:
                with socket.create_connection((self.host, self.port), timeout=0.2):
                    return self
            except OSError:
                time.sleep(0.02)
        raise TimeoutError(f"PLC did not start listening on {self.host}:{self.port}")

    def stop(self) -> None:
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = self._loop = self._stop = None

    def __enter__(self) -> "SimulatedPLC":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- server thread ------------------------------------------------------------
    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as e:  # surfaced by start()
            self._error = e

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        device = SimDevice(self.unit, simdata=[
            SimData(0, count=R.N_REGS, values=0, datatype=DataType.REGISTERS)])
        server = ModbusTcpServer(device, address=(self.host, self.port))
        regs = _Regs(server.context, self.unit)
        await self._init_registers(regs)
        serving = asyncio.create_task(server.serve_forever())
        plc = asyncio.create_task(self._plc_task(regs))
        await self._stop.wait()
        plc.cancel()
        await server.shutdown()
        serving.cancel()
        await asyncio.gather(plc, serving, return_exceptions=True)

    async def _init_registers(self, regs: "_Regs") -> None:
        c = self.cfg
        await regs.put(R.R_KP, R.enc_unsigned(c.gains.kp))
        await regs.put(R.R_KI, R.enc_unsigned(c.gains.ki))
        await regs.put(R.R_KD, R.enc_unsigned(c.gains.kd))
        await regs.put(R.R_UMIN, R.enc_unsigned(c.u_min))
        await regs.put(R.R_UMAX, R.enc_unsigned(c.u_max))
        await regs.put(R.R_SPMAX,
                       R.enc_unsigned(SimulatedPIDDevice.HEADROOM * c.plant.gain * c.u_max))

    # ---- the PLC scan cycle -------------------------------------------------------
    async def _plc_task(self, regs: "_Regs") -> None:
        get, put = regs.get, regs.put
        steps_per_tick = max(1, round(self.time_scale * self.tick_s / self.cfg.dt))
        good = {k: await get(a) for k, a in R.GAIN_REGS.items()}
        loop: _Loop | None = None
        runs = 0
        while True:
            await asyncio.sleep(self.tick_s)
            # validate gains on every scan; revert anything outside the PLC's own limits
            rejected = False
            for k, addr in R.GAIN_REGS.items():
                lo, hi = R.GAIN_LIMITS[k]
                val = await get(addr)
                if lo <= R.dec_unsigned(val) <= hi:
                    good[k] = val
                else:
                    await put(addr, good[k])
                    rejected = True
            if rejected:  # the fault stays visible until the next run starts
                await put(R.R_STATUS, R.STATUS_GAIN_REJECTED)
            if await get(R.R_RUN) != 1:
                if loop is not None:  # just stopped: outputs and clock go back to zero
                    loop = None
                    for addr in (R.R_PV, R.R_U, R.R_TIME):
                        await put(addr, 0)
                continue
            if loop is None:  # run flag just went high: fresh start
                loop = _Loop(deepcopy(self.cfg), seed=self.seed + runs)
                runs += 1
                await put(R.R_STATUS, R.STATUS_OK)
            sp = R.dec_signed(await get(R.R_SP))
            kp, ki, kd = (R.dec_unsigned(good[k]) for k in ("kp", "ki", "kd"))
            for _ in range(steps_per_tick):
                loop.step(sp, kp, ki, kd)
            await put(R.R_PV, R.enc_signed(loop.pv))
            await put(R.R_U, R.enc_signed(loop.u))
            await put(R.R_TIME, min(65535, int(round(loop.time_s * R.TIME_TICKS_PER_S))))
            if loop.time_s >= MAX_RUN_S:
                await put(R.R_RUN, 0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="diagagent-plc", description="Simulated Modbus TCP PLC")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5020)
    p.add_argument("--time-scale", type=float, default=5.0,
                   help="simulated seconds per wall-clock second")
    a = p.parse_args(argv)
    plc = SimulatedPLC(host=a.host, port=a.port, time_scale=a.time_scale).start()
    print(f"Simulated PLC listening on {a.host}:{a.port} (unit 1, time x{a.time_scale}). Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        plc.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
