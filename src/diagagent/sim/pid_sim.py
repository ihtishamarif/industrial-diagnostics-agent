"""Discrete-time simulation of a PID loop around a first-order-plus-dead-time plant.

Plant:      G(s) = K * exp(-theta*s) / (tau*s + 1)
Controller: parallel PID, u = Kp*e + Ki*integral(e) + Kd*d(-y)/dt
            - derivative on measurement (no derivative kick on setpoint steps)
            - first-order derivative filter with time constant Kd/(Kp*N)
            - output clamping with conditional-integration anti-windup

The simulation is deterministic for a given seed. Faults (sensor noise, sample
dropouts, sensor bias) can be injected so the detectors have something to find.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class PlantParams:
    gain: float = 2.0
    tau: float = 2.0
    dead_time: float = 0.4


@dataclass
class Faults:
    noise_std: float = 0.0
    dropout_prob: float = 0.0  # probability that a logged sample is lost
    sensor_bias: float = 0.0


@dataclass
class LoopConfig:
    gains: PIDGains
    plant: PlantParams = field(default_factory=PlantParams)
    faults: Faults = field(default_factory=Faults)
    u_min: float = 0.0
    u_max: float = 2.0
    deriv_filter_n: float = 10.0
    dt: float = 0.01


def simulate_step(
    cfg: LoopConfig,
    setpoint: float = 1.0,
    duration: float = 30.0,
    t_step: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Run a setpoint step test. Returns columns t, setpoint, pv, u."""
    rng = np.random.default_rng(seed)
    dt = cfg.dt
    n = int(round(duration / dt)) + 1
    t = np.arange(n) * dt
    sp = np.where(t >= t_step, setpoint, 0.0)

    delay_n = max(0, int(round(cfg.plant.dead_time / dt)))
    u_buf = np.zeros(delay_n + 1)
    y_true = 0.0
    a = np.exp(-dt / cfg.plant.tau)

    g = cfg.gains
    integ = 0.0
    d_state = 0.0
    y_meas_prev = 0.0
    tf = (g.kd / (g.kp * cfg.deriv_filter_n)) if (g.kd > 0 and g.kp > 0) else 0.0

    pv = np.empty(n)
    u_log = np.empty(n)
    for i in range(n):
        y_meas = y_true + cfg.faults.sensor_bias + (
            rng.normal(0.0, cfg.faults.noise_std) if cfg.faults.noise_std > 0 else 0.0
        )
        e = sp[i] - y_meas

        # derivative on measurement, filtered
        raw_d = -(y_meas - y_meas_prev) / dt if i > 0 else 0.0
        if tf > 0:
            alpha = dt / (tf + dt)
            d_state = d_state + alpha * (raw_d - d_state)
        else:
            d_state = raw_d
        y_meas_prev = y_meas

        u_unsat = g.kp * e + g.ki * integ + g.kd * d_state
        u = min(max(u_unsat, cfg.u_min), cfg.u_max)
        # anti-windup: integrate only if not pushing further into the limit
        if not ((u_unsat > cfg.u_max and e > 0) or (u_unsat < cfg.u_min and e < 0)):
            integ += e * dt

        pv[i] = y_meas
        u_log[i] = u

        # plant update with dead time (FIFO of past inputs)
        u_buf = np.roll(u_buf, 1)
        u_buf[0] = u
        u_delayed = u_buf[-1]
        y_true = a * y_true + (1 - a) * cfg.plant.gain * u_delayed

    df = pd.DataFrame({"t": t, "setpoint": sp, "pv": pv, "u": u_log})
    if cfg.faults.dropout_prob > 0:
        keep = rng.random(n) >= cfg.faults.dropout_prob
        keep[0] = keep[-1] = True
        df = df[keep].reset_index(drop=True)
    return df
