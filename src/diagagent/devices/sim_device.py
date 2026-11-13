"""A simulated PID controller that behaves like a real device for the agent."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from ..sim.pid_sim import Faults, LoopConfig, PIDGains, PlantParams, simulate_step
from .base import ApprovalRequired, AuditLog, WriteRequest

WRITABLE = {"kp", "ki", "kd"}
LIMITS = {"kp": (0.0, 50.0), "ki": (0.0, 50.0), "kd": (0.0, 20.0)}


class SimulatedPIDDevice:
    def __init__(self, cfg: LoopConfig, device_id: str = "sim_pid_1",
                 audit: AuditLog | None = None):
        self.device_id = device_id
        self.cfg = deepcopy(cfg)
        self.audit = audit or AuditLog()
        self._captures = 0

    @classmethod
    def oscillating_demo(cls) -> "SimulatedPIDDevice":
        """The Phase 1 demo fault: gains set far too high for the dead time."""
        return cls(LoopConfig(gains=PIDGains(kp=4.0, ki=1.5, kd=0.0),
                              plant=PlantParams(gain=2.0, tau=2.0, dead_time=0.4),
                              faults=Faults(noise_std=0.002),
                              u_min=0.0, u_max=5.0))

    def read_params(self) -> dict[str, float]:
        return {**self.cfg.gains.as_dict(), "u_min": self.cfg.u_min, "u_max": self.cfg.u_max}

    def apply(self, req: WriteRequest) -> dict[str, Any]:
        if not req.approved:
            self.audit.record(event="write_blocked", device=self.device_id, params=req.params)
            raise ApprovalRequired("write request was not approved")
        bad = set(req.params) - WRITABLE
        if bad:
            raise ValueError(f"not writable: {sorted(bad)}")
        for k, v in req.params.items():
            lo, hi = LIMITS[k]
            if not (lo <= float(v) <= hi):
                raise ValueError(f"{k}={v} outside safe range [{lo}, {hi}]")
        before = self.cfg.gains.as_dict()
        for k, v in req.params.items():
            setattr(self.cfg.gains, k, float(v))
        after = self.cfg.gains.as_dict()
        self.audit.record(event="write_applied", device=self.device_id, before=before,
                          after=after, reason=req.reason, approved_by=req.approved_by)
        return {"device_id": self.device_id, "before": before, "after": after}

    def capture(self, setpoint: float = 1.0, duration: float = 30.0, **_: Any) -> pd.DataFrame:
        """Run a step test. Seed is fixed per capture index, so runs are repeatable."""
        df = simulate_step(self.cfg, setpoint=setpoint, duration=duration, seed=self._captures)
        self._captures += 1
        return df
