"""Control-loop analysis built from the deterministic detectors.

analyze_control_loop() runs the relevant detectors on a closed-loop log and maps
the findings to ranked hypotheses with a fixed rule table. suggest_pid_retune()
proposes new gains from the measured oscillation using textbook tuning rules.
Nothing here calls an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import detectors as det
from .types import Finding


@dataclass(frozen=True)
class Hypothesis:
    cause: str
    confidence: str  # "high" | "medium" | "low"
    based_on: tuple[str, ...]  # detector names
    suggested_fix: str


@dataclass
class LoopReport:
    findings: list[Finding]
    hypotheses: list[Hypothesis] = field(default_factory=list)

    def get(self, detector: str) -> Finding | None:
        return next((f for f in self.findings if f.detector == detector), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "anomalies": [f.summary for f in self.findings if f.detected],
            "hypotheses": [h.__dict__ | {"based_on": list(h.based_on)} for h in self.hypotheses],
        }


def analyze_control_loop(
    df: pd.DataFrame,
    t_col: str = "t",
    sp_col: str = "setpoint",
    pv_col: str = "pv",
    u_col: str | None = "u",
    u_min: float | None = None,
    u_max: float | None = None,
    noise_limit: float = 0.01,
    max_overshoot_pct: float = 10.0,
) -> LoopReport:
    t = df[t_col].to_numpy(float)
    sp = df[sp_col].to_numpy(float)
    pv = df[pv_col].to_numpy(float)
    err = sp - pv
    span = float(np.ptp(sp)) or 1.0

    findings: list[Finding] = [
        det.detect_dropouts(t, pv, signal=pv_col),
        det.step_response_metrics(t, sp, pv, signal=pv_col, max_overshoot_pct=max_overshoot_pct),
        det.detect_oscillation(t, err, signal=f"error({sp_col}-{pv_col})",
                               min_amplitude=0.02 * span),
        det.detect_noise(t, pv, signal=pv_col, max_noise_std=noise_limit),
    ]
    if u_col and u_col in df and u_min is not None and u_max is not None:
        findings.append(det.detect_saturation(t, df[u_col].to_numpy(float), u_min, u_max, signal=u_col))
    if u_col and u_col in df:
        findings.append(det.detect_noise(t, df[u_col].to_numpy(float), signal=u_col,
                                         max_noise_std=noise_limit * 10))

    report = LoopReport(findings)
    report.hypotheses = _rank_hypotheses(report, u_col)
    return report


def _rank_hypotheses(r: LoopReport, u_col: str | None) -> list[Hypothesis]:
    """Fixed rule table. Order of rules is the order of confidence."""
    osc = r.get("oscillation")
    step = r.get("step_response")
    sat = r.get("saturation")
    drop = r.get("dropouts")
    noise_pv = next((f for f in r.findings if f.detector == "noise" and f.evidence
                     and f.evidence[0].signal != u_col), None)
    noise_u = next((f for f in r.findings if f.detector == "noise" and f.evidence
                    and f.evidence[0].signal == u_col), None)

    hyps: list[Hypothesis] = []
    osc_on = bool(osc and osc.detected)
    sat_heavy = bool(sat and sat.detected)

    if osc_on and not sat_heavy:
        hyps.append(Hypothesis(
            "Controller too aggressive (proportional/integral gain too high for the plant dead time)",
            "high", ("oscillation",),
            "Reduce Kp and Ki; retune from the measured oscillation period."))
    if osc_on and sat_heavy:
        hyps.append(Hypothesis(
            "Limit cycle driven by actuator saturation and integrator windup",
            "high", ("oscillation", "saturation"),
            "Check anti-windup, reduce Ki, confirm actuator range."))
    if step and step.detected and not osc_on and step.details.get("overshoot_pct", 0) > 0:
        hyps.append(Hypothesis(
            "Underdamped tuning (overshoot without sustained oscillation)",
            "medium", ("step_response",),
            "Reduce Kp by about 30 % or add a small derivative term."))
    sse_big = bool(step and step.details.get("target") is not None and abs(
        step.details["steady_state_error"]) > 0.02 * abs(step.details["target"] - step.details["y0"]))
    if sse_big and sat_heavy and not osc_on:
        hyps.append(Hypothesis(
            "Actuator at its limit cannot reach the setpoint",
            "medium", ("step_response", "saturation"),
            "Check actuator sizing or lower the setpoint."))
    if noise_pv and noise_pv.detected:
        conf = "medium" if (noise_u and noise_u.detected) else "low"
        hyps.append(Hypothesis(
            "Sensor noise is entering the controller output",
            conf, ("noise",),
            "Filter the measurement, lower Kd or increase the derivative filter."))
    if drop and drop.detected:
        hyps.append(Hypothesis(
            "Sample dropouts or communication gaps in the measurement",
            "medium", ("dropouts",),
            "Check sensor wiring, bus load and logging rate."))
    return hyps


# --------------------------------------------------------------------------- #
# Retuning
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RetuneProposal:
    kp: float
    ki: float
    kd: float
    method: str
    rationale: str
    evidence: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"kp": round(self.kp, 4), "ki": round(self.ki, 4), "kd": round(self.kd, 4),
                "method": self.method, "rationale": self.rationale,
                "evidence": list(self.evidence)}


def suggest_pid_retune(report: LoopReport, kp: float, ki: float, kd: float = 0.0,
                       use_derivative: bool = True) -> RetuneProposal | None:
    """Return new gains, or None if the loop needs no retune.

    Sustained oscillation (decay ratio >= 0.9): treat the current Kp as an
    estimate of the ultimate gain Ku and the oscillation period as Pu, then
    apply Tyreus-Luyben rules. Because integral action adds phase lag, the
    loop oscillates at a Kp below the true Ku, so this estimate is on the
    conservative side.

    Weakly damped oscillation or large overshoot: scale Kp and Ki down by 40 %.
    """
    osc = report.get("oscillation")
    step = report.get("step_response")

    if osc and osc.detected and osc.details["decay_ratio"] >= 0.9:
        ku, pu = kp, osc.details["period_s"]
        if use_derivative:
            new_kp = ku / 2.2
            ti, td = 2.2 * pu, pu / 6.3
            new_ki, new_kd = new_kp / ti, new_kp * td
            method = "Tyreus-Luyben PID"
        else:
            new_kp = ku / 3.2
            ti = 2.2 * pu
            new_ki, new_kd = new_kp / ti, 0.0
            method = "Tyreus-Luyben PI"
        return RetuneProposal(
            new_kp, new_ki, new_kd, method,
            f"Sustained oscillation: using Ku ~= {ku:g} (current Kp) and Pu = {pu:.3f}s.",
            tuple(e.cite() for e in osc.evidence))

    if (osc and osc.detected) or (step and step.detected and step.details.get("overshoot_pct", 0) > 0):
        src = osc if (osc and osc.detected) else step
        return RetuneProposal(
            kp * 0.6, ki * 0.6, kd, "gain reduction (x0.6)",
            "Underdamped response: reduce proportional and integral gain by 40 %.",
            tuple(e.cite() for e in src.evidence))
    return None
