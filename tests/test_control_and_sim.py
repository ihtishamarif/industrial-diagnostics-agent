import pytest

from diagagent.devices.base import ApprovalRequired, WriteRequest
from diagagent.devices.sim_device import SimulatedPIDDevice
from diagagent.signals.control import analyze_control_loop, suggest_pid_retune
from diagagent.sim.pid_sim import Faults, LoopConfig, PIDGains, simulate_step


def test_simulation_is_deterministic():
    cfg = LoopConfig(PIDGains(1.2, 0.6), faults=Faults(noise_std=0.01))
    a = simulate_step(cfg, seed=3)
    b = simulate_step(cfg, seed=3)
    assert a.equals(b)


def test_good_gains_settle_without_anomalies():
    df = simulate_step(LoopConfig(PIDGains(1.2, 0.6), u_max=5.0))
    r = analyze_control_loop(df, u_min=0, u_max=5)
    assert not r.get("oscillation").detected
    assert r.get("step_response").details["settled"]
    assert r.hypotheses == []


def test_oscillating_demo_is_diagnosed_and_retune_fixes_it():
    dev = SimulatedPIDDevice.oscillating_demo()
    before = analyze_control_loop(dev.capture(), u_min=0, u_max=5)
    osc = before.get("oscillation")
    assert osc.detected and osc.details["decay_ratio"] >= 0.9
    assert before.hypotheses[0].confidence == "high"
    assert "too aggressive" in before.hypotheses[0].cause

    p = dev.read_params()
    prop = suggest_pid_retune(before, p["kp"], p["ki"], p["kd"])
    assert prop is not None and prop.method.startswith("Tyreus-Luyben")
    assert prop.kp < p["kp"]
    assert prop.evidence  # proposal cites evidence

    dev.apply(WriteRequest(dev.device_id, {"kp": prop.kp, "ki": prop.ki, "kd": prop.kd},
                           "test", approved=True, approved_by="pytest"))
    after = analyze_control_loop(dev.capture(), u_min=0, u_max=5)
    assert not after.get("oscillation").detected
    assert after.get("step_response").details["overshoot_pct"] < 10
    assert after.get("step_response").details["settled"]


def test_device_refuses_unapproved_write():
    dev = SimulatedPIDDevice.oscillating_demo()
    with pytest.raises(ApprovalRequired):
        dev.apply(WriteRequest(dev.device_id, {"kp": 1.0}, "no approval"))
    assert dev.read_params()["kp"] == 4.0
    assert dev.audit.entries[-1]["event"] == "write_blocked"


def test_device_rejects_out_of_range_and_unknown_params():
    dev = SimulatedPIDDevice.oscillating_demo()
    with pytest.raises(ValueError):
        dev.apply(WriteRequest(dev.device_id, {"kp": 1000.0}, "x", approved=True))
    with pytest.raises(ValueError):
        dev.apply(WriteRequest(dev.device_id, {"u_max": 10.0}, "x", approved=True))
    assert dev.read_params()["kp"] == 4.0


def test_noise_and_dropout_hypotheses():
    cfg = LoopConfig(PIDGains(1.2, 0.6, 0.3), faults=Faults(noise_std=0.05), u_max=5.0)
    df = simulate_step(cfg, seed=1)
    df = df.drop(index=range(1500, 1600)).reset_index(drop=True)  # 1 s communication gap
    r = analyze_control_loop(df, u_min=0, u_max=5)
    causes = " ".join(h.cause for h in r.hypotheses)
    assert "noise" in causes.lower()
    assert "dropouts" in causes.lower()
