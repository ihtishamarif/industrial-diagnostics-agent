"""Modbus device and simulated PLC: real TCP, real register traffic."""

import numpy as np
import pytest

pytest.importorskip("pymodbus")

from diagagent.devices import modbus_regs as R  # noqa: E402
from diagagent.devices.base import ApprovalRequired, WriteRequest  # noqa: E402
from diagagent.devices.modbus_device import ModbusPIDDevice  # noqa: E402
from diagagent.devices.sim_device import LIMITS, SimulatedPIDDevice  # noqa: E402
from diagagent.sim.modbus_plc import SimulatedPLC, free_port  # noqa: E402

DURATION = 15.0  # seconds of simulated time; the test PLC runs 10x faster than real time


@pytest.fixture
def dev():
    port = free_port()
    with SimulatedPLC(port=port, time_scale=10.0):
        d = ModbusPIDDevice(port=port)
        yield d
        d.close()


def _approved(**params) -> WriteRequest:
    return WriteRequest("modbus_pid_1", params, reason="test", approved=True, approved_by="tester")


def _overshoot_pct(df, sp=1.0) -> float:
    return float((df.pv.max() - sp) / sp * 100)


def test_register_codecs_roundtrip():
    for x in (0.0, 1.234, -2.5, 31.999, -32.0):
        assert R.dec_signed(R.enc_signed(x)) == pytest.approx(x, abs=1e-3)
    assert R.dec_unsigned(R.enc_unsigned(49.5)) == pytest.approx(49.5)


def test_plc_limits_match_simulated_device():
    assert R.GAIN_LIMITS == LIMITS


def test_info_has_the_same_shape_as_the_simulated_device(dev):
    sim = SimulatedPIDDevice.oscillating_demo().info()
    info = dev.info()
    assert set(sim) <= set(info)
    assert info["gains"] == sim["gains"]
    assert info["actuator_limits"] == sim["actuator_limits"]
    assert info["step_test"] == sim["step_test"]
    assert info["writable"] == sim["writable"]


def test_read_params_matches_simulated_device(dev):
    assert dev.read_params() == SimulatedPIDDevice.oscillating_demo().read_params()


def test_capture_shows_the_oscillation_fault(dev):
    df = dev.capture(setpoint=1.0, duration=DURATION)
    assert list(df.columns) == ["t", "setpoint", "pv", "u"]
    assert np.allclose(np.diff(df.t), 0.05)              # uniform grid
    assert df.setpoint.iloc[0] == 0.0 and df.setpoint.iloc[-1] == 1.0
    assert _overshoot_pct(df) > 80                        # Kp=4 is far too high
    assert df.u.max() <= 5.0 and df.u.min() >= 0.0


def test_loop_is_stopped_after_a_capture(dev):
    dev.capture(setpoint=1.0, duration=5.0)
    assert dev._read(R.R_RUN)[0] == 0


def test_write_blocked_without_approval(dev):
    before = dev.read_params()
    with pytest.raises(ApprovalRequired):
        dev.apply(WriteRequest("modbus_pid_1", {"kp": 1.0}, reason="x"))
    assert dev.read_params() == before
    assert dev.audit.entries[-1]["event"] == "write_blocked"


def _metrics(df, sp=1.0) -> dict:
    """Overshoot, time of the last sample outside the 2% band, and final-value noise."""
    pv, t = df.pv.to_numpy(), df.t.to_numpy()
    outside = np.where(np.abs(pv - sp) > 0.02 * sp)[0]
    return {"overshoot": float((pv.max() - sp) / sp * 100),
            "last_outside_t": float(t[outside.max()]) if len(outside) else 0.0,
            "final_mean": float(pv[-40:].mean()), "final_std": float(pv[-100:].std())}


RETUNE = {"kp": 1.8182, "ki": 0.5165, "kd": 0.4618}


def test_approved_write_changes_gains(dev):
    out = dev.apply(_approved(**RETUNE))
    assert out["after"]["kp"] == pytest.approx(1.8182, abs=1e-3)
    assert dev.read_params()["ki"] == pytest.approx(0.5165, abs=1e-3)
    assert dev.audit.entries[-1]["approved_by"] == "tester"


def test_retune_settles_like_the_in_process_simulator(dev):
    """Second capture after a first one: the case that once returned stale registers."""
    dev.capture(setpoint=1.0, duration=DURATION)                 # the faulty "before" run
    dev.apply(_approved(**RETUNE))
    after = _metrics(dev.capture(setpoint=1.0, duration=DURATION))

    sim = SimulatedPIDDevice.oscillating_demo()
    sim.capture(1.0, DURATION)
    sim.apply(_approved(**RETUNE))
    ref = _metrics(sim.capture(1.0, DURATION))

    assert after["overshoot"] < 3.0 and abs(after["overshoot"] - ref["overshoot"]) < 2.0
    assert after["last_outside_t"] < 9.0 and abs(after["last_outside_t"] - ref["last_outside_t"]) < 2.0
    assert abs(after["final_mean"] - 1.0) < 0.01
    assert after["final_std"] > 0                                # live data, not a frozen register


def test_every_capture_is_a_fresh_run(dev):
    first = dev.capture(setpoint=1.0, duration=5.0)
    second = dev.capture(setpoint=1.0, duration=5.0)
    for df in (first, second):
        assert df.t.iloc[0] == 0.0 and abs(df.pv.iloc[0]) < 0.01
        assert df.pv.nunique() > 50                              # the signal really moves
    assert np.allclose(first.pv, second.pv, atol=0.2)            # same loop, same behaviour


def test_stale_registers_are_detected(dev):
    """If the PLC ever keeps an old clock value, the client must refuse the data."""
    dev._write(R.R_TIME, 3000)                                   # pretend a finished run is left over
    dev._wait_until_stopped = lambda timeout=2.0: None           # skip the handshake on purpose
    with pytest.raises(RuntimeError, match="stale|gap"):
        dev.capture(setpoint=1.0, duration=5.0)


def test_client_refuses_out_of_range_write(dev):
    before = dev.read_params()
    with pytest.raises(ValueError, match="outside safe range"):
        dev.apply(_approved(kp=60.0))
    with pytest.raises(ValueError, match="not writable"):
        dev.apply(_approved(u_max=9.0))
    assert dev.read_params() == before


def test_plc_rejects_out_of_range_even_if_client_check_is_bypassed(dev):
    import time
    before = dev.read_params()
    dev._write(R.R_KP, R.enc_unsigned(60.0))              # raw write, no client-side check
    time.sleep(0.1)
    assert dev.read_params() == before                    # the PLC reverted it
    assert dev._read(R.R_STATUS)[0] == R.STATUS_GAIN_REJECTED


def test_capture_validates_arguments(dev):
    with pytest.raises(ValueError, match="setpoint"):
        dev.capture(setpoint=100.0, duration=10.0)
    with pytest.raises(ValueError, match="duration"):
        dev.capture(setpoint=1.0, duration=500.0)


def test_unreachable_device_raises_connection_error():
    d = ModbusPIDDevice(port=free_port(), timeout=0.3)
    with pytest.raises(ConnectionError):
        d.info()


def test_works_through_the_agent_tool_layer_pattern(dev):
    """The agent only uses Device methods, so a Modbus device must satisfy them all."""
    for name in ("read_params", "info", "apply", "capture"):
        assert callable(getattr(dev, name))
    assert dev.device_id == "modbus_pid_1"
