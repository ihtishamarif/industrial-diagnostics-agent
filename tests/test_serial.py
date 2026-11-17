"""Serial device and simulated firmware: a real pseudo-terminal, real pyserial."""

import os

import numpy as np
import pytest

pytest.importorskip("serial")
if not hasattr(os, "openpty"):
    pytest.skip("needs a POSIX pseudo-terminal", allow_module_level=True)

from diagagent.devices import serial_protocol as P  # noqa: E402
from diagagent.devices.base import ApprovalRequired, WriteRequest  # noqa: E402
from diagagent.devices.serial_device import SerialPIDDevice  # noqa: E402
from diagagent.devices.sim_device import LIMITS  # noqa: E402
from diagagent.sim.serial_fw import SimulatedFirmware  # noqa: E402

DURATION = 8.0


@pytest.fixture
def fw():
    with SimulatedFirmware(time_scale=10.0) as f:
        yield f


@pytest.fixture
def dev(fw):
    d = SerialPIDDevice(fw.port, reset_delay=0.0)
    yield d
    d.close()


def _approved(**params) -> WriteRequest:
    return WriteRequest("serial_pid_1", params, reason="test", approved=True, approved_by="tester")


def test_gain_limits_match_the_other_devices():
    assert P.GAIN_LIMITS == LIMITS


def test_data_line_parsing():
    assert P.parse_data("D,1500,0.600,0.5123,0.250") == (1.5, 0.6, 0.5123, 0.25)
    for bad in ("D,1500,0.6,0.5", "D,xx,0.6,0.5,0.1", "OK", "D,1,2,3,4,5", ""):
        assert P.parse_data(bad) is None


def test_info_has_the_shape_the_agent_expects(dev):
    info = dev.info()
    assert info["gains"] == {"kp": 3.0, "ki": 50.0, "kd": 0.0}
    assert info["actuator_limits"] == {"u_min": P.U_MIN, "u_max": P.U_MAX}
    assert info["step_test"]["allowed_setpoint_range"] == [0.0, P.SP_MAX]
    assert info["writable"] == {k: list(v) for k, v in LIMITS.items()}


def test_connecting_to_something_else_fails(tmp_path):
    with pytest.raises(ConnectionError):
        SerialPIDDevice(str(tmp_path / "no_such_port"), reset_delay=0.0).info()


def test_capture_shows_the_oscillation_fault(dev):
    df = dev.capture(duration=DURATION)
    assert list(df.columns) == ["t", "setpoint", "pv", "u"]
    assert df.t.iloc[0] == 0.0 and df.t.iloc[-1] == pytest.approx(DURATION)
    assert df.setpoint.iloc[0] == P.BASE_SP and df.setpoint.iloc[-1] == 0.6
    tail = df[df.t > 5]
    assert tail.pv.std() > 0.05          # still swinging
    assert df.u.abs().max() <= P.U_MAX + 1e-9


def test_unapproved_write_is_blocked(dev):
    with pytest.raises(ApprovalRequired):
        dev.apply(WriteRequest("serial_pid_1", {"kp": 1.0}, reason="x"))
    assert dev.read_params()["kp"] == 3.0
    assert dev.audit.entries[-1]["event"] == "write_blocked"


def test_out_of_range_write_is_refused_by_host_and_by_firmware(dev):
    with pytest.raises(ValueError):
        dev.apply(_approved(kp=500.0))
    with pytest.raises(RuntimeError, match="ERR range"):   # bypass the host check
        dev._command("SET kp 500")
    assert dev.read_params()["kp"] == 3.0


def test_retune_stops_the_oscillation(dev):
    before = dev.capture(duration=DURATION)
    out = dev.apply(_approved(kp=1.364, ki=0.8855, kd=0.1515))
    assert out["after"]["kp"] == pytest.approx(1.364, abs=1e-3)
    after = dev.capture(duration=DURATION)
    assert after[after.t > 5].pv.std() < 0.02
    assert abs(after[after.t > 6].pv.mean() - 0.6) < 0.03
    assert before[before.t > 5].pv.std() > 5 * after[after.t > 5].pv.std()


def test_second_capture_starts_from_the_home_position(dev):
    dev.capture(duration=4.0)
    again = dev.capture(duration=4.0)
    assert abs(again.pv.iloc[0] - P.BASE_SP) < 0.03
    assert again.t.iloc[0] == 0.0


def test_unreachable_setpoint_and_duration_are_refused(dev):
    with pytest.raises(ValueError):
        dev.capture(setpoint=5.0)
    with pytest.raises(ValueError):
        dev.capture(duration=1e6)


def test_damaged_data_is_refused_not_analysed(dev, monkeypatch):
    real = P.parse_data
    n = {"i": 0}

    def flaky(line):
        n["i"] += 1
        return None if n["i"] % 3 == 0 else real(line)   # every third line "arrives damaged"
    monkeypatch.setattr("diagagent.devices.serial_device.P.parse_data", flaky)
    with pytest.raises(RuntimeError, match="damaged"):
        dev.capture(duration=4.0)


def test_loop_is_left_stopped_after_a_failed_capture(dev, fw):
    with pytest.raises(ValueError):
        dev.capture(setpoint=5.0)
    dev.capture(duration=3.0)
    assert fw._loop is None


# ---- LED + LDR plant: output is the LED duty in [0, 1], the plant has no integrator ----------

@pytest.fixture
def led_dev():
    with SimulatedFirmware(time_scale=10.0, plant="led") as f:
        d = SerialPIDDevice(f.port, reset_delay=0.0)
        yield d
        d.close()


def _overshoot(df, target=0.6):
    return (df[df.t > 1].pv.max() - target) / (target - P.BASE_SP)


def test_unknown_plant_is_rejected():
    with pytest.raises(ValueError):
        SimulatedFirmware(plant="stepper")


def test_led_plant_reports_its_own_limits_and_units(led_dev):
    info = led_dev.info()
    assert info["actuator_limits"] == {"u_min": P.LED_U_MIN, "u_max": P.LED_U_MAX}
    assert "LED duty" in info["units"] and "velocity" not in info["units"]
    assert info["gains"] == {"kp": 10.0, "ki": 20.0, "kd": 0.0}


def test_servo_plant_units_still_say_velocity(dev):
    assert "velocity command" in dev.info()["units"]


def test_led_capture_stays_inside_duty_limits_and_overshoots(led_dev):
    df = led_dev.capture(duration=DURATION)
    assert df.u.min() >= P.LED_U_MIN - 1e-9 and df.u.max() <= P.LED_U_MAX + 1e-9
    assert _overshoot(df) > 0.15
    assert abs(df[df.t > 6].pv.mean() - 0.6) < 0.03      # integral action still reaches the setpoint


def test_led_retune_cuts_the_overshoot(led_dev):
    before = led_dev.capture(duration=DURATION)
    led_dev.apply(_approved(kp=6.0, ki=12.0, kd=0.0))
    after = led_dev.capture(duration=DURATION)
    assert _overshoot(after) < 0.5 * _overshoot(before)
    assert abs(after[after.t > 7].pv.mean() - 0.6) < 0.03


# ---- the agent runs tool calls of one step in parallel threads: the port must be shared safely ----

def test_parallel_calls_from_several_threads_do_not_corrupt_the_port(dev):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as ex:
        cap = ex.submit(dev.capture, duration=3.0)
        infos = [ex.submit(dev.info) for _ in range(3)]
        reads = [ex.submit(dev.read_params) for _ in range(3)]
        for f in infos + reads:
            f.result(timeout=30)          # raises if the replies were interleaved or damaged
        df = cap.result(timeout=30)
    assert len(df) > 10 and df.t.iloc[-1] == pytest.approx(3.0)
