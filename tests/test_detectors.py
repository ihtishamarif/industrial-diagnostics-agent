import numpy as np
import pytest

from diagagent.signals import detectors as det

T = np.arange(0, 20, 0.01)


def test_sustained_sine_is_detected_as_oscillation():
    x = 0.5 * np.sin(2 * np.pi * 0.5 * T)
    f = det.detect_oscillation(T, x, signal="err")
    assert f.detected
    assert f.severity.value == "critical"
    assert f.details["period_s"] == pytest.approx(2.0, rel=0.02)
    assert f.details["decay_ratio"] == pytest.approx(1.0, abs=0.05)
    assert all(e.signal == "err" for e in f.evidence)


def test_well_damped_signal_is_not_flagged():
    x = np.exp(-1.5 * T) * np.sin(2 * np.pi * 0.5 * T)
    f = det.detect_oscillation(T, x)
    assert not f.detected


def test_detectors_are_deterministic():
    rng = np.random.default_rng(1)
    x = np.sin(T) + rng.normal(0, 0.05, T.size)
    a = det.detect_oscillation(T, x).to_dict()
    b = det.detect_oscillation(T, x).to_dict()
    assert a == b


def test_step_response_metrics_first_order():
    sp = np.where(T >= 1, 1.0, 0.0)
    y = np.where(T >= 1, 1 - np.exp(-(T - 1) / 0.5), 0.0)
    f = det.step_response_metrics(T, sp, y)
    assert not f.detected
    assert f.details["overshoot_pct"] == pytest.approx(0.0, abs=0.01)
    # 10-90 % rise time of a first-order system is 2.197 * tau
    assert f.details["rise_time_s"] == pytest.approx(2.197 * 0.5, abs=0.02)
    # 2 % settling is about 3.9 * tau
    assert f.details["settling_time_s"] == pytest.approx(3.91 * 0.5, abs=0.03)


def test_step_response_flags_overshoot():
    sp = np.where(T >= 1, 1.0, 0.0)
    tt = np.clip(T - 1, 0, None)
    y = np.where(T >= 1, 1 - np.exp(-0.5 * tt) * np.cos(3 * tt), 0.0)
    f = det.step_response_metrics(T, sp, y, max_overshoot_pct=10)
    assert f.detected
    assert f.details["overshoot_pct"] > 10
    assert any(e.metric == "overshoot_pct" for e in f.evidence)


def test_saturation_fraction_and_longest_run():
    u = np.zeros_like(T)
    u[100:600] = 5.0
    f = det.detect_saturation(T, u, 0.0, 5.0, max_fraction=0.9)
    # zeros are at u_min too, so everything is at a limit
    assert f.details["fraction"] == pytest.approx(1.0)
    u2 = np.full_like(T, 2.0)
    u2[100:600] = 5.0
    f2 = det.detect_saturation(T, u2, 0.0, 5.0)
    assert f2.details["fraction"] == pytest.approx(0.25)
    assert f2.detected
    assert f2.details["longest_run_s"] == pytest.approx(4.99, abs=0.02)


def test_dropouts_find_gaps_and_nans():
    t = np.delete(T, np.s_[500:560])
    x = np.sin(t)
    x[10] = np.nan
    f = det.detect_dropouts(t, x)
    assert f.detected
    assert f.details["n_gaps"] == 1
    assert f.details["n_nan"] == 1
    assert f.details["max_gap_s"] == pytest.approx(0.61, abs=0.001)


def test_noise_estimate_close_to_truth():
    rng = np.random.default_rng(0)
    x = np.sin(0.3 * T) + rng.normal(0, 0.05, T.size)
    f = det.detect_noise(T, x, max_noise_std=0.01)
    assert f.detected
    assert f.details["noise_std"] == pytest.approx(0.05, rel=0.15)


def test_drift():
    f = det.detect_drift(T, 0.01 * T, max_change=0.05)
    assert f.detected
    assert f.details["slope"] == pytest.approx(0.01, rel=1e-6)
    assert not det.detect_drift(T, np.zeros_like(T)).detected


def test_spikes():
    x = np.sin(T)
    x[[300, 900]] += 3.0
    f = det.detect_spikes(T, x)
    assert f.details["n_spikes"] == 2
    assert f.details["spike_times"] == pytest.approx([3.0, 9.0])


def test_bad_input_raises():
    with pytest.raises(ValueError):
        det.detect_oscillation([0, 1], [0, 1, 2])
