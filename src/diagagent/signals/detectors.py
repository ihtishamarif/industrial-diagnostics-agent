"""Deterministic signal detectors.

Rules for this module:
- Plain numpy/scipy. No LLM, no randomness, no global state.
- Same input gives the same Finding every time.
- Every positive Finding carries Evidence (signal, window, metric, value, threshold).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .types import Evidence, Finding, Severity


def _as_arrays(t, x) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.shape != x.shape or t.ndim != 1:
        raise ValueError("t and x must be 1-D arrays of equal length")
    if len(t) < 3:
        raise ValueError("need at least 3 samples")
    mask = np.isfinite(t) & np.isfinite(x)
    return t[mask], x[mask]


# --------------------------------------------------------------------------- #
# Oscillation
# --------------------------------------------------------------------------- #
def detect_oscillation(
    t,
    x,
    signal: str = "x",
    min_cycles: int = 3,
    min_amplitude: float = 0.02,
    max_decay_ratio: float = 0.5,
) -> Finding:
    """Detect sustained or weakly damped oscillation.

    The signal is centred on its median. Peaks and troughs are paired into half
    amplitudes; the decay ratio is the median ratio of successive amplitudes.
    A well damped loop has a decay ratio near 0.25 or lower; 1.0 means a
    sustained oscillation; above 1.0 means it is growing.
    """
    t, x = _as_arrays(t, x)
    xc = x - np.median(x)
    prom = 2 * min_amplitude
    peaks, _ = find_peaks(xc, prominence=prom)
    troughs, _ = find_peaks(-xc, prominence=prom)

    # Pair each peak with the next trough to get peak-to-peak half amplitudes.
    amps, amp_times = [], []
    for p in peaks:
        nxt = troughs[troughs > p]
        if len(nxt) == 0:
            break
        q = nxt[0]
        amps.append((xc[p] - xc[q]) / 2.0)
        amp_times.append(t[p])
    amps = np.asarray(amps)
    n_cycles = len(amps)

    if n_cycles < 2:
        return Finding(
            "oscillation", False, Severity.INFO,
            f"No oscillation in {signal}: fewer than 2 full cycles above amplitude {min_amplitude:g}.",
            details={"cycles": int(n_cycles)},
        )

    periods = np.diff(t[peaks])
    period = float(np.median(periods)) if len(periods) else float("nan")
    period_cv = float(np.std(periods) / np.mean(periods)) if len(periods) > 1 else 0.0
    ratios = amps[1:] / amps[:-1]
    decay = float(np.median(ratios))
    last_amp = float(amps[-1])

    detected = (
        n_cycles >= min_cycles
        and decay > max_decay_ratio
        and last_amp > min_amplitude
        and period_cv < 0.35
    )
    severity = Severity.INFO
    if detected:
        severity = Severity.CRITICAL if decay >= 0.9 else Severity.WARNING

    t0, t1 = float(t[peaks[0]]), float(t[peaks[min(n_cycles, len(peaks)) - 1]])
    ev = (
        Evidence(signal, t0, t1, "decay_ratio", decay, max_decay_ratio),
        Evidence(signal, t0, t1, "period", period, None, "s"),
        Evidence(signal, t0, t1, "last_amplitude", last_amp, min_amplitude),
        Evidence(signal, t0, t1, "cycles", float(n_cycles), float(min_cycles)),
    )
    if detected:
        kind = "growing" if decay > 1.05 else ("sustained" if decay >= 0.9 else "weakly damped")
        summary = (
            f"{kind.capitalize()} oscillation in {signal}: {n_cycles} cycles, "
            f"period {period:.3f}s, decay ratio {decay:.2f}."
        )
    else:
        summary = f"No problematic oscillation in {signal} (decay ratio {decay:.2f}, {n_cycles} cycles)."
    return Finding(
        "oscillation", detected, severity, summary, ev,
        {"cycles": int(n_cycles), "period_s": period, "period_cv": period_cv,
         "decay_ratio": decay, "last_amplitude": last_amp,
         "frequency_hz": (1.0 / period) if period and np.isfinite(period) else None},
    )


# --------------------------------------------------------------------------- #
# Step response
# --------------------------------------------------------------------------- #
def step_response_metrics(
    t,
    setpoint,
    pv,
    signal: str = "pv",
    settle_band: float = 0.02,
    max_overshoot_pct: float = 10.0,
) -> Finding:
    """Overshoot, rise time (10 to 90 %), settling time and steady-state error.

    The step is taken at the largest single change in the setpoint.
    """
    t = np.asarray(t, float)
    sp = np.asarray(setpoint, float)
    y = np.asarray(pv, float)
    if not (len(t) == len(sp) == len(y)):
        raise ValueError("t, setpoint and pv must have equal length")

    dsp = np.abs(np.diff(sp))
    if len(dsp) == 0 or dsp.max() == 0:
        return Finding("step_response", False, Severity.INFO, "No setpoint step found.")
    k = int(np.argmax(dsp)) + 1
    t_step = float(t[k])
    y0 = float(np.mean(y[max(0, k - 10):k])) if k > 0 else float(y[0])
    target = float(sp[k])
    span = target - y0
    if span == 0:
        return Finding("step_response", False, Severity.INFO, "Setpoint step has zero span.")

    tt, yy = t[k:], y[k:]
    norm = (yy - y0) / span  # 0 at start, 1 at target

    def _first_cross(level):
        idx = np.nonzero(norm >= level)[0]
        return float(tt[idx[0]]) if len(idx) else None

    t10, t90 = _first_cross(0.1), _first_cross(0.9)
    rise = (t90 - t10) if (t10 is not None and t90 is not None) else None

    overshoot = max(0.0, float(np.max(norm) - 1.0) * 100.0)
    outside = np.nonzero(np.abs(norm - 1.0) > settle_band)[0]
    if len(outside) == 0:
        settling = 0.0
        settled = True
    elif outside[-1] == len(norm) - 1:
        settling = None
        settled = False
    else:
        settling = float(tt[outside[-1] + 1] - t_step)
        settled = True

    tail = yy[int(len(yy) * 0.9):]
    sse = float(target - np.mean(tail))

    problems = []
    if overshoot > max_overshoot_pct:
        problems.append(f"overshoot {overshoot:.1f}%")
    if not settled:
        problems.append(f"did not settle within +/-{settle_band*100:.0f}%")
    if abs(sse) > settle_band * abs(span):
        problems.append(f"steady-state error {sse:.3g}")

    t_end = float(t[-1])
    ev = [
        Evidence(signal, t_step, t_end, "overshoot_pct", overshoot, max_overshoot_pct, "%"),
        Evidence(signal, t_step, t_end, "steady_state_error", sse, settle_band * abs(span)),
    ]
    if rise is not None:
        ev.append(Evidence(signal, t10, t90, "rise_time", rise, None, "s"))
    if settling is not None:
        ev.append(Evidence(signal, t_step, t_step + settling, "settling_time", settling, None, "s"))

    detected = bool(problems)
    summary = (
        f"Step response of {signal} has problems: {', '.join(problems)}."
        if detected else f"Step response of {signal} is within limits."
    )
    return Finding(
        "step_response", detected,
        Severity.WARNING if detected else Severity.INFO,
        summary, tuple(ev),
        {"t_step": t_step, "y0": y0, "target": target, "overshoot_pct": overshoot,
         "rise_time_s": rise, "settling_time_s": settling, "settled": settled,
         "steady_state_error": sse},
    )


# --------------------------------------------------------------------------- #
# Actuator saturation
# --------------------------------------------------------------------------- #
def detect_saturation(
    t, u, u_min: float, u_max: float, signal: str = "u",
    tol: float = 1e-6, max_fraction: float = 0.2,
) -> Finding:
    t, u = _as_arrays(t, u)
    at_lim = (u >= u_max - tol) | (u <= u_min + tol)
    frac = float(np.mean(at_lim))

    longest, start_best, run, run_start = 0, 0, 0, 0
    for i, s in enumerate(at_lim):
        if s:
            if run == 0:
                run_start = i
            run += 1
            if run > longest:
                longest, start_best = run, run_start
        else:
            run = 0
    if longest:
        ts, te = float(t[start_best]), float(t[start_best + longest - 1])
    else:
        ts, te = float(t[0]), float(t[-1])

    detected = frac > max_fraction
    ev = (
        Evidence(signal, float(t[0]), float(t[-1]), "fraction_at_limit", frac, max_fraction),
        Evidence(signal, ts, te, "longest_saturated_run", te - ts, None, "s"),
    )
    summary = (
        f"Actuator {signal} saturated {frac*100:.1f}% of the time (longest run {te-ts:.2f}s)."
        if detected else f"Actuator {signal} saturation {frac*100:.1f}% is acceptable."
    )
    return Finding("saturation", detected, Severity.WARNING if detected else Severity.INFO,
                   summary, ev, {"fraction": frac, "longest_run_s": te - ts})


# --------------------------------------------------------------------------- #
# Dropouts / gaps
# --------------------------------------------------------------------------- #
def detect_dropouts(t, x=None, signal: str = "x", gap_factor: float = 3.0) -> Finding:
    """Timestamp gaps larger than gap_factor * median dt, plus NaN runs in x."""
    t = np.asarray(t, float)
    dt = np.diff(t)
    if len(dt) == 0:
        raise ValueError("need at least 2 samples")
    med = float(np.median(dt))
    gap_idx = np.nonzero(dt > gap_factor * med)[0]
    ev = [Evidence(signal, float(t[i]), float(t[i + 1]), "gap", float(dt[i]), gap_factor * med, "s")
          for i in gap_idx[:20]]

    n_nan = 0
    if x is not None:
        xv = np.asarray(x, float)
        nan = ~np.isfinite(xv)
        n_nan = int(nan.sum())
        if n_nan:
            idx = np.nonzero(nan)[0]
            ev.append(Evidence(signal, float(t[idx[0]]), float(t[idx[-1]]), "nan_samples", float(n_nan), 0))

    detected = len(gap_idx) > 0 or n_nan > 0
    summary = (
        f"{len(gap_idx)} timestamp gaps and {n_nan} missing samples in {signal} "
        f"(nominal dt {med*1000:.1f} ms)."
        if detected else f"No dropouts in {signal}."
    )
    return Finding("dropouts", detected, Severity.WARNING if detected else Severity.INFO,
                   summary, tuple(ev),
                   {"n_gaps": int(len(gap_idx)), "n_nan": n_nan, "nominal_dt_s": med,
                    "max_gap_s": float(dt.max())})


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #
def detect_noise(t, x, signal: str = "x", max_noise_std: float = 0.01) -> Finding:
    """Estimate white-noise std from second differences.

    For white noise n, std(diff(diff(n))) = sqrt(6) * std(n). Slow dynamics barely
    contribute to the second difference at the sample rate, so this is robust to
    the underlying signal shape.
    """
    t, x = _as_arrays(t, x)
    d2 = np.diff(x, n=2)
    # MAD based estimate so single spikes do not dominate
    mad = np.median(np.abs(d2 - np.median(d2))) * 1.4826
    noise = float(mad / np.sqrt(6.0))
    detected = noise > max_noise_std
    ev = (Evidence(signal, float(t[0]), float(t[-1]), "noise_std", noise, max_noise_std),)
    summary = (f"High measurement noise on {signal}: std {noise:.4g}."
               if detected else f"Noise on {signal} is low (std {noise:.4g}).")
    return Finding("noise", detected, Severity.WARNING if detected else Severity.INFO,
                   summary, ev, {"noise_std": noise})


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #
def detect_drift(t, x, signal: str = "x", max_change: float = 0.05) -> Finding:
    """Least-squares slope; flags if slope * duration exceeds max_change."""
    t, x = _as_arrays(t, x)
    slope, intercept = np.polyfit(t, x, 1)
    duration = float(t[-1] - t[0])
    change = float(slope * duration)
    detected = abs(change) > max_change
    ev = (
        Evidence(signal, float(t[0]), float(t[-1]), "slope", float(slope), None, "/s"),
        Evidence(signal, float(t[0]), float(t[-1]), "total_drift", change, max_change),
    )
    summary = (f"{signal} drifts by {change:.4g} over {duration:.1f}s."
               if detected else f"No significant drift in {signal}.")
    return Finding("drift", detected, Severity.WARNING if detected else Severity.INFO,
                   summary, ev, {"slope": float(slope), "total_change": change})


# --------------------------------------------------------------------------- #
# Spikes
# --------------------------------------------------------------------------- #
def detect_spikes(t, x, signal: str = "x", window: int = 11, k: float = 6.0) -> Finding:
    """Hampel filter: points more than k robust sigmas from the rolling median."""
    t, x = _as_arrays(t, x)
    if window % 2 == 0:
        window += 1
    half = window // 2
    pad = np.pad(x, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(pad, window)
    med = np.median(windows, axis=1)
    mad = np.median(np.abs(windows - med[:, None]), axis=1) * 1.4826
    floor = max(1e-9, float(np.median(mad)))
    mad = np.maximum(mad, floor)
    z = np.abs(x - med) / mad
    idx = np.nonzero(z > k)[0]
    ev = tuple(Evidence(signal, float(t[i]), float(t[i]), "robust_z", float(z[i]), k) for i in idx[:20])
    detected = len(idx) > 0
    summary = (f"{len(idx)} spikes in {signal}." if detected else f"No spikes in {signal}.")
    return Finding("spikes", detected, Severity.WARNING if detected else Severity.INFO,
                   summary, ev, {"n_spikes": int(len(idx)),
                                 "spike_times": [float(t[i]) for i in idx[:50]]})
