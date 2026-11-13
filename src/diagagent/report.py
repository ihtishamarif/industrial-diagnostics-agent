"""Plots and markdown results for demos."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_before_after(before: pd.DataFrame, after: pd.DataFrame, path: str | Path,
                      title: str = "PID step response: before vs after retune") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(before.t, before.setpoint, "k--", lw=1, label="setpoint")
    ax1.plot(before.t, before.pv, color="#c0392b", lw=1.4, label="pv before")
    ax1.plot(after.t, after.pv, color="#2471a3", lw=1.6, label="pv after")
    ax1.set_ylabel("process value")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)
    ax1.set_title(title)
    ax2.plot(before.t, before.u, color="#c0392b", lw=1.2, label="u before")
    ax2.plot(after.t, after.u, color="#2471a3", lw=1.4, label="u after")
    ax2.set_ylabel("controller output")
    ax2.set_xlabel("time [s]")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def results_table(before: dict, after: dict, gains_before: dict, gains_after: dict) -> str:
    def f(v, nd=2):
        if v is None:
            return "not settled"
        if isinstance(v, bool):
            return "yes" if v else "no"
        return f"{v:.{nd}f}"

    rows = [
        ("Kp / Ki / Kd",
         f"{gains_before['kp']:.3g} / {gains_before['ki']:.3g} / {gains_before['kd']:.3g}",
         f"{gains_after['kp']:.3g} / {gains_after['ki']:.3g} / {gains_after['kd']:.3g}"),
        ("Overshoot [%]", f(before.get("overshoot_pct"), 1), f(after.get("overshoot_pct"), 1)),
        ("Settling time 2% [s]", f(before.get("settling_time_s")), f(after.get("settling_time_s"))),
        ("Rise time 10-90% [s]", f(before.get("rise_time_s")), f(after.get("rise_time_s"))),
        ("Sustained oscillation", f(before.get("oscillating")), f(after.get("oscillating"))),
        ("Detected anomalies", str(before.get("n_anomalies")), str(after.get("n_anomalies"))),
    ]
    out = ["| Metric | Before | After |", "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(out)
