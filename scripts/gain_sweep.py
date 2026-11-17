"""Find where a serial loop starts to oscillate, by raising Kp with Ki = Kd = 0.

For each Kp the script runs a step test, then prints how much pv swings after the loop
has had time to settle, whether the oscillation detector fires, and the period. The
smallest Kp with a sustained oscillation is the ultimate gain Ku, and its period is Pu
(Ziegler-Nichols). Use them to pick a demo fault that is just above Ku (a retune by the
Tyreus-Luyben rule then lands in a stable range) and to see how noisy pv really is.

The original gains are restored at the end, even if the sweep fails.

    python scripts/gain_sweep.py --port /dev/ttyACM0
    python scripts/gain_sweep.py --simulate led          # against the simulated firmware
"""

from __future__ import annotations

import argparse
import csv
import sys

from diagagent.devices.base import WriteRequest
from diagagent.devices.serial_device import SerialPIDDevice
from diagagent.signals import control as ctl


SWING = 0.10     # pv peak-to-peak after settling that counts as a large swing


def _set(dev: SerialPIDDevice, **gains: float) -> None:
    dev.apply(WriteRequest(dev.device_id, gains, reason="gain sweep", approved=True,
                           approved_by="gain_sweep"))


def sweep(dev: SerialPIDDevice, gains: list[float], duration: float, settle: float):
    limits = dev.read_params()
    original = {k: limits[k] for k in ("kp", "ki", "kd")}
    rows = []
    try:
        for kp in gains:
            _set(dev, kp=kp, ki=0.0, kd=0.0)
            df = dev.capture(duration=duration)
            tail = df[df.t > settle]
            rep = ctl.analyze_control_loop(df, sp_col="setpoint", pv_col="pv", u_col="u",
                                           u_min=limits["u_min"], u_max=limits["u_max"])
            osc = rep.get("oscillation")
            detected = bool(osc and osc.detected)
            rows.append({
                "kp": kp,
                "pv_mean": round(float(tail.pv.mean()), 3),
                "pv_peak_to_peak": round(float(tail.pv.max() - tail.pv.min()), 3),
                "pv_std": round(float(tail.pv.std()), 4),
                "u_min": round(float(tail.u.min()), 2),
                "u_max": round(float(tail.u.max()), 2),
                "oscillating": detected,
                "period_s": round(float(osc.details["period_s"]), 3) if detected else "",
            })
            print("  kp {kp:>5g}  pv mean {pv_mean:.3f}  p-p {pv_peak_to_peak:.3f}  std {pv_std:.4f}  "
                  "u {u_min:.2f}..{u_max:.2f}  oscillating {oscillating}  period {period_s}".format(**rows[-1]))
    finally:
        _set(dev, **original)
        print(f"Restored gains {original}")
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--port", help="serial port of the board, e.g. /dev/ttyACM0")
    g.add_argument("--simulate", choices=["servo", "led"], help="use the simulated firmware")
    p.add_argument("--gains", default="1,2,3,4,5,6,8,10,15",
                   help="comma-separated Kp values, ascending (default: %(default)s)")
    p.add_argument("--duration", type=float, default=6.0, help="seconds per step test")
    p.add_argument("--settle", type=float, default=3.0, help="ignore the first seconds for the statistics")
    p.add_argument("--out", default="gain_sweep.csv")
    a = p.parse_args(argv)
    gains = [float(x) for x in a.gains.split(",")]

    fw = None
    if a.simulate:
        from diagagent.sim.serial_fw import SimulatedFirmware
        fw = SimulatedFirmware(time_scale=10.0, plant=a.simulate).start()
        dev = SerialPIDDevice(fw.port, reset_delay=0.0)
    else:
        dev = SerialPIDDevice(a.port)
    try:
        print(f"Sweeping Kp over {gains} with Ki = Kd = 0 ({a.duration:g} s each)")
        rows = sweep(dev, gains, a.duration, a.settle)
    finally:
        dev.close()
        if fw:
            fw.stop()
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    first = next((r for r in rows if r["oscillating"]), None)
    if first:
        print(f"\nFirst sustained oscillation at Kp = {first['kp']:g} (period {first['period_s']} s): "
              "Ku is at or just below this.")
    else:
        big = next((r for r in rows if r["pv_peak_to_peak"] > SWING), None)
        if big:
            print(f"\nThe detector did not fire, but pv swings more than {SWING:g} peak to peak from "
                  f"Kp = {big['kp']:g} on: look at the table, Ku is probably near there.")
        else:
            print("\nNo oscillation or large swing in this range: raise --gains.")
    print(f"Saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
