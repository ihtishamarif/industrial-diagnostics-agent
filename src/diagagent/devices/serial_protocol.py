"""Line protocol between the host and a serial PID controller (Arduino Uno, ESP32, ...).

Plain ASCII, one command per line (LF), 115200 baud 8N1. The device answers every
command with exactly one line that starts with OK or ERR. While the loop runs it
also streams data lines. Both kinds of line can interleave, so the host tells them
apart by their first characters.

    host -> device         device -> host
    PING                   OK diagfw 1
    GET                    OK kp=4.000 ki=1.500 kd=0.000 umin=-1.000 umax=1.000 spmax=0.900
    SET kp 1.82            OK kp=1.820        or  ERR range   (the device validates itself)
    SP 0.6                 OK                 or  ERR range
    HOME 0.2               OK                 (goes to the start where pv = 0.2, blocks 1.5 to 3 s)
    RUN 1                  OK, then data lines
    RUN 0                  OK, data lines stop

    data line while running:   D,<ms since RUN 1>,<setpoint>,<pv>,<u>

Units: setpoint and pv are normalized sensor readings, 0.0 to 1.0. The firmware says in
GET what range u has (umin, umax), and the host uses those limits as they come:

    servo plant   pv = servo position (1.0 = full travel), u in [-1, 1] is a velocity
                  command: the commanded position moves by u * VMAX per second.
    LED plant     pv = LDR reading between LED off (0.0) and LED full (1.0), u in [0, 1]
                  is the LED duty itself.
"""

VERSION = 1
BAUD = 115200
HELLO = f"OK diagfw {VERSION}"

GAIN_LIMITS = {"kp": (0.0, 50.0), "ki": (0.0, 50.0), "kd": (0.0, 20.0)}
U_MIN, U_MAX = -1.0, 1.0          # servo plant: velocity command
LED_U_MIN, LED_U_MAX = 0.0, 1.0   # LED plant: duty
SP_MAX = 0.9          # highest setpoint a step test may use
BASE_SP = 0.2         # the loop is homed here before a step test
VMAX = 2.0            # position units per second at |u| = 1
SAMPLE_DT = 0.02      # the firmware runs the loop at 50 Hz


def parse_kv(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for tok in text.split():
        k, _, v = tok.partition("=")
        out[k] = float(v)
    return out


def parse_data(line: str) -> tuple[float, float, float, float] | None:
    """Return (t_seconds, setpoint, pv, u) or None if the line is damaged."""
    parts = line.split(",")
    if len(parts) != 5 or parts[0] != "D":
        return None
    try:
        ms, sp, pv, u = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return None
    return ms / 1000.0, sp, pv, u
