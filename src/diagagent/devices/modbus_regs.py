"""Register map shared by the simulated PLC and the Modbus client.

Modbus moves 16-bit registers, so real numbers are stored as scaled integers.
Everything here is a holding register (function codes 3, 6, 16), addressed from 0.

    addr  meaning                          encoding            access
    ----  -------------------------------  ------------------  ------
      0   process value (pv)               int16  x 1000       read
      1   setpoint                         int16  x 1000       read/write
      2   controller output (u)            int16  x 1000       read
      3   simulated time since run start   uint16 x 100 (10 ms) read
     10   Kp                               uint16 x 1000       read/write
     11   Ki                               uint16 x 1000       read/write
     12   Kd                               uint16 x 1000       read/write
     20   run flag (0 stop, 1 run)         uint16              read/write
     30   status (0 ok, 1 gain rejected)   uint16              read
     40   actuator u_min                   uint16 x 1000       read
     41   actuator u_max                   uint16 x 1000       read
     42   highest safe step setpoint       uint16 x 1000       read

Writing 1 to the run flag resets the loop (plant, integrator, clock) and starts it.
"""

SCALE = 1000
TIME_TICKS_PER_S = 100

R_PV, R_SP, R_U, R_TIME = 0, 1, 2, 3
R_KP, R_KI, R_KD = 10, 11, 12
R_RUN = 20
R_STATUS = 30
R_UMIN, R_UMAX, R_SPMAX = 40, 41, 42
N_REGS = 64

STATUS_OK = 0
STATUS_GAIN_REJECTED = 1

GAIN_REGS = {"kp": R_KP, "ki": R_KI, "kd": R_KD}
# What the PLC accepts. Mirrors LIMITS in sim_device.py (a test checks they agree).
GAIN_LIMITS = {"kp": (0.0, 50.0), "ki": (0.0, 50.0), "kd": (0.0, 20.0)}


def enc_signed(x: float) -> int:
    v = max(-32768, min(32767, int(round(x * SCALE))))
    return v & 0xFFFF


def dec_signed(r: int) -> float:
    return (r - 65536 if r >= 32768 else r) / SCALE


def enc_unsigned(x: float) -> int:
    return max(0, min(65535, int(round(x * SCALE))))


def dec_unsigned(r: int) -> float:
    return r / SCALE
