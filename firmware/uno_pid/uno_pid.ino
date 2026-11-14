/*
 * diagfw for Arduino Uno: a PID position loop around a hobby servo (SG90),
 * with a potentiometer on the servo horn as the measurement.
 *
 * Wiring
 *   servo   signal (orange) -> D9, red -> 5V, brown -> GND
 *   pot     one outer pin -> 5V, other outer pin -> GND, middle pin -> A0
 *   The pot shaft must turn with the servo horn (tape, glue or a printed coupler).
 *   Power: an SG90 on USB 5V is fine for light loads, but if the Uno resets while
 *   the servo moves, power the servo from its own 5V supply and join the GNDs.
 *
 * Protocol: see src/diagagent/devices/serial_protocol.py (115200 baud, ASCII lines).
 *   Extra command for setup, not used by the host: "ADC" -> "OK <raw 0..1023>".
 *
 * Calibration (once)
 *   1. Open a serial monitor at 115200, line ending "Newline".
 *   2. Send "HOME 0", wait 2 s, send "ADC". Write the number into ADC_AT_0.
 *   3. Send "HOME 1", wait 2 s, send "ADC". Write the number into ADC_AT_1.
 *   4. Re-upload. If the pot reads backwards, ADC_AT_0 > ADC_AT_1 handles it.
 *
 * Plant modes (EMULATED_PLANT below)
 *   1 = the Uno simulates the servo and pot in software (lag, speed limit, noise, 10-bit
 *       ADC). The real servo still moves to the commanded position so you can watch it,
 *       but the pot is not read. No pot-to-horn coupling needed. The plant is a model.
 *   0 = real closed loop: pv comes from the pot on A0 (wiring and calibration above).
 *
 * Units: setpoint and pv are normalized position 0..1 (1 = full servo travel),
 * u in [-1, 1] is a velocity command: commanded position += u * VMAX * dt.
 */
#include <Servo.h>

#define EMULATED_PLANT 1             // 1: simulated servo+pot plant, 0: read the real pot

const uint8_t SERVO_PIN = 9;
const uint8_t POT_PIN = A0;
const int ADC_AT_0 = 100;            // CALIBRATE: ADC reading with the servo at position 0
const int ADC_AT_1 = 900;            // CALIBRATE: ADC reading with the servo at position 1
const int US_AT_0 = 600;             // pulse width for position 0 (stay inside the servo's range)
const int US_AT_1 = 2300;            // pulse width for position 1

const float VMAX = 2.0f;             // position units per second at |u| = 1
const float U_MIN = -1.0f, U_MAX = 1.0f;
const float SP_MAX = 0.9f, BASE_SP = 0.2f;
const unsigned long SAMPLE_US = 20000UL;   // 50 Hz

const float KP_MAX = 50.0f, KI_MAX = 50.0f, KD_MAX = 20.0f;
float kp = 3.0f, ki = 0.0f, kd = 0.0f;     // start conservative; the host retunes them

Servo servo;
bool running = false;
float sp = BASE_SP;
float thetaCmd = 0.0f;               // commanded servo position
float integ = 0.0f, dState = 0.0f, yPrev = 0.0f;
bool havePrev = false;
unsigned long t0Ms = 0, nextUs = 0;

char line[48];
uint8_t len = 0;

const float SERVO_TAU = 0.08f;       // s, lag between commanded and actual position (emulation)
const float SERVO_MAX_RATE = 3.0f;   // position units per second (emulation)
const float POT_NOISE = 0.003f;      // measurement noise, position units (emulation)
float theta = 0.0f;                  // emulated actual servo position

float readPv() {
#if EMULATED_PLANT
  float noise = random(-1000, 1001) / 1000.0f * POT_NOISE * 1.7f;   // roughly 0.003 std
  float y = constrain(theta + noise, 0.0f, 1.0f);
  return round(y * 1023.0f) / 1023.0f;                               // 10-bit quantization
#else
  float y = (analogRead(POT_PIN) - ADC_AT_0) / (float)(ADC_AT_1 - ADC_AT_0);
  return constrain(y, 0.0f, 1.0f);
#endif
}

void moveServo(float theta) {
  servo.writeMicroseconds(US_AT_0 + (int)(theta * (US_AT_1 - US_AT_0)));
}

void resetLoop() {
  integ = dState = 0.0f;
  havePrev = false;
}

void stepLoop() {
  const float dt = SAMPLE_US / 1e6f;
  float y = readPv();
  float e = sp - y;
  float rawD = havePrev ? -(y - yPrev) / dt : 0.0f;
  float tf = (kd > 0 && kp > 0) ? kd / (kp * 10.0f) : 0.0f;
  dState = (tf > 0) ? dState + dt / (tf + dt) * (rawD - dState) : rawD;
  yPrev = y;
  havePrev = true;
  float uUnsat = kp * e + ki * integ + kd * dState;
  float u = constrain(uUnsat, U_MIN, U_MAX);
  if (!((uUnsat > U_MAX && e > 0) || (uUnsat < U_MIN && e < 0))) integ += e * dt;  // anti-windup
  thetaCmd = constrain(thetaCmd + u * VMAX * dt, 0.0f, 1.0f);
#if EMULATED_PLANT
  float next = theta + (thetaCmd - theta) * (1.0f - exp(-dt / SERVO_TAU));
  theta += constrain(next - theta, -SERVO_MAX_RATE * dt, SERVO_MAX_RATE * dt);
#endif
  moveServo(thetaCmd);
  Serial.print(F("D,")); Serial.print(millis() - t0Ms);
  Serial.print(','); Serial.print(sp, 3);
  Serial.print(','); Serial.print(y, 4);
  Serial.print(','); Serial.println(u, 3);
}

void handle(char *cmd) {
  char *w = strtok(cmd, " ");
  if (!w) return;
  char *a = strtok(NULL, " ");
  char *b = strtok(NULL, " ");
  if (!strcmp(w, "PING")) {
    Serial.println(F("OK diagfw 1"));
  } else if (!strcmp(w, "GET")) {
    Serial.print(F("OK kp=")); Serial.print(kp, 3);
    Serial.print(F(" ki=")); Serial.print(ki, 3);
    Serial.print(F(" kd=")); Serial.print(kd, 3);
    Serial.print(F(" umin=")); Serial.print(U_MIN, 3);
    Serial.print(F(" umax=")); Serial.print(U_MAX, 3);
    Serial.print(F(" spmax=")); Serial.println(SP_MAX, 3);
  } else if (!strcmp(w, "SET") && a && b) {
    float v = atof(b);
    float *target = !strcmp(a, "kp") ? &kp : !strcmp(a, "ki") ? &ki : !strcmp(a, "kd") ? &kd : NULL;
    float hi = !strcmp(a, "kp") ? KP_MAX : !strcmp(a, "ki") ? KI_MAX : KD_MAX;
    if (!target) { Serial.println(F("ERR unknown")); return; }
    if (v < 0 || v > hi) { Serial.println(F("ERR range")); return; }   // the firmware protects itself
    *target = v;
    Serial.print(F("OK ")); Serial.print(a); Serial.print('='); Serial.println(v, 3);
  } else if (!strcmp(w, "SP") && a) {
    float v = atof(a);
    if (v < 0 || v > SP_MAX) { Serial.println(F("ERR range")); return; }
    sp = v;
    Serial.println(F("OK"));
  } else if (!strcmp(w, "HOME") && a) {
    float v = atof(a);
    if (running) { Serial.println(F("ERR state")); return; }
    if (v < 0 || v > 1) { Serial.println(F("ERR range")); return; }
    servo.attach(SERVO_PIN);
    thetaCmd = v;
    theta = v;
    moveServo(v);
    delay(1500);                       // let the servo arrive
    Serial.println(F("OK"));
  } else if (!strcmp(w, "RUN") && a) {
    if (a[0] == '1' && !running) {
      servo.attach(SERVO_PIN);
      thetaCmd = readPv();             // bumpless start from where the servo is (or is emulated)
      resetLoop();
      t0Ms = millis();
      nextUs = micros() + SAMPLE_US;
      running = true;
    } else if (a[0] == '0') {
      running = false;                 // the servo holds its last position
    }
    Serial.println(F("OK"));
  } else if (!strcmp(w, "ADC")) {
    Serial.print(F("OK ")); Serial.println(analogRead(POT_PIN));
  } else {
    Serial.println(F("ERR unknown"));
  }
}

void setup() {
  Serial.begin(115200);
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { line[len] = 0; handle(line); len = 0; }
    else if (c != '\r' && len < sizeof(line) - 1) line[len++] = c;
  }
  if (running && (long)(micros() - nextUs) >= 0) {
    nextUs += SAMPLE_US;
    stepLoop();
  }
}
