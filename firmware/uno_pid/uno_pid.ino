/*
 * diagfw for Arduino Uno: a 50 Hz PID loop around one of three plants (PLANT below).
 *
 *   PLANT 2  LED + LDR (default). The LED is the actuator, an LDR facing it is the sensor.
 *            No servo, no pot, no mechanical coupling. The plant is real: the LED's
 *            brightness is measured by the LDR, so the loop is closed through hardware.
 *   PLANT 1  Emulated servo + pot. The Uno models the servo and the pot in software; the
 *            real servo still moves for show. The plant is a model.
 *   PLANT 0  Servo + pot on the servo horn. Real closed loop, needs a mechanical coupling
 *            and a two-point calibration.
 *
 * Wiring, PLANT 2
 *   D3 (PWM) -> 220 ohm -> LED anode (long leg); LED cathode (short leg) -> GND
 *   5V -> LDR -> A0, and A0 -> 10 kohm -> GND   (A0 is the middle of the divider)
 *   Face the LED at the LDR, 3 to 10 mm apart, and tape a dark tube around both so room
 *   light does not dominate. Nothing else may be connected to A0 (unplug the pot wiper).
 *   Brighter LED -> lower LDR resistance -> higher voltage on A0.
 *
 * Wiring, PLANT 0 and 1
 *   servo   signal (orange) -> D9, red -> 5V, brown -> GND
 *   pot     one outer pin -> 5V, other outer pin -> GND, middle pin -> A0
 *   PLANT 0 only: the pot shaft must turn with the servo horn (tape, glue or a coupler).
 *   Power: an SG90 on USB 5V is fine for light loads, but if the Uno resets while
 *   the servo moves, power the servo from its own 5V supply and join the GNDs.
 *
 * Protocol: see src/diagagent/devices/serial_protocol.py (115200 baud, ASCII lines).
 *   Extra commands for setup, not used by the host:
 *     "ADC" -> "OK <raw 0..1023>"
 *     "CAL" (PLANT 2, loop stopped) -> "OK lo=<adc> hi=<adc>" or "ERR cal"
 *
 * Calibration
 *   PLANT 2: automatic. After every reset (opening the serial port resets an Uno) the
 *     firmware measures the LDR with the LED off (pv = 0) and at full power (pv = 1), which
 *     also absorbs changes in room light. It takes about 1 s. If the two readings are less
 *     than MIN_CAL_RANGE counts apart (LED or LDR not connected, LED not facing the LDR),
 *     "RUN 1" answers "ERR cal". Send "CAL" to repeat it. Set AUTO_CAL to 0 to use the
 *     fixed ADC_AT_0 / ADC_AT_1 below.
 *   PLANT 0: once, by hand. Open a serial monitor at 115200, line ending "Newline".
 *     Send "HOME 0", wait 2 s, send "ADC", write the number into ADC_AT_0. Send "HOME 1",
 *     wait 2 s, send "ADC", write the number into ADC_AT_1, re-upload. A pot that reads
 *     backwards is handled by ADC_AT_0 > ADC_AT_1.
 *
 * Units: setpoint and pv are normalized 0..1 (PLANT 2: LDR reading between LED off and LED
 * full; PLANT 0 and 1: servo travel).
 *   PLANT 2: u in [0, 1] IS the LED duty (PWM). The plant has a static gain, no integrator.
 *   PLANT 0 and 1: u in [-1, 1] is a servo velocity command: position += u * VMAX * dt.
 * The firmware reports its own u limits to the host in "GET" (umin, umax).
 *
 * The LDR is nonlinear (resistance falls roughly with a power of the light level) and its
 * response is slower when the light falls than when it rises. That is part of the plant.
 */

#define PLANT 2                      // 2: LED + LDR, 1: emulated servo + pot, 0: servo + real pot
#define AUTO_CAL 1                   // PLANT 2: calibrate LED off / full at every reset

#define EMULATED_PLANT (PLANT == 1)
#define LDR_PLANT (PLANT == 2)

#if !LDR_PLANT
#include <Servo.h>
const uint8_t SERVO_PIN = 9;
const int US_AT_0 = 600;             // pulse width for position 0 (stay inside the servo's range)
const int US_AT_1 = 2300;            // pulse width for position 1
Servo servo;
#endif

const uint8_t LED_PIN = 3;           // PLANT 2: PWM output to the LED
const uint8_t ADC_PIN = A0;          // pot wiper (PLANT 0) or LDR divider (PLANT 2)
const int ADC_AT_0 = LDR_PLANT ? 50 : 100;    // fixed calibration: reading at pv = 0
const int ADC_AT_1 = LDR_PLANT ? 800 : 900;   // fixed calibration: reading at pv = 1
const int MIN_CAL_RANGE = 60;        // PLANT 2: smaller LED-off/LED-full difference = wiring fault
const unsigned long CAL_SETTLE_MS = 400;      // PLANT 2: let the LDR settle after a LED change
const uint8_t ADC_AVG = 4;           // PLANT 2: readings averaged per sample

const float VMAX = 2.0f;             // servo: position units per second at |u| = 1
#if LDR_PLANT
const float U_MIN = 0.0f, U_MAX = 1.0f;       // LED duty
#else
const float U_MIN = -1.0f, U_MAX = 1.0f;      // servo velocity command
#endif
const float SP_MAX = 0.9f, BASE_SP = 0.2f;
const unsigned long SAMPLE_US = 20000UL;      // 50 Hz

const float KP_MAX = 50.0f, KI_MAX = 50.0f, KD_MAX = 20.0f;
// The gains reset whenever the board resets, and opening the serial port resets an Uno,
// so these are what the host always starts from. The emulated plant and the LED loop start
// with a bad tune as the demo fault; the real-pot servo loop starts conservative.
// LED + LDR, measured on the bench (2026-11-17, LED a few cm from the LDR, P-only loop):
// stable at kp 0.5, sustained oscillation at kp 1 with a 0.2 s period, so Ku is between 0.5
// and 1 and Pu is 0.2 s. The demo fault starts just above Ku. Re-measure with
// scripts/gain_sweep.py if you move the LED or the LDR: the plant gain depends on the
// geometry (too close and the LDR saturates at a small duty, and everything oscillates).
#if EMULATED_PLANT
float kp = 3.0f, ki = 50.0f, kd = 0.0f;
#elif LDR_PLANT
float kp = 1.0f, ki = 1.0f, kd = 0.0f;
#else
float kp = 3.0f, ki = 0.0f, kd = 0.0f;
#endif

bool running = false;
float sp = BASE_SP;
float thetaCmd = 0.0f;               // servo: commanded position
float ledDuty = 0.0f;                // PLANT 2: LED duty, 0..1
int adcLo = ADC_AT_0, adcHi = ADC_AT_1;
bool calOk = true;
float integ = 0.0f, dState = 0.0f, yPrev = 0.0f;
bool havePrev = false;
unsigned long t0Ms = 0, nextUs = 0;

char line[48];
uint8_t len = 0;

const float SERVO_TAU = 0.08f;       // s, lag between commanded and actual position (emulation)
const float SERVO_MAX_RATE = 3.0f;   // position units per second (emulation)
const float POT_NOISE = 0.003f;      // measurement noise, position units (emulation)
float theta = 0.0f;                  // emulated actual servo position

int readAdc(uint8_t n) {
  long sum = 0;
  for (uint8_t i = 0; i < n; i++) sum += analogRead(ADC_PIN);
  return (int)(sum / n);
}

float adcToPv(int adc) {
  if (adcHi == adcLo) return 0.0f;
  return constrain((adc - adcLo) / (float)(adcHi - adcLo), 0.0f, 1.0f);
}

float readPv() {
#if EMULATED_PLANT
  float noise = random(-1000, 1001) / 1000.0f * POT_NOISE * 1.7f;   // roughly 0.003 std
  float y = constrain(theta + noise, 0.0f, 1.0f);
  return round(y * 1023.0f) / 1023.0f;                               // 10-bit quantization
#elif LDR_PLANT
  return adcToPv(readAdc(ADC_AVG));
#else
  return adcToPv(analogRead(ADC_PIN));
#endif
}

void setLed(float duty) {
  ledDuty = constrain(duty, 0.0f, 1.0f);
  analogWrite(LED_PIN, (int)(ledDuty * 255.0f + 0.5f));
}

#if LDR_PLANT
// Measure the LDR with the LED off and at full power. Returns false if they are too close.
bool calibrate() {
  setLed(0.0f);
  delay(CAL_SETTLE_MS);
  int lo = readAdc(16);
  setLed(1.0f);
  delay(CAL_SETTLE_MS);
  int hi = readAdc(16);
  setLed(0.0f);
  if (abs(hi - lo) < MIN_CAL_RANGE) return false;
  adcLo = lo;
  adcHi = hi;
  return true;
}
#else
void moveServo(float pos) {
  servo.writeMicroseconds(US_AT_0 + (int)(pos * (US_AT_1 - US_AT_0)));
}
#endif

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
#if LDR_PLANT
  setLed(u);
#else
  thetaCmd = constrain(thetaCmd + u * VMAX * dt, 0.0f, 1.0f);
#if EMULATED_PLANT
  float next = theta + (thetaCmd - theta) * (1.0f - exp(-dt / SERVO_TAU));
  theta += constrain(next - theta, -SERVO_MAX_RATE * dt, SERVO_MAX_RATE * dt);
#endif
  moveServo(thetaCmd);
#endif
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
#if LDR_PLANT
    setLed(v);                         // open loop: LED duty = v, then let the LDR settle
#else
    servo.attach(SERVO_PIN);
    thetaCmd = v;
    theta = v;
    moveServo(v);
#endif
    delay(1500);                       // let the actuator arrive
    Serial.println(F("OK"));
  } else if (!strcmp(w, "RUN") && a) {
    if (a[0] == '1' && !running) {
#if LDR_PLANT
      if (!calOk) { Serial.println(F("ERR cal")); return; }
      resetLoop();
      // bumpless start: choose the integrator so the first output equals the current LED duty
      float e0 = sp - readPv();
      integ = (ki > 0.001f) ? constrain((ledDuty - kp * e0) / ki, 0.0f, 1.0f / ki) : 0.0f;
#else
      servo.attach(SERVO_PIN);
      thetaCmd = readPv();             // bumpless start from where the servo is (or is emulated)
      resetLoop();
#endif
      t0Ms = millis();
      nextUs = micros() + SAMPLE_US;
      running = true;
    } else if (a[0] == '0') {
      running = false;                 // the actuator holds its last output
    }
    Serial.println(F("OK"));
  } else if (!strcmp(w, "ADC")) {
    Serial.print(F("OK ")); Serial.println(analogRead(ADC_PIN));
#if LDR_PLANT
  } else if (!strcmp(w, "CAL")) {
    if (running) { Serial.println(F("ERR state")); return; }
    calOk = calibrate();
    if (calOk) {
      Serial.print(F("OK lo=")); Serial.print(adcLo);
      Serial.print(F(" hi=")); Serial.println(adcHi);
    } else {
      Serial.println(F("ERR cal"));
    }
#endif
  } else {
    Serial.println(F("ERR unknown"));
  }
}

void setup() {
  Serial.begin(115200);
#if LDR_PLANT
  pinMode(LED_PIN, OUTPUT);
  setLed(0.0f);
#if AUTO_CAL
  calOk = calibrate();                 // the host retries PING for 5 s, so this ~1 s is fine
#endif
#endif
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
