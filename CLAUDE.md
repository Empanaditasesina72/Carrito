# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current focus & roadmap (updated 2026-07-28 — read this first when resuming)

**Goal:** the car drives itself down the right lane, stops at the STOP sign, and
the paper gets its P2 braking row (N=10). Sign detection and steering both run on
the Pi CPU (`USE_IMX500_NPU=False`).

**The vehicle moved to a NEW track**, in the city apartment. Everything geometric
from the 3.46 m track is gone. Photos of it: `Pictures/Car3` and `Car4` (their DSC
numbers repeat across folders but the files differ — never mix them).

### THE BRAKE GATE FIRED. 2026-07-28, first time ever:

```
[FSM] Braking (camera 313mm)
[FSM] Braking (camera 261mm)
```

`SIGN_BBOX_STOP_MM` had **never** been reachable before — the section this
replaces said flatly that it "can never fire". Moving the sign against the lane
edge is what did it: the post went from ~28 cm off the camera axis to ~20 cm, so
with `tan(HFOV/2) = 320/490` the octagon's centre now stays in frame to ~306 mm,
past the 320 mm gate. Stops are decided by the threshold now, not by whatever
distance the sign happened to vanish at.

`SIGN_LOST_COAST_S` was cut 0.6 → **0.15** for the same reason: the blind stretch
shrank from ~13 cm to ~3.6 cm and 0.6 s would dead-reckon straight into the sign.

### City track, tape-measured 2026-07-28

Two lines, not three: white tape on the RIGHT edge, dashed centre, and an
**unpainted far edge** (mat against bright tile).

| | |
|---|---|
| Carriageway | 66.0 cm = 35.0 (far→dashed) + 31.0 (dashed→tape) |
| Start line → sign post | 150 cm |
| Octagon | 9.0 cm across the flats (was 8.5 — the pinhole under-read every distance by 5.9 %) |
| Sign post off the camera axis | ~20 cm (base 7.4 cm, sitting against the tape) |
| Detection | **86 % at 33 cm**, 71–78 % at 158 cm |

### Open blockers (2026-07-28 night, work stopped here)

1. **The car stalls.** `estancado -- reintentando arranque` fires repeatedly, once
   for 12 s straight. `--cruise 35` gives PRECAUCION 29.75 % in `demo_full`, and it
   still stalls. Suspect the battery again, or the felt mat's rolling resistance.
   **This is what blocks the 10 trials** — a stalled trial is a `max_drive` row and
   does not count.
2. **Lane error walks to +60…+120 px (9–18 cm)** during the later phases, with
   confidence dropping to 50 % (RIGHT only). The car is losing the dashed line and
   drifting. Weave 18.3 px on the second pilot.
3. `STEER_KP = 0.08` was sized for **6.8 px/cm**; the BEV now measures ~**11.6
   px/cm**, so effective loop gain is ~1.7× the design. If it weaves, try 0.05.
   (The scale moved because the camera was re-aimed — see below.)

### Calibration, all measured on the car

| Parameter | Value | How it was found |
|---|---|---|
| `SERVO_TRIM_DEG` | **−7.5** | Closed-loop; residual went +3.97° → +0.58°. Unchanged by the move. |
| `LANE_ERROR_OFFSET_PX` | **8.5** | Car centred by hand, `diag_track`: +8.5 px settled, std 1.1. Verified against the mask's own column bands — this one is on painted lines. |
| `LANE_RIGHT_BIAS` | **0.50** | NOT 0.765. See below: the pair classifier mislabels the driven lane as the carriageway, and 0.50 is right under either branch on a two-line track. |
| `LANE_DRIVEN_WIDTH_M` | **0.31** | Tape measure. |
| `STEER_KP/KI/KD` | **0.08 / 0 / 0** | See blocker 3 — may need 0.05 here. |
| `MOTOR_MIN_MOVE_PWM` | **20.0** | Unladen: 20 % → +20.8 cm in 1.4 s. |

**Weight matters more than duty.** With a spare battery on the chassis the car
would not move at **95 %**; unladen it moves at **20 %**. Keep the chassis light.

### Two traps this track set, both already paid for

**The camera was aimed level, not 10° down.** For hours the BEV contained a
doorway, a wall and a bench — no road at all — while `diag_track` reported "both
lines found, 100 % confidence" in 20/20 frames. Three runs gave separations of
184, 316 and 386 px: three different pairs of *furniture*, held rock-steady
because a static scene produces a static error. A calibration measured from it
(58.5 → 23.1) was the offset between two pieces of furniture. **Always look at
`/tmp/diag/2_bev.png` before believing any confidence number.**

**`BEV_SCALE_PX_PER_CM` is defined, not measured.** `384 / (LANE_WIDTH_M * 100)`
merely asserts the carriageway fills the destination window. That was verified at
56.5 cm and silently became false at 66 cm — the homography is a fixed trapezoid
in frame coordinates and knows nothing about road width. The result is a
misclassification, not a rounding error: a real 275 px lane reads 53 % wide for
`lane_px` (rejected) and 28 % narrow for `road_px` (accepted), so the pipeline
applied `LANE_RIGHT_BIAS` *across the driven lane* and aimed 8 cm off centre every
frame. Hence bias 0.50.

**A "the line must lie on the road" mask gate was tested and rejected**, on the
real BEV: the background under the three parasitic bands measured 41/41/49 against
36/44 for the dashed line and the tape. They are narrow bright strips along the
bench's pale base — geometrically identical to a lane line, and brightness does
not separate them either (187 dashed vs 172 parasite). The fix is physical: dark
tape over the bench base.

### Also blocked on hardware (do not re-diagnose in software)

**REVERSE IS DEAD** — the LPWM leg of the IBT-2 does not conduct. Three
independent methods: sign distance identical at 6 duties (36.5 cm every time);
lane line positions dL/dR/dSep all +0.0; whole-frame difference at the noise floor
at 40/60/80/95 %. Forward at **20 %** moves 20 cm. So **`ParkingFSM` cannot
execute** and the car cannot return itself to a start line — reposition by hand.

### Pi access, 2026-07-28

`ssh angel01@100.68.26.120` (Tailscale; the 192.168.x addresses are stale). The
Pi's repo had **no remotes** — it only ever received pushes from the PC — so
`git pull origin main` failed for a while. `origin` is now added and the PC's `pi`
remote points at the Tailscale address; both paths work. `ncnn` is NOT installed
(PEP 668), so the detector runs the slower PyTorch path: `pip install
--break-system-packages ncnn` fixes it.

### Paper status — WITCOM 2026 / Springer CCIS, paper 2069

Scores were 0/1/1 and all three reviewers said the same thing in different words:
no physical validation. `docs/RESPONSE_TO_REVIEWERS.md` lists the ten changes
promised in the rebuttal. Nine are done.

| # | Promised | State |
|---|---|---|
| 1 | Reframe Sim2Real → SIL | text edit, wording in `docs/paper_snippets.md` |
| 2 | On-device latency + distribution | ✅ 3.83 ms mean, p99 6.16, 0 deadline misses / 999 cycles |
| **3** | **10 physical braking trials ± std** | 🔴 **THE ONLY BLOCKER** |
| 4 | Table 2 → test matrix | ✅ `docs/TEST_MATRIX.md`, 9 rows + LaTeX |
| 5 | Fix 20 ms vs 200 ms | text edit |
| 6 | Error against the ±30 mm band | depends on #3 |
| 7 | Perception metrics + CPU/NPU | ✅ 100 % recall, conf 0.824, 0 FP |
| 8 | PID equations, gains, limitations | ✅ all measured on the car |
| 9 | Figures ≥300 dpi + hardware photo | ✅ `Desktop/TMR2026_figuras/` (16 MP, 2 annotated) |
| 10 | Tagged release + DOI | pending, ~1 h of admin |

**Without #3 the rebuttal promises something that was never done** — worse than not
promising it. It is no longer a technical risk: the car drove, detected and braked.

Two things to settle before submitting:
- **P4 contradicts itself.** CPU inference measured 81.3 ms (2026-07-25) and
  64.4 ms (2026-07-27), same model and imgsz — probably the CPU governor or the
  retrained weights. Re-run `tools/bench_ablation.py` and quote ONE figure.
- **The shipped config is the CPU path.** `USE_IMX500_NPU=False` because the .rpk
  has the channel swap baked in. The P4 numbers stand, but the text must say so.

### Next steps

1. Charge the motor battery.
2. Move the sign closer to the lane edge — it is what tightens the stop (see the
   LENS section above), and it costs nothing.
3. 2–3 pilot runs, tape-measure lens→octagon, tune `SIGN_LOST_COAST_S` to centre
   on 270 mm.
4. The 10 P2 trials. `tools/bench_braking_physical.py` fsyncs each row, so a crash
   no longer loses them. Only `stop_reason == braked` counts.
5. Record with the DJI **lateral to the track, at car height**.
6. Only if the LPWM leg gets fixed: parking.

**Standing decision:** all training on the PC GPU; the Pi converts, tests and runs.

### Detector — deployed 2026-07-27

`weights/tmr_signs.pt`, 131 epochs at imgsz 320 (the production size; 640 costs
185 ms on the Pi against a 66 ms budget). On `car_hard` — 294 degraded frames of
the untouched Roboflow test split — **100 % recall at the 0.55 gate, mean
confidence 0.824, 0 false positives**, up from 0.767 / 1 FP.

Deployed `last.pt`, NOT `best.pt`. Ultralytics picks best.pt by validation mAP,
which was at its ceiling (0.995) from epoch 2 because training starts from an
already-converged model — best.pt was written at epoch 2 and never updated across
the remaining 129. Always rank with `tools/eval_hard.py`, which measures what the
FSM actually consumes, instead of trusting the checkpoint labelled "best".

Synthetic signs are generated in the band where braking is actually decided
(0.40–1.30 m = 32–104 px), not the far range: past 1.3 m nothing is reliably
detected, and closer than 43 cm the sign is out of frame.

**The printed sign is under-saturated** — measured on the Sony RAWs, H=173, S≈150,
V≈215, where a regulation red sits above S 180. That is why it reads as pink and
why on-track confidence ranges 56–81 %. Reprinting it in a stronger red will do
more than any retraining.

### Instrumentation lessons that cost hours today — do not repeat them

- **A stalled run looks perfect to every metric.** Lane-error std 0.1 px, drift
  0.0, servo frozen — that was a car that never moved. A static scene cannot
  produce a changing error.
- **Frame differencing cannot tell driving from vibrating.** Chassis vibration
  displaces each frame a few pixels at random, so consecutive frames — and even
  frames a second apart — differ as much as while driving. It reported motion at
  five duties on a stationary car, twice. Use `vision/motion_check.py` only as a
  hint; the trustworthy ground truth is **sign distance** (YOLO bbox through the
  pinhole) or **lane line positions**.
- **Mean error over a run measures where the operator PLACED the car**, not where
  it ended up. A run that started 6 cm off and corrected to centre showed a large
  mean and a large "drift" and was reported as a failure. Judge by the final
  error and the weave once settled — `tools/drive_straight.py` does.
- **A test window shorter than the measurement window measures nothing.** 0.9 s
  bursts against a 1.0 s lookback produced a whole duty sweep that described the
  test, not the car.
- **`auto_trim` writing config.py leaves the Pi's tree dirty and silently rejects
  `git push pi`.** The car ran old code for a while because of this. Commit the
  trim after applying it.

## Active system: TMR2026/

Everything under `TMR2026/` is the current vehicle. Legacy prototypes live in `_legacy/` and must not be imported from TMR2026. Project-wide docs (architecture diagram + generator) live in `docs/`; TMR2026-specific docs (SETUP, Sim2Real protocol, calibration, professor deliveries) live in `TMR2026/docs/`.

**`config.py` is the single source of truth** for GPIO pins, servo angles/limits, PID gains, speeds and gamepad button mapping. `main.py`, `main_simulator.py` and `control/fsm.py` import these values — never re-hardcode them per file.

**Root `main.py` is a loader** — it `chdir`s into `TMR2026/` and runs `TMR2026/main.py` with `runpy` so imports like `from hardware.motor import MotorDriver` keep working. The systemd service (`TMR2026/systemd/carrito_tmr.service`) points directly to `TMR2026/main.py --display` and starts under `graphical.target` (i.e. after the desktop is ready), with `DISPLAY=:0` and `XAUTHORITY=/home/angel01/.Xauthority` exported, so OpenCV can open a window on the HDMI monitor when VISION/AUTONOMOUS mode is entered. Root `main.py` is only for manual execution.

### Hardware Target

Raspberry Pi 5 with:
- Sony IMX500 NPU camera via `Picamera2`. `"RGB888"` already hands back **BGR-ordered** bytes — **do NOT add a `cv2.COLOR_RGB2BGR`**, see the hard rule below. (This line used to say the opposite.)
- IBT-2 H-bridge motor: BCM 18 (RPWM) + 13 (LPWM), `R_EN`/`L_EN` tied to 3.3 V
- PCA9685 servo on I²C bus 3 (dtoverlay GPIO 0/1), channel `config.py:SERVO_CHANNEL` (15, verified on the Pi)
- 2× VL53L0X ToF on I²C bus 4 (dtoverlay GPIO 23/22), addresses 0x30 (front) / 0x29 (rear), XSHUT pin `TMR2026/config.py:PIN_TOF_XSHUT_FRONT`
- Gamepad via `pygame` (PS4/Xbox) — buttons: A=MANUAL, B=VISION, X=AUTONOMOUS, Y=PARKING, Start=EMERGENCY (mapping in `config.py:BTN_*`). Hot-plug supported: `main.py:_pump_gamepad_events()` runs every loop iteration and reacts to SDL2 `JOYDEVICEADDED` / `JOYDEVICEREMOVED` events, so the PS4 (paired+trusted as `A0:5A:5F:0B:F7:5A`) connects automatically when powered on, even if the system booted without it. BlueZ has `AutoEnable=true` in `/etc/bluetooth/main.conf` so the BT controller comes up at boot ready to accept the trusted device.
- GPIO LEDs for turn signals / hazards / brake — pins defined in `TMR2026/vision_config.yaml` → `gpio:` and mirrored in `config.py`

GPIO is accessed via `lgpio` (chip 4 on Pi 5) with a `RPi.GPIO` fallback.

## Running the System

```bash
# From repo root (recommended for manual runs)
python main.py               # production
python main.py --display     # with debug window

# Direct (what systemd uses)
python TMR2026/main.py
```

Runtime modes (selected via gamepad/keyboard): `STANDBY · MANUAL · VISION · AUTONOMOUS · PARKING`. `Start` button = emergency freeze (brake + MANUAL). Keyboard mirror when stdin is a TTY: `A/B/X/P/Space/S/Q`.

## Installing Dependencies

```bash
pip install -r TMR2026/requirements.txt
# Pi-specific extras:
pip install picamera2 lgpio adafruit-circuitpython-vl53l0x adafruit-circuitpython-pca9685 ultralytics
```

See `TMR2026/docs/SETUP.md` for dtoverlay config and udev rules.

## Architecture (TMR2026/)

### Threads
- `CameraStream` (vision/camera_stream.py) — 30 FPS, BGR frames, locks AE/AWB after warmup
- `SignDetector` (vision/sign_detector.py) — YOLO CPU capped at 15 Hz. Auto-prefers the NCNN export `weights/tmr_signs_ncnn_model/` (3-4× faster than PyTorch on the Pi 5's ARM CPU, identical detections); falls back to `weights/tmr_signs.pt`, then to the color detector
- **NPU mode**: when `config.py:USE_IMX500_NPU=True` AND `weights/tmr_signs_imx500.rpk` exists, `main.py:_build_vision()` replaces BOTH threads above with a single `IMX500CameraStream` (vision/imx500_detector.py) — the model runs inside the camera sensor, one thread captures frame+tensors atomically via `capture_request()`. Same public API as CameraStream+SignDetector (`self.camera` and `self.sign_det` are the same object; its `start()`/`stop()` are idempotent for that reason). Full fallback chain: NPU → NCNN → .pt → color. The .rpk is generated ON THE PI with `tools/export_imx500.py` (Linux-only toolchain); see `TMR2026/docs/IMX500_NPU.md`
- `DistanceSensor` (hardware/distance_sensor.py) — 50 Hz polling, front + rear VL53L0X
- `MotorDriver` (hardware/motor.py) — internal 50 Hz soft-start ramp thread (prevents voltage sag)
- Main loop in `main.py` at 50 Hz: gamepad → FSM → servo → motor

### Perception → decision → actuation
- `vision/lane_pipeline.py` — BEV + HSV-white + sliding windows + EMA; emits `LaneResult(error_px, confidence)`
- `vision/sign_detector.py` — non-blocking queue of `Detection(label, confidence, bbox, distance_m)`; surfaces the 7 model classes (`stop`→`stop_sign`, `red`, `green`, `yellow`, `left`, `right`, `straight`) with 3-frame hysteresis, plus a red/purple color-blob STOP fallback when YOLO misses. Distance is estimated per class via pinhole.
- **Brake gating lives in `main.py:_update_vision` / `main_simulator.py:_update_vision`**: only `stop_sign` and `red` set `fsm.sign_visible` (`stop_like`). Green/arrows/yellow must NEVER brake the car — that bug (using `has_any_sign()`) is what made the physical car stop at green lights. Keep both files' gating identical (Sim2Real parity).
- `control/fsm.py` — 5-state FSM: `CRUCERO → PRECAUCION → FRENADO → ESPERA → REANUDAR`. Stop wait uses `time.monotonic()`, never `sleep()`. `brake()` is instantaneous and must not be wrapped/changed. Servo limits come from `config.py` (58°–122°) so sim and Pi share the same steering authority.
- `control/parking_fsm.py` — battery/perpendicular parking sub-FSM (`PARKING_SEARCH → PARKING_MANEUVER → PARKED`), hardware-agnostic; wired to the Y/Triángulo button in `main.py` and to the `--parking` sequence in `main_simulator.py`.
- `control/pid_controller.py` — generic PID with anti-windup and derivative-on-measurement, used for steering (lane error → servo angle)

### Vehicle lighting (signals + brake)
Three GPIO LEDs driven via `lgpio` chip 4 (BCM 17 left, 5 right, 6 brake — see `config.py:PIN_LED_*`):
- `hardware/signals.py` — `TurnSignals` with modes `OFF / LEFT / RIGHT / HAZARD`. Blink at 2 Hz (TMR regulation) is computed each frame from `time.monotonic()`; no thread, no sleep. Caller must invoke `signals.tick()` every loop iteration.
- `hardware/brake_light.py` — simple `on()` / `off()` (idempotent — only writes GPIO on state change).
- `control/fsm.py:_apply_lights()` runs every tick (not just on transitions). In `CRUCERO`/`REANUDAR` it reads `steering.current_angle` vs `SERVO_CENTER`; deviation beyond `SIGNAL_DIR_THRESH_DEG` (12°) sets `LEFT` or `RIGHT`. In `PRECAUCION`/`FRENADO`/`ESPERA` it forces `HAZARD` and `brake_light.on()`. Anywhere else → all OFF.
- `main.py` mirrors this for non-FSM modes:
  - `_do_standby` / `_do_vision` → all signals OFF, brake OFF.
  - `_do_manual` → joystick `steer_raw < -0.30` → LEFT, `> +0.30` → RIGHT, else OFF. `brake_light.on()` when `motor.current_duty < -1.0` (reversing).
  - `_do_parking` → HAZARD during the whole maneuver; `brake_light.on()` once `PARKED`.
  - `signals.tick()` is called once per frame in the main loop, after `_run_mode()`, so blink is always advanced regardless of mode.

### Steering inversion
The servo is mounted reversed on this chassis. `config.py:STEERING_INVERTED = True` flips the physical write inside `SteeringDriver.set_angle()`:
- `physical = 2 * SERVO_CENTER_ANGLE - angle_deg` is sent to the servo.
- `current_angle` always returns the **logical** angle (90 = recto, <90 = izq, >90 = der).
- All consumers (FSM lights, PID, signals, telemetry) see the logical convention. Never invert per-mode in callers — fix it at the driver if hardware changes.

### Telemetry log lines
- `_do_manual` prints (carriage-return updated): `[MAN] steer:±x.xx (angle°)  t:y.yy  b:z.zz  duty:±NN%  signs:<label>@<cm>cm, …`
- `_log_autonomous` (called every tick after `fsm.update`) prints: `[AUT] <STATE>  err:±NNNpx  angle:NN.N°  duty:±NN%  lidar:NNNNmm  signs:<label>@<cm>cm, …`
- `_do_vision` prints `[VIS] err:±Npx conf:NN%  P/I/D:±x.xx  corr:±x.xx° angle:NN.N° lidar:NNmm signs:…` — same fields as the on-screen panel.
- `signs:` field shows up to 2 detections from `SignDetector.get_detections()`; `—` if empty.
- `PIDController` exposes the last computed components via public attrs `last_error / last_p / last_i / last_d / last_output`. Read-only — they are written every `compute()` and reset by `reset()`. Used by both the on-screen overlay and console logs.

### Debug display (`--display` flag)
When `python main.py --display` is set, the system opens a single OpenCV window `TMR 2026 - Vision Debug` whenever the mode is in `DISPLAY_MODES` (**VISION**, **AUTONOMOUS** or **PARKING**). The window is closed automatically when leaving those modes for STANDBY/MANUAL.
- Renderer lives in `main.py:_render_debug_view(mode_label)` and is shared by all display modes — do not duplicate it per mode.
- Layout: top half = BEV (left) + HSV white mask (right), bottom half = annotated frame with lane center line + YOLO bboxes.
- Two side-by-side overlay panels at y≈200: left = PID telemetry (`err`, `P/I/D`, `corr`, target servo angle, lidar); right = `OBJETOS DETECTADOS` list with up to 4 sign labels + confidence + distance, plus the action line (`-> ALTO total (5 s)`, etc.) from `SIGN_ACTIONS`.
- The bottom status bar shows the driving FSM state, or the ParkingFSM state when mode is PARKING.
- VISION mode brakes motors and centers steering, then *simulates* the PID purely for the overlay (servo never moves). `_set_mode` calls `pid.reset()` on entry/exit of VISION so the integrator does not contaminate AUTONOMOUS afterward.
- AUTONOMOUS mode does its normal work (FSM updates servo + motor) and additionally calls `_render_debug_view(mode_label="AUT")` after `_log_autonomous()`.

### Diagnostic preview tool: `tools/test_camera.py`
The "common test" entry point for camera/vision iteration. Imports CameraStream + LanePipeline + PIDController + SignDetector and renders the same overlay as `_render_debug_view` — but **never imports any GPIO hardware**, so it is safe to run with the systemd service active and on dev machines. Flags: `--no-yolo` skips loading the YOLO weights for instant startup. Exit with `q` or ESC.

### Alternative modules (exist but not wired into main.py)
These are full implementations kept for future wiring. Treat as library code:
- `hardware/camera_manager.py` — early IMX500 NPU prototype (COCO EfficientDet). **Superseded by `vision/imx500_detector.py`** (the production NPU path with the custom tmr_signs model); kept as reference only
- `hardware/motor_driver.py` — simpler lgpio-only motor (alternative to soft-start version)
- `vision/lane_detector.py` — classic ROI/threshold/histogram lane detector + crosswalk detection
- `vision/object_detector.py` — HSV traffic-light classifier + STOP distance via bbox + overtake/parking cues
- `control/gamepad_reader.py` — threaded gamepad reader at 100 Hz (main.py uses pygame directly)
- `autonomy/autonomous_mode.py` — advanced 9-state FSM (CROSSWALK_STOP, OVERTAKING_*, PARKING, OBSTACLE_HOLD)
- `autonomy/parking_maneuver.py` — Ackermann-based parallel parking sub-FSM

### Personal test scripts (do not wire into main.py)
- `vision_module.py` — user's standalone camera experiment with its own 9-state FSM and its own hazard/turn-signal implementation via `lgpio` chip 4. Pins come from `vision_config.yaml`.
- `test_gamepad.py`, `test_servo.py`, `test_vision.py` — diagnostics.
- `TMR2026/tools/test_camera.py` — official preview tool (camera + lane + PID + YOLO, no motors). See "Diagnostic preview tool" above.

## YOLO Models

- `TMR2026/weights/tmr_signs.pt` — active model loaded by `SignDetector` at `conf=0.55` (same as the validated simulator — 0.15 caused phantom detections that made the FSM brake randomly). All 7 classes are surfaced (`green, left, red, right, stop, straight, yellow`), but only `stop`/`red` gate the FSM (see brake gating above).
- `TMR2026/weights/tmr_signs_ncnn_model/` — NCNN export of the same model (FP16, imgsz=320), **preferred automatically** by `SignDetector._resolve_model_path()`. Committed to the repo so the Pi never has to export. Regenerate after retraining with `python tools/export_model.py` (works on PC or Pi; output is portable). Verified: identical labels/confidences to the `.pt` on dataset images.
- `TMR2026/weights/tmr_signs_imx500.rpk` — INT8 package for the IMX500 NPU (on-camera inference). NOT in the repo by default: it must be generated on the Pi with `python tools/export_imx500.py` (Sony's converter is Linux-only; quantization takes 15-60 min, once). Class order lives in `tmr_signs_imx500_labels.txt`. Tune `config.py:IMX500_CONF` after quantization. After retraining the model, regenerate BOTH exports (NCNN + rpk).
- `_legacy/runs/detect/train2/weights/` — source of the active model (checkpoint + training artifacts).
- `_legacy/runs/detect/train/weights/best.pt` — larger variant (~18 MB) kept as backup.
- `traffic_lights/` — Roboflow v9 dataset, 1470 images. **This entry used to say
  "close-up sign images, no track photos". That was wrong and it cost hours of
  work aimed at the wrong gap.** Measured 2026-07-26:
  - images are **320×240**, not 640×480;
  - they are **scale-model track photography** very close to this vehicle's own
    setup (track surface, lane lines, signs on stands, indoor floors), not
    close-ups;
  - ground-truth box heights are **14–116 px, median 45**. Scaled to the
    camera's 640×480 that is 28–232 px, median 90, versus the car's own pinhole
    range of 28 px @1.5 m → 154 px @0.27 m. So the set covers the **near** half
    of the car's range densely and **thins out past ~1 m**, which is where the
    car has to see first.
  Two consequences: training at imgsz 640 on 320×240 source is 2× interpolation,
  and inferring at imgsz 320 halves a 640×480 camera frame so the 28 px sign at
  1.5 m arrives as 14 px — under the dataset's own p10. That is the mechanism
  behind the measured 61 % @320 vs 78 % @640. Re-measure with
  `python TMR2026/tools/eval_hard.py` before trusting any claim about this set.
- **Sign-detector retraining**: `tools/train_signs.py` fine-tunes `tmr_signs.pt` with generalization augmentation. Flips are disabled on purpose (`fliplr=flipud=0`) — directional arrow classes (left/right/straight) would be mislabeled by mirroring. Auto-selects CUDA; this PC's torch is CPU-only (GTX 1650 unused until a CUDA wheel is installed).

## Learned steering (DriveNet, opt-in behavioral cloning)

An optional CNN replacement for the classic lane follower. `vision/drive_net.py:DriveNet` predicts `error_px` with the **same `.process()`/`.draw_debug()` contract as `LanePipeline`**, so it is a true drop-in — the FSM, PID, sign gating and lights are untouched. Enabled by `config.py:USE_DRIVE_NET` (default **False**); `main.py:_maybe_drive_net()` and `main_simulator.py` swap it in only if the flag is set AND `weights/drive_net.pt` exists, else they keep the classic pipeline (Sim2Real parity preserved).

- It is behavioral cloning: it **cannot train without `(image → error_px)` data**. Sources: the Unity sim and/or the classic pipeline as an auto-labeling teacher.
- Pipeline tools (all share one tub format `frames/ + labels.csv`): `tools/gen_synth_driving.py` (synthetic, no hardware), `tools/record_driving.py` (sim/camera/video/images), `tools/train_drive.py` (augmentation → `weights/drive_net.pt` + `.json` meta), `tools/test_drive_net.py` (eval, no GPIO), `tools/export_drive.py` (TorchScript/ONNX/NCNN).
- `weights/drive_net.*` and `TMR2026/datasets/` are gitignored (regenerable; the committed demo would be synthetic-only). Full workflow: `TMR2026/docs/DRIVE_NET.md`.

## Hard rules (don't break these)

- **Never modify `motor.brake()`** — it must remain an instantaneous hard-cut to 0.
- **Do NOT add a `cv2.COLOR_RGB2BGR` conversion after `capture_array()`** (this rule
  was the opposite until 2026-07-25, when it was measured to be wrong). Picamera2's
  `"RGB888"` names the 24-bit packing, not the channel order OpenCV sees: the array
  already arrives **BGR-ordered**. Measured against the printed red STOP sign on this
  camera — array as-is `H=9.4` (red), after `COLOR_RGB2BGR` `H=110.8` (cyan). The
  conversion inverted red and blue, so the STOP looked cyan, the red/purple blob
  fallback could never fire (it needs `H<=12` or `H>=165` and was handed 111), and the
  `red`/`green` traffic-light classes would have been swapped. White lane lines have
  `R=G=B`, so the lane pipeline was unaffected and hid the bug for a long time. If
  colours ever look inverted again, re-measure with a known-red object before
  changing this.
- **Never edit `vision_module.py`** — it's the user's personal camera experiment. Its hazard/turn-signal code is independent from the production `hardware/signals.py` module.
- **Never import from `_legacy/`** inside `TMR2026/`.
- **ESPERA state must use `time.monotonic()`**, not `time.sleep()` — the loop must keep serving the FSM.
- Turn-signal / hazard blink rate is `2 Hz` (per TMR regulation).
- **Steering inversion lives in `SteeringDriver.set_angle()` only** (driven by `config.py:STEERING_INVERTED`). Never re-invert in FSM, PID, signals, or per-mode code; always trust `current_angle` as the logical value.
- **`signals.tick()` must be called every main-loop iteration** (after `_run_mode()`), or LEDs freeze mid-blink.

## Vision tuning notes

**Both the camera exposure and the lane threshold are adaptive as of 2026-07-26. Do not re-pin either without reading why.** Measured on this track inside six hours: a pinned 33 ms / gain 4.0 gave 100 % lane confidence with 9.1 % mask fill at 14:00, and a completely empty 0.0 % mask by 15:40 after a 3.4× drop in scene brightness. Stock sensor AE is not the answer either — it targets mid-grey over a frame that is mostly dark plastic track, so it opens up until the red STOP octagon clips (measured S 23.7, 100 % of its pixels at 255, `stop` confidence exactly 0.000).

- **Exposure**: `vision/camera_stream.py:_adapt_exposure()` closes a loop on *both* objectives every 1.5 s (0.35 s while out of band) — enough light for the white lines, little enough clipping that the sign's red survives. Clipping outranks brightness, because a clipped sign is unrecoverable while a dark lane is not. Bounds live in `config.py:CAMERA_ADAPT_*`, each with the measurement behind it. Set `CAMERA_EXPOSURE_US` **and** `CAMERA_GAIN` to numbers to pin them and disable the loop. Verified recovering from a forced total blackout (V=0.0) in ~4 s. **The over-exposure branch has only been exercised by construction** — ambient light was too low to blow out the sensor even wide open — so re-test it in daylight.
- **Lane threshold**: `V_min` is derived from the bird's-eye view's own histogram (`LanePipeline._white_bounds()`), because the lines never stop being the brightest thing on a dark track, they just stop clearing a fixed number. Verified: same frame at full brightness and at 45 % returns the same error to within 1.6 px, where the fixed threshold gave 0 % confidence. `HSV_WHITE_LO`/`HSV_WHITE_HI` remain the **fallback floor** for the no-contrast case, and the `hsv_white_lo`/`hsv_white_hi` constructor args still shadow them per instance. Set `ADAPTIVE_WHITE = False` to go back to a fixed threshold.
  - The two guards in `_white_bounds()` are load-bearing. The percentile must beat the median by `MIN_CONTRAST`, or an adaptive threshold will happily turn sensor noise into a plausible-looking mask — and since it yields a constant mask fill by construction, it would silently defeat the degenerate-mask check in `tools/diag_track.py`, which is what caught a false 100 % lock before.
- `main_simulator.py` (Unity) still overrides to a brighter white (`[0,0,200]`/`[179,40,255]`) and pins `error_offset_px=0.0`. **Never re-tune the physical class defaults to the sim's values — that blinds the physical car.**
- `config.py:LANE_ERROR_OFFSET_PX` is a **mechanical** calibration, subtracted from the raw error before smoothing: with the car centred by hand the pipeline read a stable +44 px, which at 6.8 px/cm is 6.5 cm the follower would otherwise hold off-centre on purpose. Cause is the camera not sitting exactly on the chassis centreline plus an asymmetric BEV trapezoid. Re-measure after any camera remount: centre the car, run `tools/diag_track.py --frames 12`, put the reported mean here.
- Pick exposure manually with `tools/tune_exposure.py`, which sweeps settings and reports sign confidence, sign **saturation**, lane confidence and mask fill together. Rank by saturation, not confidence: among settings that served both, confidence spanned 0.761–0.772 (noise) while saturation spanned 46.6–102.4, and saturation is what predicts the blinding failure.
- Inspect the live mask via the top-right tile of `python main.py --display` (in VISION/AUTONOMOUS) or via `python tools/test_camera.py --no-yolo`.

## Known inconsistencies

- LED pins in `config.py` (`PIN_LED_TURN_LEFT=17`, `PIN_LED_TURN_RIGHT=5`, `PIN_LED_BRAKE=6`) and `vision_config.yaml` `gpio:` block belong to two separate programs: production `main.py` reads `config.py`; `vision_module.py` reads the YAML. They live in separate processes so there's no live conflict, but don't run both at once.
- `TMR2026/main.py` no longer hardcodes hardware constants — pins, servo angles and button mapping are imported from `config.py` (single source of truth). The PCA9685 servo channel (15) and ToF XSHUT pins are read by the drivers directly from `config.py`.

## Common Pi-side gotchas

- `lgpio.error: 'GPIO not allocated'` on `python main.py` means the systemd service is holding pins. `TMR2026/main.py:_release_gpio_from_systemd()` now detects this on startup and runs `sudo -n systemctl stop carrito_tmr` automatically (passwordless sudo is configured for `angel01`). The function skips itself when launched *by* systemd (`INVOCATION_ID` env var is set), so the service can still run normally at boot.
- Old folders from the pre-reorg layout (`AUTO_YOLO/`, `CAMARA/`, `CONTROL/`, …) may need `sudo rm -rf` if they were created under root by a prior `sudo` run.
