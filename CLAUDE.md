# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current focus & roadmap (updated 2026-07-07 — read this first when resuming)

**Goal:** make the car drive any basic road-like ("carretera") track *and* read its signs, with both models running on the Pi for the TMR competition.

Two learned models, two destinations on the Pi:
- **Sign detection** (YOLO `tmr_signs`) → runs on the **IMX500 NPU** (inside the camera, ~0 % CPU).
- **Steering** (`DriveNet`, behavioral cloning) → runs on the **Pi CPU** (tiny, ~1 MB). The IMX500 holds **one** model at a time, so it stays on signs; DriveNet never shares the NPU.

**Train here, deploy there** (the NPU and the Pi CANNOT train — the IMX500 is inference-only and the Pi 5 has no CUDA GPU):

| Stage | Machine |
|---|---|
| **TRAIN** (`.pt`) | **PC + GTX 1650** — CUDA is set up: `torch 2.12.0+cu126` |
| Move weights | git: PC commits → `push` → Pi `pull` (or `git push pi main` over LAN — remote `pi`, much faster than the Pi pulling GitHub) |
| **Convert** `.pt`→`.rpk` | **PC WSL Ubuntu** (quantize) + **Pi** (`imx500-package`). The Pi can NO LONGER run the full converter: Sony's `uni-pytorch` needs Python <3.13 and the Pi has 3.13.5. See "IMX500 conversion (2026-07-07 procedure)" below. |
| **Infer** | **Pi + IMX500** (signs) · **Pi CPU** (DriveNet) |

**Done so far (this machine):**
- Built the full **DriveNet behavioral-cloning pipeline** (opt-in, `config.py:USE_DRIVE_NET=False`): `vision/drive_net.py` + tools `gen_synth_driving` / `record_driving` / `train_drive` / `test_drive_net` / `export_drive`, plus `tools/train_signs.py`. Docs: `TMR2026/docs/DRIVE_NET.md`. Commits `1b2c9b9`, `25010f0`.
- **Configured CUDA** on the PC GTX 1650: `pip install torch==2.12.0+cu126 torchvision==0.27.0+cu126 --index-url https://download.pytorch.org/whl/cu126`. `torch.cuda.is_available()==True`. (cu128 has no torch 2.12 build; cu126 is the right index for Python 3.14.)
- **✅ Retrained the sign detector on the GPU** (`tools/train_signs.py`, data in `traffic_lights/`). Early-stopped at epoch 92 (best 62): val mAP@50 0.995, mAP@50-95 0.647, all 7 classes recall 1.0; held-out test @conf 0.55 → P 99.3 % / R 98.6 % / F1 99.0 %. Deployed `best.pt` → `weights/tmr_signs.pt` + regenerated NCNN. Commit `5fdb844`. **The `SignDetector(conf=0.55)` threshold is confirmed still optimal.** (Used `traffic_lights/data_local.yaml`, a gitignored copy of `data.yaml` with an absolute `path:` — the Roboflow `../train/images` resolves to the repo root, wrong.)
- **✅ DriveNet GPU path validated + baseline trained.** Synthetic 4000/1000, `--workers 4` → best val_RMSE 14.7 px, eval RMSE 10.6 px. The `drive_net.pt` is synthetic-only (gitignored, NOT deployed; `USE_DRIVE_NET` stays False until real-data training). Also **fixed `train_drive.py`** so `--workers>0` works on Windows (moved `TubDataset` to module level — `<locals>` classes can't be pickled by spawn). Commit `e405e70`. Capture plan: `TMR2026/docs/DRIVE_NET_CAPTURE_PLAN.md` (commit `bf02372`).

- **✅ Detector deployed to the Pi + `.rpk` generated (2026-07-07).** SSH access to the Pi works from this PC (`ssh angel01@192.168.1.71`, key auth; repo at `~/Carrito`). Pi synced to `9b163a6` via LAN push (remote `pi`; the Pi's GitHub pull was crawling). The `.rpk` is installed at `~/Carrito/TMR2026/weights/tmr_signs_imx500.rpk` (3 MB) + labels file; `USE_IMX500_NPU=True` so the NPU path activates on the next `main.py` start. NOT yet smoke-tested on the car.

**IMX500 conversion (2026-07-07 procedure — `tools/export_imx500.py` on the Pi is BROKEN, Python 3.13):**
1. On the PC, WSL Ubuntu has the toolchain ready: venv `~/imx_venv` (Python 3.12, torch CPU, ultralytics, model-compression-toolkit, imx500-converter[pt]) + portable JRE at `~/jre`. Calibration yaml: `traffic_lights/data_wsl.yaml` (gitignored, WSL-absolute `path:`).
2. Run the export in WSL (quantizes INT8 + converts, ~18 min): loads `weights/tmr_signs.pt`, `model.export(format="imx", data=data_wsl.yaml, fraction=0.25)` → `weights/tmr_signs_imx_model/packerOut.zip` (+ `labels.txt`). Ultralytics does NOT produce the `.rpk` itself (that needs `imx500-package`, only packaged for Pi OS).
3. `scp packerOut.zip labels.txt angel01@192.168.1.71:/tmp/`, then on the Pi: `imx500-package -i /tmp/packerOut.zip -o /tmp/rpk_out` (seconds; needs apt `imx500-tools` + `default-jre`, already installed) → copy `network.rpk` to `~/Carrito/TMR2026/weights/tmr_signs_imx500.rpk` and labels to `tmr_signs_imx500_labels.txt`.

**Next steps:**
1. **Smoke-test the NPU on the car:** `python main.py --display`, enter VISION, expect `[VISION] Backend: IMX500 NPU (on-chip inference)`; tune `config.py:IMX500_CONF` (0.55) on track.
2. **Capture real DriveNet driving data** (none exists yet — the only blocker for the steering model). Follow `TMR2026/docs/DRIVE_NET_CAPTURE_PLAN.md`: Pi camera (`capture_track.py` → `record_driving.py --source images`) or Unity sim → train on GPU here (`train_drive.py --device cuda --workers 4`) → fine-tune over the synthetic baseline → deploy to Pi CPU → set `USE_DRIVE_NET=True`.

**Standing decision:** all training stays on the PC (GPU); the Pi is for conversion, on-track testing and running the car.

## Active system: TMR2026/

Everything under `TMR2026/` is the current vehicle. Legacy prototypes live in `_legacy/` and must not be imported from TMR2026. Project-wide docs (architecture diagram + generator) live in `docs/`; TMR2026-specific docs (SETUP, Sim2Real protocol, calibration, professor deliveries) live in `TMR2026/docs/`.

**`config.py` is the single source of truth** for GPIO pins, servo angles/limits, PID gains, speeds and gamepad button mapping. `main.py`, `main_simulator.py` and `control/fsm.py` import these values — never re-hardcode them per file.

**Root `main.py` is a loader** — it `chdir`s into `TMR2026/` and runs `TMR2026/main.py` with `runpy` so imports like `from hardware.motor import MotorDriver` keep working. The systemd service (`TMR2026/systemd/carrito_tmr.service`) points directly to `TMR2026/main.py --display` and starts under `graphical.target` (i.e. after the desktop is ready), with `DISPLAY=:0` and `XAUTHORITY=/home/angel01/.Xauthority` exported, so OpenCV can open a window on the HDMI monitor when VISION/AUTONOMOUS mode is entered. Root `main.py` is only for manual execution.

### Hardware Target

Raspberry Pi 5 with:
- Sony IMX500 NPU camera via `Picamera2` (RGB888 → BGR via `cv2.cvtColor(RGB2BGR)` — must preserve)
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

- HSV white filter in `vision/lane_pipeline.py` is **configurable per instance** via the `hsv_white_lo` / `hsv_white_hi` constructor args (they shadow the class attrs). The **class default targets the physical track under medium-low light** (e.g. phone flashlight, not just direct lamp): `HSV_WHITE_LO = [0, 0, 130]`, `HSV_WHITE_HI = [179, 60, 255]`. `main.py` (Pi) and `tools/test_camera.py` use this default; `main_simulator.py` (Unity) overrides to a much brighter white (`[0,0,200]`/`[179,40,255]`) because the sim's lines are pure-white on a dark floor. So Pi and sim share the whole algorithm but each gets the right white threshold. If the black plastic track leaks into the mask under bright light, raise `V_min` toward 150–160; if dim conditions miss the lines, lower `V_min` toward 100–110. **Never re-tune the class default to the sim's bright values — that blinds the physical car.**
- Inspect the live mask via the top-right tile of `python main.py --display` (in VISION/AUTONOMOUS) or via `python tools/test_camera.py --no-yolo`.

## Known inconsistencies

- LED pins in `config.py` (`PIN_LED_TURN_LEFT=17`, `PIN_LED_TURN_RIGHT=5`, `PIN_LED_BRAKE=6`) and `vision_config.yaml` `gpio:` block belong to two separate programs: production `main.py` reads `config.py`; `vision_module.py` reads the YAML. They live in separate processes so there's no live conflict, but don't run both at once.
- `TMR2026/main.py` no longer hardcodes hardware constants — pins, servo angles and button mapping are imported from `config.py` (single source of truth). The PCA9685 servo channel (15) and ToF XSHUT pins are read by the drivers directly from `config.py`.

## Common Pi-side gotchas

- `lgpio.error: 'GPIO not allocated'` on `python main.py` means the systemd service is holding pins. `TMR2026/main.py:_release_gpio_from_systemd()` now detects this on startup and runs `sudo -n systemctl stop carrito_tmr` automatically (passwordless sudo is configured for `angel01`). The function skips itself when launched *by* systemd (`INVOCATION_ID` env var is set), so the service can still run normally at boot.
- Old folders from the pre-reorg layout (`AUTO_YOLO/`, `CAMARA/`, `CONTROL/`, …) may need `sudo rm -rf` if they were created under root by a prior `sudo` run.
