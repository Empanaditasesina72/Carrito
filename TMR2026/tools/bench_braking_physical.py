"""Physical straight-line braking experiment (Raspberry Pi 5 + real car).

RUN THIS ON THE PI, WITH THE CAR. It drives the car straight at a low fixed
speed toward a STOP sign and lets the REAL controller (AutonomousFSM + PID +
ToF) brake it, then logs the final stopping distance. Repeats for N trials so
the paper can report mean +/- std and a success rate instead of a single run,
and compare against the simulator (SIL stopped at 292.5 mm, setpoint 270 mm).

This is the physical counterpart of Test 2 (P2). Steering is held straight,
so it isolates the braking controller and is safe on a short straight track.

MANUAL MODE (--manual): run the experiment WITHOUT the ToF sensors. The car
still brakes on the real camera path (the sign detector feeds the FSM exactly as
in production); only the *measurement* changes -- you measure the final gap to
the sign with a tape measure and type it in. Use this when the ToF is
unavailable. A tape measure is arguably better evidence for the paper than an
uncalibrated sensor, but the trade-off is real and must be stated: without the
ToF there is no automatic emergency cutoff, so --max-drive bounds how long the
car may move before it is braked regardless of what the camera saw.

>>> SAFETY - read before running <<<
  * FIRST run with the drive wheels OFF THE GROUND to confirm behaviour.
  * Low speed only (default cruise 25 % PWM).
  * Emergency ToF cutoff: brakes hard if the front sensor reads < 120 mm.
    NOT AVAILABLE in --manual mode; --max-drive is the only automatic stop.
  * Press Ctrl+C at any time -> immediate brake + exit.
  * Keep a hand ready to catch the car. Supervise every trial.

Setup per trial:
  * A short straight (>= ~1.2 m) with a STOP sign (red) at the end.
  * Mark a start line; place the car there each trial.

Output:
  validation_results/braking_physical.csv
    columns: trial, stopped_mm, min_mm, overshoot, within_tol, duration_s,
             stop_reason, source

Usage (from TMR2026/):
  python tools/bench_braking_physical.py --trials 10 --cruise 25
  python tools/bench_braking_physical.py --manual --trials 10 --max-drive 3.0
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    PIN_MOTOR_RPWM, PIN_MOTOR_LPWM,
    SERVO_CENTER_ANGLE, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE,
    STOP_TARGET_MM, STOP_TOLERANCE_MM, EMERGENCY_STOP_MM,
    USE_IMX500_NPU, IMX500_RPK_PATH, IMX500_LABELS_PATH, IMX500_CONF,
)
from hardware.motor import MotorDriver
from hardware.steering_driver import SteeringDriver
from hardware.distance_sensor import DistanceSensor
from control.pid_controller import PIDController
from control.fsm import AutonomousFSM, FSMState

CAMERA_W, CAMERA_H, CAMERA_FPS = 640, 480, 30
PID_KP, PID_KI, PID_KD = 0.08, 0.002, 0.025
LOOP_HZ = 50
TRIAL_TIMEOUT_S = 25.0


def _build_vision():
    if USE_IMX500_NPU and os.path.isfile(IMX500_RPK_PATH):
        try:
            from vision.imx500_detector import IMX500CameraStream
            npu = IMX500CameraStream(
                rpk_path=IMX500_RPK_PATH, labels_path=IMX500_LABELS_PATH,
                width=CAMERA_W, height=CAMERA_H, fps=CAMERA_FPS, conf=IMX500_CONF)
            return npu, npu
        except Exception as e:
            print(f"[BRAKE] NPU unavailable ({e}) - CPU path.")
    from vision.camera_stream import CameraStream
    from vision.sign_detector import SignDetector
    cam = CameraStream(width=CAMERA_W, height=CAMERA_H, fps=CAMERA_FPS)
    sign = SignDetector(model_path="weights/tmr_signs.pt", conf=0.55, imgsz=320)
    return cam, sign


def run_trial(fsm, camera, sign_det, sensor, cruise_pwm, max_drive_s) -> dict:
    fsm.MAX_AUTO_PWM = float(cruise_pwm)      # cap cruise speed for safety
    fsm.PRECAUCION_PWM = min(fsm.PRECAUCION_PWM, cruise_pwm * 0.6)
    fsm.activate()

    t0 = time.monotonic()
    t_last = t0
    min_mm = 1e9
    stopped_readings = []
    espera_ticks = 0
    reason = "timeout"

    while time.monotonic() - t0 < TRIAL_TIMEOUT_S:
        now = time.monotonic()
        dt = now - t_last
        t_last = now

        if sign_det is not camera and camera.get_frame() is not None:
            sign_det.update_frame(camera.get_frame())

        front = sensor.front_mm if sensor is not None else None
        if front is not None:
            min_mm = min(min_mm, front)
            if front < EMERGENCY_STOP_MM:        # hard safety cutoff
                fsm.motor.brake()
                reason = "tof_emergency"
                break

        # straight-line braking test: force straight steering, real speed/brake logic
        fsm.lane_error = 0.0
        fsm.lane_conf = 1.0
        fsm.lidar_mm = front
        fsm.sign_visible = (sign_det.has_sign("stop_sign") or sign_det.has_sign("red"))
        closest = sign_det.closest_sign("stop_sign") or sign_det.closest_sign("red")
        fsm.sign_distance_mm = (closest.distance_m * 1000.0
                                if closest and closest.distance_m else None)
        fsm.update(dt)

        if fsm.state in (FSMState.ESPERA,):       # the controller has stopped
            espera_ticks += 1
            if front is not None and front < 1000:
                stopped_readings.append(front)
            if sensor is None:
                if espera_ticks >= 15:            # ~0.3 s settled, nothing to sample
                    reason = "braked"
                    break
            elif len(stopped_readings) >= 15:     # ~0.3 s of stopped samples
                reason = "braked"
                break

        # Without the ToF this is the only automatic stop. Bound how far the car
        # may travel when the sign is never detected, so it cannot run off the
        # track while the loop waits on a detection that is not coming.
        if fsm.state not in (FSMState.ESPERA,) and (now - t0) >= max_drive_s:
            fsm.motor.brake()
            reason = "max_drive"
            break

        time.sleep(max(0.0, (1.0 / LOOP_HZ) - (time.monotonic() - now)))

    fsm.deactivate()

    stopped = statistics.median(stopped_readings) if stopped_readings else None
    return {
        "stopped_mm": round(stopped, 1) if stopped else "",
        "min_mm": round(min_mm, 1) if min_mm < 1e9 else "",
        "overshoot": (min_mm < STOP_TARGET_MM - 2.5 * STOP_TOLERANCE_MM) if min_mm < 1e9 else "",
        "within_tol": (abs(stopped - STOP_TARGET_MM) <= STOP_TOLERANCE_MM) if stopped else "",
        "duration_s": round(time.monotonic() - t0, 2),
        "stop_reason": reason,
        "source": "tof" if sensor is not None else "tape",
    }


def ask_measured_mm(trial: int):
    """Prompt for a tape-measure reading. Blank marks the trial as failed."""
    while True:
        raw = input(f"  [{trial}] Measured gap car-to-sign in mm "
                    f"(Enter = trial failed): ").strip()
        if raw == "":
            return None
        try:
            v = float(raw.replace(",", "."))
        except ValueError:
            print("      Not a number. Type millimetres, e.g. 285")
            continue
        if not 0 < v < 3000:
            print("      Out of range. Expected roughly 50-1500 mm.")
            continue
        return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--cruise", type=float, default=25.0, help="cruise PWM %% (keep low)")
    ap.add_argument("--manual", action="store_true",
                    help="no ToF: brake on the camera, measure with a tape")
    ap.add_argument("--max-drive", type=float, default=3.0,
                    help="seconds the car may move before being braked anyway")
    ap.add_argument("--out", default=str(ROOT / "validation_results" / "braking_physical.csv"))
    args = ap.parse_args()

    print("=" * 60)
    print("  PHYSICAL BRAKING EXPERIMENT (P2)  -  car will MOVE")
    print(f"  setpoint {STOP_TARGET_MM:.0f} mm, tolerance +/-{STOP_TOLERANCE_MM:.0f} mm, "
          f"cruise {args.cruise:.0f}% PWM")
    if args.manual:
        print("  MODE: manual  -  no ToF, no automatic emergency cutoff.")
        print(f"  The car is braked after {args.max_drive:.1f} s no matter what.")
        print("  Measure the gap with a tape after each trial and type it in.")
    print("  Ctrl+C = emergency brake + exit")
    print("=" * 60)

    motor = MotorDriver(pin_rpwm=PIN_MOTOR_RPWM, pin_lpwm=PIN_MOTOR_LPWM)
    steering = SteeringDriver()
    sensor = None if args.manual else DistanceSensor()
    camera, sign_det = _build_vision()
    pid = PIDController(kp=PID_KP, ki=PID_KI, kd=PID_KD, setpoint=0.0,
                        output_limits=(-(SERVO_CENTER_ANGLE - SERVO_MIN_ANGLE),
                                       (SERVO_MAX_ANGLE - SERVO_CENTER_ANGLE)),
                        integral_limits=(-25.0, 25.0))
    fsm = AutonomousFSM(motor, steering, pid)

    if sensor is not None:
        sensor.start()
    camera.start()
    if sign_det is not camera:
        sign_det.start()
    steering.center()

    results = []
    try:
        for k in range(1, args.trials + 1):
            input(f"\n[Trial {k}/{args.trials}] Place car at the start line, "
                  f"press Enter (Ctrl+C to stop)...")
            r = run_trial(fsm, camera, sign_det, sensor, args.cruise, args.max_drive)
            r["trial"] = k
            if sensor is None:
                print(f"  -> the car stopped ({r['stop_reason']}, "
                      f"{r['duration_s']} s). Now measure it.")
                m = ask_measured_mm(k)
                r["stopped_mm"] = round(m, 1) if m is not None else ""
                r["within_tol"] = (abs(m - STOP_TARGET_MM) <= STOP_TOLERANCE_MM
                                   if m is not None else "")
            print(f"  -> stopped {r['stopped_mm']} mm  (min {r['min_mm']} mm, "
                  f"within tol: {r['within_tol']}, {r['stop_reason']})")
    except KeyboardInterrupt:
        print("\n[BRAKE] Aborted by user.")
    finally:
        motor.brake()
        steering.center()
        time.sleep(0.1)
        camera.stop()
        if sign_det is not camera:
            sign_det.stop()
        if sensor is not None:
            sensor.stop()
        motor.cleanup()

    if results:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["trial", "stopped_mm", "min_mm",
                                              "overshoot", "within_tol", "duration_s",
                                              "stop_reason", "source"])
            w.writeheader()
            w.writerows(results)
        dists = [r["stopped_mm"] for r in results if isinstance(r["stopped_mm"], (int, float))]
        oks = sum(1 for r in results if r["within_tol"] is True)
        braked = sum(1 for r in results if r["stop_reason"] == "braked")
        capped = sum(1 for r in results if r["stop_reason"] == "max_drive")
        print("\n" + "=" * 60)
        print(f"  measurement source : {results[0]['source']}")
        print(f"  braked on the sign : {braked}/{len(results)}")
        if capped:
            print(f"  hit --max-drive    : {capped}/{len(results)}  "
                  f"(sign never detected -- exclude these from the braking stats)")
        if dists:
            print(f"  trials with a stop : {len(dists)}/{len(results)}")
            print(f"  stopping distance  : {statistics.mean(dists):.1f} +/- "
                  f"{statistics.pstdev(dists):.1f} mm  (setpoint {STOP_TARGET_MM:.0f})")
            print(f"  within +/-{STOP_TOLERANCE_MM:.0f} mm : {oks}/{len(results)} "
                  f"({100*oks/len(results):.0f}%)")
            print(f"  vs SIL (292.5 mm)  : diff "
                  f"{statistics.mean(dists) - 292.5:+.1f} mm")
        print(f"  CSV: {args.out}")
        print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
