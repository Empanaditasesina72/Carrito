"""Physical braking experiment with lane following (Raspberry Pi 5 + real car).

RUN THIS ON THE PI, WITH THE CAR. It drives the car straight at a low fixed
speed toward a STOP sign and lets the REAL controller (AutonomousFSM + PID +
ToF) brake it, then logs the final stopping distance. Repeats for N trials so
the paper can report mean +/- std and a success rate instead of a single run,
and compare against the simulator (SIL stopped at 292.5 mm, setpoint 270 mm).

This is the physical counterpart of Test 2 (P2). Steering is CLOSED-LOOP by
default: the lane pipeline drives the PID and the car keeps itself centred on the
approach, matching how the simulator runs it. Pass --straight to force the wheels
centred instead, which isolates the braking controller but lets the car drift.

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
    LANE_RIGHT_BIAS,
)
from hardware.motor import MotorDriver
from hardware.steering_driver import SteeringDriver
from hardware.distance_sensor import DistanceSensor, make_distance_sensor
from control.pid_controller import PIDController
from control.fsm import AutonomousFSM, FSMState

CAMERA_W, CAMERA_H, CAMERA_FPS = 640, 480, 30
# Lane-following gains come from config.py -- ONE definition. These three
# entry points used to redefine them locally (0.08/0.002/0.025), agreeing
# with each other but not with config's 0.09, so tuning config changed
# nothing on the car.
from config import STEER_KP as PID_KP, STEER_KI as PID_KI, STEER_KD as PID_KD
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


def run_trial(fsm, camera, sign_det, sensor, cruise_pwm, max_drive_s,
              kick_pwm=0.0, kick_s=0.25, lane_pipe=None) -> dict:
    fsm.MAX_AUTO_PWM = float(cruise_pwm)      # cap cruise speed for safety
    # FLOOR, not just a fraction. min(20, 25*0.6)=15 % put PRECAUCION below this
    # motor's stall threshold, so the instant the car saw the sign it slowed to a
    # standstill, lost the flickering detection, resumed cruise, saw it again --
    # the CRUCERO<->PRECAUCION ping-pong that filled the 2026-07-27 logs, with
    # every trial timing out mid-track. 20 % sustains motion once rolling.
    fsm.PRECAUCION_PWM = max(20.0, cruise_pwm * 0.8)
    fsm.activate()

    # The kick used to run BEFORE activate(), as motor.kick() blocking for its
    # whole duration -- so the most violent 0.25 s of the run, a 60 % launch,
    # happened with NO steering control at all. Any trim bias yawed the car
    # right at the start, and the controller then began the run already carrying
    # a heading error it can barely observe (the pipeline measures lateral
    # offset, not heading). Trial photos showed exactly that: the car diagonal
    # across the lane. Now the kick is a phase INSIDE the control loop: each
    # tick re-applies the kick duty (kick() with seconds=0 bypasses the
    # soft-start ramp without sleeping), while fsm.update() keeps steering.
    kick_until = time.monotonic() + kick_s if kick_pwm > 0 else 0.0

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

        frame = camera.get_frame()
        if sign_det is not camera and frame is not None:
            sign_det.update_frame(frame)

        front = sensor.front_mm if sensor is not None else None
        if front is not None:
            min_mm = min(min_mm, front)
            if front < EMERGENCY_STOP_MM:        # hard safety cutoff
                fsm.motor.brake()
                reason = "tof_emergency"
                break

        # Closed-loop steering when a lane pipeline is supplied. Forcing the error
        # to zero "isolates" the braking controller in principle, but the vehicle
        # cannot hold a line open-loop -- it drifts out of the lane before braking
        # matters, which measures nothing. Letting the lane follower correct the
        # drift is both what the production system does and the only way the run
        # stays on a 3.46 m track.
        if lane_pipe is not None and frame is not None:
            lane = lane_pipe.process(frame)
            fsm.lane_error = lane.error_px
            fsm.lane_conf = lane.confidence
            fsm.lane_heading = lane.heading
        else:
            fsm.lane_error = 0.0
            fsm.lane_conf = 1.0
            fsm.lane_heading = 0.0
        fsm.lidar_mm = front
        fsm.sign_visible = (sign_det.has_sign("stop_sign") or sign_det.has_sign("red"))
        closest = sign_det.closest_sign("stop_sign") or sign_det.closest_sign("red")
        fsm.sign_distance_mm = (closest.distance_m * 1000.0
                                if closest and closest.distance_m else None)
        fsm.update(dt)

        # Never once braking has begun: brake() is an instantaneous hard-cut and
        # a late kick tick must not override it. But CRUCERO alone was too
        # strict. On a short track the sign is already in frame at the start
        # line, so the FSM enters PRECAUCION on the very first tick and the kick
        # never fired at all -- leaving the car to break static friction on
        # PRECAUCION_PWM alone, which is a floor sized to SUSTAIN motion, not to
        # start it. PRECAUCION still drives (set_speed), so kicking through it is
        # safe and keeps the kick time-bounded by kick_until either way.
        if now < kick_until and fsm.state in (FSMState.CRUCERO,
                                              FSMState.PRECAUCION):
            fsm.motor.kick(kick_pwm, 0.0)

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
        # FRENADO is also excluded: a brake decided at t=2.99 must not be
        # reclassified as max_drive by the cap firing one tick later.
        if (fsm.state not in (FSMState.FRENADO, FSMState.ESPERA)
                and (now - t0) >= max_drive_s):
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
    ap.add_argument("--max-drive", type=float, default=8.0,
                    help="seconds the car may move before being braked anyway")
    ap.add_argument("--straight", action="store_true",
                    help="force zero lane error instead of following the lane. "
                         "Open loop: the car will drift, use only with the wheels "
                         "off the ground")
    ap.add_argument("--kick", type=float, default=0.0,
                    help="breakaway pulse %% PWM before each trial, for a loaded "
                         "car that stalls at the cruise duty (try 60)")
    ap.add_argument("--kick-ms", type=float, default=250.0,
                    help="how long the breakaway pulse lasts")
    ap.add_argument("--no-prompt", action="store_true",
                    help="never wait on stdin: run the trials back to back and "
                         "leave stopped_mm blank for an operator to fill in. Lets "
                         "the run be triggered over SSH.")
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

    lane_pipe = None
    if not args.straight:
        from vision.lane_pipeline import LanePipeline
        calib = {}
        try:
            import json
            with open(ROOT / "track_calib.json", "r", encoding="utf-8") as f:
                calib = json.load(f)
            print(f"  lane calibration: {dict(calib)}")
        except FileNotFoundError:
            print("  lane calibration: none (config defaults)")
        except Exception as e:
            print(f"  lane calibration: unreadable ({e}); using defaults")
        lane_pipe = LanePipeline(
            frame_w=CAMERA_W, frame_h=CAMERA_H, debug=False,
            right_bias=calib.get("right_bias", LANE_RIGHT_BIAS),
            roi_frac=calib.get("roi_frac", 0.5),
            hsv_white_lo=calib.get("hsv_white_lo"),
            hsv_white_hi=calib.get("hsv_white_hi"),
        )
        print("  steering: CLOSED LOOP on the lane (like the simulator)")
    else:
        print("  steering: forced straight, OPEN LOOP - the car will drift")

    motor = MotorDriver(pin_rpwm=PIN_MOTOR_RPWM, pin_lpwm=PIN_MOTOR_LPWM)
    steering = SteeringDriver()
    sensor = None if args.manual else make_distance_sensor()
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

    # The CSV is written INCREMENTALLY, one row per trial, fsync'd. It used to be
    # written once at the very end, so anything that killed the process before
    # that point -- an uncaught exception, a dropped SSH session, a power cut --
    # discarded every measurement the operator had already typed in. That is not
    # hypothetical: the first real 10-trial session on 2026-07-26 ended with no
    # CSV on disk and the numbers surviving only in a terminal scrollback.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = ["trial", "stopped_mm", "min_mm", "overshoot", "within_tol",
                  "duration_s", "stop_reason", "source"]
    csv_f = open(args.out, "w", newline="", encoding="utf-8")
    csv_w = csv.DictWriter(csv_f, fieldnames=fieldnames)
    csv_w.writeheader()
    csv_f.flush()

    def persist(row: dict) -> None:
        csv_w.writerow(row)
        csv_f.flush()
        os.fsync(csv_f.fileno())

    results = []
    try:
        for k in range(1, args.trials + 1):
            if args.no_prompt:
                print(f"\n[Trial {k}/{args.trials}] starting in 3 s - stand clear")
                time.sleep(3.0)
            else:
                input(f"\n[Trial {k}/{args.trials}] Place car at the start line, "
                      f"press Enter (Ctrl+C to stop)...")
            r = run_trial(fsm, camera, sign_det, sensor, args.cruise,
                          args.max_drive, args.kick, args.kick_ms / 1000.0,
                          lane_pipe)
            r["trial"] = k
            if sensor is None and not args.no_prompt:
                print(f"  -> the car stopped ({r['stop_reason']}, "
                      f"{r['duration_s']} s). Now measure it.")
                m = ask_measured_mm(k)
                r["stopped_mm"] = round(m, 1) if m is not None else ""
                r["within_tol"] = (abs(m - STOP_TARGET_MM) <= STOP_TOLERANCE_MM
                                   if m is not None else "")
            print(f"  -> stopped {r['stopped_mm']} mm  (min {r['min_mm']} mm, "
                  f"within tol: {r['within_tol']}, {r['stop_reason']})")
            # THE data-loss bug of the first real session: this append (and any
            # write) was simply missing, so every trial was measured, printed and
            # then dropped -- the end-of-run block wrote an empty list. Each row
            # now also goes to disk immediately.
            results.append(r)
            persist(r)
    except KeyboardInterrupt:
        print("\n[BRAKE] Aborted by user.")
    finally:
        csv_f.close()
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
        # Rows are already on disk (persisted per trial); this is summary only.
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
