#!/usr/bin/env python3
"""Closed-loop straight-line test: does the car actually hold the lane?

This is the acceptance test the project never had. Every other tool measures a
piece -- the lane lock (diag_track), the steering direction (verify_steering),
the trim (auto_trim) -- but none of them answers the only question that matters:
put the car on the track, let the real controller drive, does it go straight.

Runs the production AutonomousFSM with the sign gate forced off, so the car stays
in CRUCERO for the whole run and nothing brakes but the timer. Logs the lane error
every tick and reports the three numbers that decide it:

  mean   standing offset. Non-zero means the car settles off-centre -- either
         SERVO_TRIM_DEG is still wrong or LANE_ERROR_OFFSET_PX is.
  std    weave. Large with a small mean means the gain is too high, or the aim
         row (LANE_AIM_WINDOW_FRAC) is too close.
  drift  px/s of steady drift. Non-zero means the controller is losing to a bias it
         cannot overcome -- go back to auto_trim.

SAFETY
  * First run with the drive wheels OFF THE GROUND.
  * Keep a hand ready. Ctrl+C brakes and exits immediately.
  * --seconds bounds the run; there is no other automatic stop.

Usage (from TMR2026/):
    python tools/drive_straight.py                 # 4 s at 25 % duty
    python tools/drive_straight.py --seconds 6 --cruise 22
    python tools/drive_straight.py --runs 3        # repeat, one summary each
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (PIN_MOTOR_RPWM, PIN_MOTOR_LPWM, STEER_KP, STEER_KI, STEER_KD,
                    SERVO_CENTER_ANGLE, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE,
                    SERVO_TRIM_DEG, LANE_ERROR_OFFSET_PX, LANE_RIGHT_BIAS,
                    LANE_DEADBAND_PX, STEER_HEADING_GAIN)
from hardware.motor import MotorDriver
from hardware.steering_driver import SteeringDriver
from vision.camera_stream import CameraStream
from vision.lane_pipeline import LanePipeline
from control.pid_controller import PIDController
from control.fsm import AutonomousFSM, FSMState

LOOP_HZ = 50
PX_PER_CM = 6.796


def run_once(fsm, cam, lane, steering, cruise, kick, seconds):
    """One straight run. Returns per-tick samples."""
    fsm.MAX_AUTO_PWM = float(cruise)
    fsm.activate()

    kick_until = time.monotonic() + 0.25 if kick > 0 else 0.0
    t0 = last = time.monotonic()
    samples = []

    try:
        while time.monotonic() - t0 < seconds:
            now = time.monotonic()
            dt = max(1e-3, now - last)
            last = now

            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            r = lane.process(frame)

            fsm.lane_error   = r.error_px
            fsm.lane_conf    = r.confidence
            fsm.lane_heading = r.heading
            fsm.lidar_mm     = None
            fsm.sign_visible = False        # stay in CRUCERO; only the timer stops us
            fsm.sign_distance_mm = None
            fsm.update(dt)

            if now < kick_until and fsm.state == FSMState.CRUCERO:
                fsm.motor.kick(kick, 0.0)

            samples.append((now - t0, r.error_px, r.confidence,
                            steering.current_angle, r.heading))
            time.sleep(max(0.0, (1.0 / LOOP_HZ) - (time.monotonic() - now)))
    finally:
        fsm.motor.brake()
        fsm.deactivate()
    return samples


def report(samples, idx, total) -> dict:
    if len(samples) < 8:
        print(f"  corrida {idx}/{total}: solo {len(samples)} muestras -- invalida")
        return {}

    a = np.asarray(samples, dtype=float)
    # Skip the launch transient: the kick and the first corrections are not
    # steady-state behaviour and would dominate a 4 s average.
    keep = a[a[:, 0] >= 0.8]
    if len(keep) < 8:
        keep = a
    t, err, conf, ang = keep[:, 0], keep[:, 1], keep[:, 2], keep[:, 3]
    drift = float(np.polyfit(t, err, 1)[0])

    print(f"\n  --- corrida {idx}/{total} ({len(a)} muestras, "
          f"{a[-1,0]:.1f} s) ---")
    print(f"  error medio : {err.mean():+7.1f} px  ({err.mean()/PX_PER_CM:+5.1f} cm)"
          f"   <- desviacion permanente")
    print(f"  desviacion  : {err.std():7.1f} px  ({err.std()/PX_PER_CM:5.1f} cm)"
          f"   <- serpenteo")
    print(f"  deriva      : {drift:+7.1f} px/s ({drift/PX_PER_CM:+5.1f} cm/s)"
          f"   <- sesgo no vencido")
    print(f"  servo       : medio {ang.mean():6.2f} deg  "
          f"rango {ang.min():.1f}..{ang.max():.1f}")
    print(f"  confianza   : {conf.mean():.0%}  "
          f"(100% en {int((conf >= 1.0).sum())}/{len(keep)})")
    return {"mean": err.mean(), "std": err.std(), "drift": drift}


def verdict(stats: list[dict]) -> None:
    ok = [s for s in stats if s]
    if not ok:
        print("\nSin corridas validas.")
        return
    m = float(np.mean([s["mean"] for s in ok]))
    s = float(np.mean([s["std"] for s in ok]))
    d = float(np.mean([s["drift"] for s in ok]))

    print("\n" + "=" * 62)
    print(f"  PROMEDIO DE {len(ok)} CORRIDA(S)")
    print(f"    error medio {m:+.1f} px ({m/PX_PER_CM:+.1f} cm)   "
          f"desv {s:.1f} px   deriva {d:+.1f} px/s")
    print()
    good = True
    if abs(m) > 20:
        good = False
        print(f"  X DESVIADO {abs(m)/PX_PER_CM:.1f} cm del centro del carril.")
        print(f"    -> el cero esta mal: recentra el carro a mano y remide")
        print(f"       LANE_ERROR_OFFSET_PX (ahora {LANE_ERROR_OFFSET_PX}).")
    if abs(d) > 12:
        good = False
        side = "izquierda" if d > 0 else "derecha"
        print(f"  X SE VA HACIA LA {side.upper()} a {abs(d)/PX_PER_CM:.1f} cm/s.")
        print(f"    -> el trim no venci el sesgo mecanico. Vuelve a correr")
        print(f"       tools/auto_trim.py (ahora {SERVO_TRIM_DEG:+.1f} deg).")
    if s > 35:
        good = False
        print(f"  X SERPENTEA {s/PX_PER_CM:.1f} cm.")
        print(f"    -> apunta mas lejos: sube LANE_AIM_WINDOW_FRAC hacia 0.85,")
        print(f"       o baja STEER_KP (ahora {STEER_KP}).")
    if good:
        print("  OK -- VA DERECHO. Listo para las corridas de frenado.")
    print("=" * 62)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--cruise", type=float, default=25.0)
    ap.add_argument("--kick", type=float, default=80.0)
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    print(f"trim {SERVO_TRIM_DEG:+.1f} deg | offset {LANE_ERROR_OFFSET_PX} px | "
          f"bias {LANE_RIGHT_BIAS} | Kp {STEER_KP} Ki {STEER_KI} Kd {STEER_KD} | "
          f"banda {LANE_DEADBAND_PX} px | heading {STEER_HEADING_GAIN}")

    motor    = MotorDriver(pin_rpwm=PIN_MOTOR_RPWM, pin_lpwm=PIN_MOTOR_LPWM)
    steering = SteeringDriver()
    cam      = CameraStream(width=640, height=480, fps=30)
    cam.start()
    lane = LanePipeline()
    pid  = PIDController(kp=STEER_KP, ki=STEER_KI, kd=STEER_KD, setpoint=0.0,
                         output_limits=(SERVO_MIN_ANGLE - SERVO_CENTER_ANGLE,
                                        SERVO_MAX_ANGLE - SERVO_CENTER_ANGLE),
                         integral_limits=(-25.0, 25.0))
    fsm = AutonomousFSM(motor, steering, pid)

    stats = []
    try:
        for k in range(1, args.runs + 1):
            input(f"\n[{k}/{args.runs}] Pon el carro en el CARRIL DERECHO "
                  f"apuntando recto y presiona Enter (Ctrl+C aborta)...")
            s = run_once(fsm, cam, lane, steering, args.cruise, args.kick,
                         args.seconds)
            stats.append(report(s, k, args.runs))
    except KeyboardInterrupt:
        print("\n[STRAIGHT] Abortado.")
    finally:
        motor.brake()
        steering.center()
        time.sleep(0.2)
        cam.stop()
        motor.cleanup()

    verdict(stats)


if __name__ == "__main__":
    main()
