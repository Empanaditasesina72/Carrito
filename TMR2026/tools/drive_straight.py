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
from vision.motion_check import MotionCheck
from control.pid_controller import PIDController
from control.fsm import AutonomousFSM, FSMState

LOOP_HZ = 50
PX_PER_CM = 6.796


def run_once(fsm, cam, lane, steering, cruise, kick, seconds, motion):
    """One straight run. Returns (samples, stalled_seconds)."""
    fsm.MAX_AUTO_PWM = float(cruise)

    # Baseline the motion detector on this scene while the car is still stopped,
    # so the stall test is relative to the sensor noise at the CURRENT gain.
    motion.reset()
    tc = time.monotonic()
    while time.monotonic() - tc < 0.5:
        motion.calibrate(cam.get_frame())
        time.sleep(0.03)
    base = motion.finish_calibration()

    fsm.activate()
    kick_until = time.monotonic() + 0.30 if kick > 0 else 0.0
    next_rekick = 0.0
    stalled_s = 0.0
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

            # A stall is invisible to every other metric -- the scene stops
            # changing, so the error freezes and the run reports flawless
            # tracking. Re-kick instead of finishing a run that measured
            # nothing. Only after the launch window, and rate-limited so a
            # genuinely stuck car does not get hammered every tick.
            moving = motion.update(frame)
            if (not moving and now - t0 > 0.8 and now >= next_rekick
                    and fsm.state == FSMState.CRUCERO and kick > 0):
                fsm.motor.kick(kick, 0.0)
                next_rekick = now + 0.5
                stalled_s += 0.5
                print(f"    [{now-t0:4.1f}s] ESTANCADO (mov {motion.last_diff:.2f} "
                      f"< umbral {motion.threshold:.2f}) -- reintentando arranque",
                      flush=True)

            samples.append((now - t0, r.error_px, r.confidence,
                            steering.current_angle, r.heading))
            time.sleep(max(0.0, (1.0 / LOOP_HZ) - (time.monotonic() - now)))
    finally:
        fsm.motor.brake()
        fsm.deactivate()
    return samples, stalled_s, base


def report(samples, stalled_s, idx, total) -> dict:
    """Judge one run by WHERE IT ENDED UP, not by its average.

    The average was the wrong measure and it produced a wrong verdict on real
    data. The operator places the car by hand before each run, so the mean error
    over 4 s mostly reports the STARTING position, and a run that began 6 cm off
    and corrected all the way to centre showed a large mean and a large "drift"
    -- and got flagged as a failure when it was the controller working exactly as
    intended. What matters is the error at the END and whether it is still
    swinging once it gets there.
    """
    if len(samples) < 20:
        print(f"  corrida {idx}/{total}: solo {len(samples)} muestras -- INVALIDA")
        return {}

    a = np.asarray(samples, dtype=float)
    t_end = a[-1, 0]
    first = a[(a[:, 0] >= 0.5) & (a[:, 0] < 1.5)]
    last  = a[a[:, 0] >= max(1.5, t_end - 1.5)]
    if len(first) < 5 or len(last) < 5:
        print(f"  corrida {idx}/{total}: demasiado corta para juzgar")
        return {}

    e0, e1 = float(first[:, 1].mean()), float(last[:, 1].mean())
    weave  = float(last[:, 1].std())
    ang    = float(last[:, 3].mean())
    conf   = float(a[:, 2].mean())

    print(f"\n--- corrida {idx}/{total} ({len(a)} muestras, {t_end:.1f} s) ---")
    if stalled_s > 0:
        print(f"  !! el motor se estanco ~{stalled_s:.1f} s -- corrida NO valida")
    print(f"  empezo en   : {e0:+7.1f} px ({e0/PX_PER_CM:+5.1f} cm)")
    print(f"  TERMINO en  : {e1:+7.1f} px ({e1/PX_PER_CM:+5.1f} cm)   <- lo que importa")
    print(f"  serpenteo   : {weave:7.1f} px ({weave/PX_PER_CM:5.1f} cm) al final")
    print(f"  servo       : medio {ang:6.2f} deg  "
          f"rango {last[:,3].min():.1f}..{last[:,3].max():.1f}")
    print(f"  confianza   : {conf:.0%}")
    if abs(e1) < abs(e0) - 3:
        print(f"  -> CONVERGIENDO: corrigio {abs(e0)-abs(e1):.0f} px hacia el centro")
    elif abs(e1) > abs(e0) + 3:
        print(f"  -> DIVERGIENDO: se alejo {abs(e1)-abs(e0):.0f} px del centro")

    return {"end": e1, "weave": weave, "angle": ang,
            "converged": abs(e1) < abs(e0) - 3, "stalled": stalled_s > 0}


def verdict(stats: list[dict]) -> float | None:
    """Print the diagnosis; return the suggested SERVO_TRIM_DEG, or None."""
    all_runs = [s for s in stats if s]
    ok = [s for s in all_runs if not s["stalled"]]
    if not ok:
        print("\nSin corridas validas (todas se estancaron o fueron muy cortas).")
        return None
    if len(ok) < len(all_runs):
        print(f"\n({len(all_runs)-len(ok)} corrida(s) descartada(s) por estancamiento)")

    e = float(np.mean([s["end"] for s in ok]))
    w = float(np.mean([s["weave"] for s in ok]))
    worst = max(abs(s["end"]) for s in ok)

    print("\n" + "=" * 62)
    print(f"  {len(ok)} CORRIDA(S) VALIDA(S)")
    print(f"    termino en {e:+.1f} px ({e/PX_PER_CM:+.1f} cm)   "
          f"serpenteo {w:.1f} px ({w/PX_PER_CM:.1f} cm)   "
          f"peor {worst:.0f} px")
    print()
    good = True
    if abs(e) > 25:
        good = False
        print(f"  X TERMINA {abs(e)/PX_PER_CM:.1f} cm fuera del centro del carril.")
        print(f"    -> recentra el carro a mano, corre diag_track y ajusta")
        print(f"       LANE_ERROR_OFFSET_PX (ahora {LANE_ERROR_OFFSET_PX}).")
    if w > 25:
        good = False
        print(f"  X SERPENTEA {w/PX_PER_CM:.1f} cm al final del recorrido.")
        print(f"    -> apunta mas lejos: LANE_AIM_WINDOW_FRAC hacia 0.85,")
        print(f"       o baja STEER_KP (ahora {STEER_KP}).")
    if good:
        print("  OK -- VA DERECHO Y SE QUEDA EN EL CARRIL.")

    # TRIM, measured directly instead of inferred from a drift rate.
    #
    # In closed loop with pure P the car travels straight only when the commanded
    # steering exactly cancels the mechanical bias, so the servo's steady-state
    # deviation from centre IS that bias -- read off in one run, where the
    # open-loop drift search needed eight and never converged.
    #
    # _physical() computes (2*90 - logical) + TRIM, so holding logical `a` gives
    # the same wheels as centre would with TRIM shifted by -(a - 90):
    #     TRIM_new = TRIM_old - (mean_angle - 90)
    a = float(np.mean([s["angle"] for s in ok]))
    dev = a - 90.0
    suggested = SERVO_TRIM_DEG - dev
    print()
    print(f"  TRIM: el servo se sostuvo en {a:.2f} deg ({dev:+.2f} del centro)")
    if AutonomousFSM.DEADBAND_PX > 0:
        print(f"    (banda muerta activa -> esta lectura NO sirve para el trim;")
        print(f"     repite con --no-deadband)")
        print("=" * 62)
        return None
    if abs(dev) < 0.4:
        print(f"    trim correcto: el controlador no necesita sostener nada.")
        print("=" * 62)
        return None
    print(f"    sesgo mecanico residual {-dev:+.2f} deg  ->  "
          f"SERVO_TRIM_DEG {SERVO_TRIM_DEG:+.1f} deberia ser {suggested:+.1f}")
    print(f"    aplicalo con:  python tools/drive_straight.py --apply-trim")
    print("=" * 62)
    return suggested


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--cruise", type=float, default=45.0)
    ap.add_argument("--kick", type=float, default=80.0)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--no-deadband", action="store_true",
                    help="disable the steering deadband for this run. Required "
                         "for a trustworthy trim estimate: inside the band the "
                         "servo is pinned to centre, so the mean angle no longer "
                         "reflects the correction the car actually needs.")
    ap.add_argument("--apply-trim", action="store_true",
                    help="write the measured trim into config.py (implies "
                         "--no-deadband)")
    args = ap.parse_args()

    if args.apply_trim:
        args.no_deadband = True
    if args.no_deadband:
        AutonomousFSM.DEADBAND_PX = 0.0
        print("[STRAIGHT] banda muerta DESACTIVADA (medicion de trim)")

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
    motion = MotionCheck()

    stats = []
    try:
        for k in range(1, args.runs + 1):
            input(f"\n[{k}/{args.runs}] Pon el carro en el CARRIL DERECHO "
                  f"apuntando recto y presiona Enter (Ctrl+C aborta)...")
            s, stalled, base = run_once(fsm, cam, lane, steering,
                                        args.cruise, args.kick,
                                        args.seconds, motion)
            stats.append(report(s, stalled, k, args.runs))
    except KeyboardInterrupt:
        print("\n[STRAIGHT] Abortado.")
    finally:
        motor.brake()
        steering.center()
        time.sleep(0.2)
        cam.stop()
        motor.cleanup()

    suggested = verdict(stats)

    if args.apply_trim and suggested is not None:
        import re
        cfg = ROOT / "config.py"
        txt = cfg.read_text(encoding="utf-8")
        txt = re.sub(r"^SERVO_TRIM_DEG\s*=.*$",
                     f"SERVO_TRIM_DEG      = {suggested:.1f}",
                     txt, count=1, flags=re.M)
        cfg.write_text(txt, encoding="utf-8")
        print(f"\n[STRAIGHT] config.py: SERVO_TRIM_DEG = {suggested:.1f}")
        print("[STRAIGHT] vuelve a correr sin --apply-trim para confirmar.")
        print("[STRAIGHT] recuerda: esto ensucia el repo del Pi y bloquea el")
        print("           siguiente 'git push pi'. Avisame para commitearlo.")
    elif args.apply_trim:
        print("\n[STRAIGHT] nada que aplicar: el trim ya esta bien.")


if __name__ == "__main__":
    main()
