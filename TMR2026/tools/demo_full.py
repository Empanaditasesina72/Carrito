#!/usr/bin/env python3
"""The whole run, unattended: follow the lane -> stop at the sign -> park.

One command, one deliverable. Everything else in tools/ measures a component;
this drives the sequence the competition and the paper actually need, using the
production AutonomousFSM and ParkingFSM with no changes, and narrates what it is
doing so a recording of the terminal explains the recording of the car.

    FASE 1  CARRIL    lane following, closed loop, until the sign is seen
    FASE 2  ALTO      brake, hold 5 s (the FSM's own ESPERA)
    FASE 3  REANUDA   pull away again
    FASE 4  ESTACIONA hand over to the parking FSM

Phases 1-3 are the FSM's own CRUCERO/PRECAUCION/FRENADO/ESPERA/REANUDAR cycle --
this only watches it. Phase 4 starts once the stop has been served and the
resume has run for --park-after seconds.

WHAT IS AND IS NOT PROVEN
  Lane following and the stop are exercised by tools/drive_straight.py and the
  braking trials. THE PARKING PHASE HAS NEVER RUN ON THIS CAR. It is a timed
  open-loop manoeuvre (ParkingFSM: search, turn in, straighten) with the ToF
  disabled, so it cannot see the bay -- it drives the pattern blind. Expect to
  tune ParkingFSM.TURN_IN_S / STRAIGHTEN_S. Use --no-park until the first three
  phases are reliable.

SAFETY
  * First run with the drive wheels OFF THE GROUND.
  * Ctrl+C brakes and exits at any point.
  * --timeout bounds the whole run; there is no other automatic stop, since the
    ToF sensors are disabled (USE_TOF_SENSORS=False).

Usage (from TMR2026/):
    python tools/demo_full.py --no-park            # lane + stop only
    python tools/demo_full.py                      # full sequence
    python tools/demo_full.py --cruise 22 --timeout 40
"""
from __future__ import annotations

import argparse
import os
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
                    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
                    USE_IMX500_NPU, IMX500_RPK_PATH, IMX500_LABELS_PATH,
                    IMX500_CONF)
from hardware.motor import MotorDriver
from hardware.steering_driver import SteeringDriver
from vision.lane_pipeline import LanePipeline
from vision.motion_check import MotionCheck
from control.pid_controller import PIDController
from control.fsm import AutonomousFSM, FSMState
from control.parking_fsm import ParkingFSM, ParkingState

LOOP_HZ = 50
PX_PER_CM = 6.796


def build_vision():
    if USE_IMX500_NPU and os.path.isfile(IMX500_RPK_PATH):
        try:
            from vision.imx500_detector import IMX500CameraStream
            npu = IMX500CameraStream(
                rpk_path=IMX500_RPK_PATH, labels_path=IMX500_LABELS_PATH,
                width=CAMERA_WIDTH, height=CAMERA_HEIGHT,
                fps=CAMERA_FPS, conf=IMX500_CONF)
            return npu, npu
        except Exception as e:
            print(f"[DEMO] NPU no disponible ({e}); ruta CPU.")
    from vision.camera_stream import CameraStream
    from vision.sign_detector import SignDetector
    cam = CameraStream(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS)
    return cam, SignDetector(model_path="weights/tmr_signs.pt",
                             conf=0.55, imgsz=320)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cruise", type=float, default=25.0)
    ap.add_argument("--kick", type=float, default=80.0)
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--park-after", type=float, default=3.0,
                    help="seconds of REANUDAR before handing over to parking")
    ap.add_argument("--no-park", action="store_true")
    ap.add_argument("--no-prompt", action="store_true",
                    help="skip the Enter prompt and count down instead, so the "
                         "run can be launched over SSH with no terminal")
    args = ap.parse_args()

    print("=" * 66)
    print("  DEMO COMPLETA -- el carro se va a MOVER")
    print(f"  trim {SERVO_TRIM_DEG:+.1f} deg | offset {LANE_ERROR_OFFSET_PX} px | "
          f"bias {LANE_RIGHT_BIAS} | Kp {STEER_KP}")
    print(f"  crucero {args.cruise:.0f}% | tope {args.timeout:.0f} s | "
          f"estacionar: {'NO' if args.no_park else 'SI'}")
    print("  Ctrl+C = freno inmediato")
    print("=" * 66)

    motor    = MotorDriver(pin_rpwm=PIN_MOTOR_RPWM, pin_lpwm=PIN_MOTOR_LPWM)
    steering = SteeringDriver()
    camera, sign_det = build_vision()
    camera.start()
    if sign_det is not camera:
        sign_det.start()
    lane   = LanePipeline()
    motion = MotionCheck()
    pid = PIDController(kp=STEER_KP, ki=STEER_KI, kd=STEER_KD, setpoint=0.0,
                        output_limits=(SERVO_MIN_ANGLE - SERVO_CENTER_ANGLE,
                                       SERVO_MAX_ANGLE - SERVO_CENTER_ANGLE),
                        integral_limits=(-25.0, 25.0))
    fsm  = AutonomousFSM(motor, steering, pid)
    park = ParkingFSM(motor, steering)

    if args.no_prompt:
        # input() needs a terminal; launched from a remote shell it hits EOF and
        # the process dies before the car ever moves.
        print("\n[DEMO] arrancando en 5 s -- APARTATE")
        for s in (5, 4, 3, 2, 1):
            print(f"  {s}...", flush=True)
            time.sleep(1.0)
    else:
        input("\nPon el carro en el CARRIL DERECHO al inicio de la pista y "
              "presiona Enter...")

    print("\n[DEMO] midiendo el ruido de la camara (carro quieto)...")
    tc = time.monotonic()
    while time.monotonic() - tc < 0.6:
        motion.calibrate(camera.get_frame())
        time.sleep(0.03)
    motion.finish_calibration()

    fsm.MAX_AUTO_PWM = float(args.cruise)
    # PRECAUCION defaults to 20 % PWM, which is at or below this motor's stall
    # threshold: the moment the sign is seen the car slows to a standstill, the
    # marginal detection flickers off, cruise resumes, and the log fills with
    # CRUCERO<->PRECAUCION. Keep it close to cruise; the braking distance is set
    # by SIGN_BBOX_STOP_MM, not by crawling up to the sign.
    fsm.PRECAUCION_PWM = max(20.0, args.cruise * 0.85)
    fsm.activate()

    t0 = last = time.monotonic()
    kick_until  = t0 + 0.30 if args.kick > 0 else 0.0
    next_rekick = 0.0
    stop_served_at = None
    parking = False
    log, phase = [], "CARRIL"
    print(f"\n[FASE 1] CARRIL -- siguiendo la linea\n")

    try:
        while time.monotonic() - t0 < args.timeout:
            now = time.monotonic()
            dt  = max(1e-3, now - last)
            last = now
            el = now - t0

            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            if sign_det is not camera:
                sign_det.update_frame(frame)
            r = lane.process(frame)

            if parking:
                park.lidar_mm = None
                park.update(dt)
                if park.done:
                    print(f"\n[FASE 4] ESTACIONADO en t={el:.1f} s")
                    break
            else:
                fsm.lane_error   = r.error_px
                fsm.lane_conf    = r.confidence
                fsm.lane_heading = r.heading
                fsm.lidar_mm     = None
                fsm.sign_visible = (sign_det.has_sign("stop_sign")
                                    or sign_det.has_sign("red"))
                closest = (sign_det.closest_sign("stop_sign")
                           or sign_det.closest_sign("red"))
                fsm.sign_distance_mm = (closest.distance_m * 1000.0
                                        if closest and closest.distance_m else None)
                fsm.update(dt)

                if now < kick_until and fsm.state == FSMState.CRUCERO:
                    motor.kick(args.kick, 0.0)

                # A stall freezes the scene, so every metric reads perfect while
                # the car sits still. Re-kick rather than time out on a run that
                # measured nothing. Never during a braking state -- brake() must
                # win.
                # PRECAUCION is included: it is not a braking state, and the
                # car spends most of an approach in it. Excluding it meant the
                # re-kick never fired exactly when the duty was lowest and a
                # stall was most likely. FRENADO and ESPERA stay excluded --
                # brake() must always win.
                if (not motion.update(frame) and el > 1.0 and now >= next_rekick
                        and fsm.state in (FSMState.CRUCERO, FSMState.PRECAUCION,
                                          FSMState.REANUDAR)
                        and args.kick > 0):
                    motor.kick(args.kick, 0.0)
                    next_rekick = now + 0.5
                    print(f"    [{el:4.1f}s] estancado -- reintentando arranque")

                if fsm.state == FSMState.ESPERA and phase == "CARRIL":
                    phase = "ALTO"
                    d = fsm.sign_distance_mm
                    print(f"\n[FASE 2] ALTO en t={el:.1f} s"
                          + (f", senal a {d:.0f} mm" if d else "")
                          + " -- esperando 5 s\n")
                if fsm.state == FSMState.REANUDAR and phase == "ALTO":
                    phase = "REANUDA"
                    stop_served_at = now
                    print(f"\n[FASE 3] REANUDA en t={el:.1f} s\n")

                if (not args.no_park and stop_served_at is not None
                        and now - stop_served_at >= args.park_after):
                    fsm.deactivate()
                    parking = True
                    phase = "ESTACIONA"
                    print(f"\n[FASE 4] ESTACIONA en t={el:.1f} s "
                          f"-- maniobra a ciegas, sin ToF\n")
                    park.activate()

            st = park.state.name if parking else fsm.state.name
            log.append((el, r.error_px, r.confidence, steering.current_angle))
            print(f"  {el:5.1f}s {phase:9} {st:16} err {r.error_px:+7.1f}px "
                  f"({r.error_px/PX_PER_CM:+5.1f}cm) conf {r.confidence:3.0%} "
                  f"servo {steering.current_angle:6.2f}   ", end="\r", flush=True)

            time.sleep(max(0.0, (1.0 / LOOP_HZ) - (time.monotonic() - now)))
        else:
            print(f"\n[DEMO] tope de {args.timeout:.0f} s alcanzado.")
    except KeyboardInterrupt:
        print("\n[DEMO] abortado por el usuario.")
    finally:
        motor.brake()
        if parking:
            park.deactivate()
        else:
            fsm.deactivate()
        steering.center()
        time.sleep(0.2)
        camera.stop()
        if sign_det is not camera:
            sign_det.stop()
        motor.cleanup()

    if len(log) > 20:
        a = np.asarray(log, dtype=float)
        tail = a[a[:, 0] >= max(1.5, a[-1, 0] - 2.0)]
        print("\n" + "=" * 66)
        print(f"  duracion {a[-1,0]:.1f} s   fase final: {phase}")
        print(f"  error al final : {tail[:,1].mean():+.1f} px "
              f"({tail[:,1].mean()/PX_PER_CM:+.1f} cm)")
        print(f"  serpenteo      : {tail[:,1].std():.1f} px "
              f"({tail[:,1].std()/PX_PER_CM:.1f} cm)")
        print(f"  confianza      : {a[:,2].mean():.0%}")
        print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
