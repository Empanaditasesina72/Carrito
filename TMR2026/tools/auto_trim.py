#!/usr/bin/env python3
"""Auto-calibrate SERVO_TRIM_DEG: short open-loop bursts, camera measures drift.

The push test confirmed the chassis itself pulls to one side with the servo
centred, so "logical 90" is not "wheels straight" on this car. The controller
cannot absorb that (a bias of b deg leaves a standing offset of b/Kp px, and the
integrator needs ~40 s to cancel what a run gives it 3 s for), so the bias must
die here, in the trim.

Each iteration: you place the car centred on the lane pointing straight, it
drives OPEN LOOP (servo fixed at centre+trim) for a short burst, and the lane
pipeline measures how fast the error drifts. Drift right => wheels point right
=> trim moves the servo left. Steps shrink as the drift shrinks; convergence is
|drift| <= DONE_PX_S. The result is written into config.py.

Sign chain, all verified on this car 2026-07-26:
  error_px falls when the car moves right (displacement test)
  physical angle above 90 = wheels LEFT (logical 58 -> physical 122 turned left)
  _physical() adds SERVO_TRIM_DEG after inversion
  => drift < 0 (car going right) => trim += step (wheels left).

Usage (car on the track, you replace it at centre before each burst):
    python tools/auto_trim.py
    python tools/auto_trim.py --burst 1.6 --cruise 25 --kick 80
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PY = ROOT / "config.py"

DONE_PX_S   = 15.0    # |drift| below this = straight enough (2.2 cm/s lateral)
COARSE_PX_S = 40.0    # above this, take the big step
STEP_COARSE = 1.5     # deg
STEP_FINE   = 0.6
TRIM_LIMIT  = 8.0     # give up beyond this: that is a bent linkage, not trim


def measure_burst(motor, steering, cam, lane, cruise, kick, burst_s, trim):
    """One open-loop burst; returns (drift px/s, n samples, mean heading)."""
    import hardware.steering_driver as sd
    sd.SERVO_TRIM_DEG = trim              # module global read by _physical()
    steering.center()
    time.sleep(0.3)

    motor.kick(kick, 0.25)
    motor.kick(cruise, 0.0)

    t0 = time.monotonic()
    ts, errs, heads = [], [], []
    while time.monotonic() - t0 < burst_s:
        f = cam.get_frame()
        if f is not None:
            r = lane.process(f)
            if r.confidence >= 0.5:
                ts.append(time.monotonic() - t0)
                errs.append(r.error_px)
                heads.append(r.heading)
        time.sleep(0.05)
    motor.brake()

    if len(ts) < 5:
        return None, len(ts), 0.0
    drift = float(np.polyfit(ts, errs, 1)[0])
    return drift, len(ts), float(np.mean(heads))


def write_config(trim: float) -> None:
    txt = CONFIG_PY.read_text(encoding="utf-8")
    txt = re.sub(r"^SERVO_TRIM_DEG\s*=.*$",
                 f"SERVO_TRIM_DEG      = {trim:.1f}", txt, count=1, flags=re.M)
    CONFIG_PY.write_text(txt, encoding="utf-8")
    print(f"\n[TRIM] config.py: SERVO_TRIM_DEG = {trim:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst", type=float, default=1.6)
    ap.add_argument("--cruise", type=float, default=25.0)
    ap.add_argument("--kick", type=float, default=80.0)
    ap.add_argument("--max-iters", type=int, default=8)
    args = ap.parse_args()

    from config import (PIN_MOTOR_RPWM, PIN_MOTOR_LPWM, SERVO_TRIM_DEG)
    from hardware.motor import MotorDriver
    from hardware.steering_driver import SteeringDriver
    from vision.camera_stream import CameraStream
    from vision.lane_pipeline import LanePipeline

    motor = MotorDriver(pin_rpwm=PIN_MOTOR_RPWM, pin_lpwm=PIN_MOTOR_LPWM)
    steering = SteeringDriver()
    cam = CameraStream(width=640, height=480, fps=30)
    cam.start()
    lane = LanePipeline()

    trim = float(SERVO_TRIM_DEG)
    print(f"[TRIM] arrancando desde SERVO_TRIM_DEG = {trim:.1f}")
    print(f"[TRIM] convergencia: |deriva| <= {DONE_PX_S:.0f} px/s")

    try:
        for it in range(1, args.max_iters + 1):
            input(f"\n[{it}/{args.max_iters}] Centra el carro apuntando recto y "
                  f"presiona Enter (Ctrl+C aborta)...")
            drift, n, head = measure_burst(motor, steering, cam, lane,
                                           args.cruise, args.kick,
                                           args.burst, trim)
            if drift is None:
                print(f"  solo {n} muestras validas -- repite (¿carril visible?)")
                continue

            side = "DERECHA" if drift < 0 else "IZQUIERDA"
            print(f"  deriva {drift:+7.1f} px/s hacia la {side}  "
                  f"(n={n}, head {head:+.1f}, trim actual {trim:+.1f})")

            if abs(drift) <= DONE_PX_S:
                print(f"\n[TRIM] CONVERGIO: deriva {drift:+.1f} px/s con "
                      f"trim {trim:+.1f}")
                write_config(trim)
                return

            step = STEP_COARSE if abs(drift) > COARSE_PX_S else STEP_FINE
            trim += step if drift < 0 else -step
            if abs(trim) > TRIM_LIMIT:
                print(f"\n[TRIM] |trim| > {TRIM_LIMIT} deg -- esto ya no es "
                      f"calibracion, es la varilla de la direccion doblada. "
                      f"Ajusta la varilla y vuelve a correr.")
                return
            print(f"  -> nuevo trim {trim:+.1f}")

        print(f"\n[TRIM] no convergio en {args.max_iters} iteraciones; "
              f"mejor valor probado {trim:+.1f} (no escrito).")
    except KeyboardInterrupt:
        print("\n[TRIM] abortado.")
    finally:
        motor.brake()
        steering.center()
        time.sleep(0.2)
        cam.stop()
        motor.cleanup()


if __name__ == "__main__":
    main()
