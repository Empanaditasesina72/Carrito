"""Find the steering trim: the offset that makes the wheels actually point straight.

A logical 90 deg is the geometric centre of the servo, not necessarily the angle
at which the vehicle drives straight. When the two differ the car pulls to one
side and the controller cannot correct it: it is told the lane error is zero, so
it outputs zero correction and the drift goes unopposed. That is a mechanical
offset and it belongs in SERVO_TRIM_DEG, not in the controller.

Only the servo moves. The motor is never touched.

Usage (from TMR2026/, on the Pi):
    python tools/trim_steering.py

Put the car on a flat surface with room ahead. Type an offset, look at the front
wheels, repeat until they are parallel to the direction of travel. Sighting along
the chassis from behind is more reliable than eyeballing the wheels alone; a
straight edge laid against both front tyres is better still.

IMPORTANT: wheels that look straight are not the same as a car that tracks
straight. That is why trim exists on every RC car. Off-centre weight, a fraction
of a degree of toe, uneven tyre friction or a sloping floor all pull a car whose
wheels are visually parallel. So find the trim by DRIVING, not by eyeballing the
wheels: with --drive the tool rolls the car forward at cruise with the servo
centred, you watch which way it goes, adjust, and repeat.

Commands:
    <number>   set the trim to that value in degrees (e.g. -4, 2.5)
    + / -      step by 0.5 deg
    l / r      nudge 2 deg left / right
    0          back to no trim
    d          DRIVE forward briefly to see which way it pulls (needs --drive)
    s          show the final value to put in config.py
    q          quit

Which way to correct: if the car drifts LEFT, press `r` (or use a positive trim)
until it runs straight; if it drifts RIGHT, press `l`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SERVO_CENTER_ANGLE, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE


def main() -> int:
    import argparse
    import time
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", action="store_true",
                    help="enable the 'd' command, which MOVES the car")
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--cruise", type=float, default=25.0)
    ap.add_argument("--kick", type=float, default=60.0)
    args = ap.parse_args()

    import hardware.steering_driver as sd
    from hardware.steering_driver import SteeringDriver

    steering = SteeringDriver()
    trim = float(sd.SERVO_TRIM_DEG)

    motor = None
    if args.drive:
        from config import PIN_MOTOR_RPWM, PIN_MOTOR_LPWM
        from hardware.motor import MotorDriver
        motor = MotorDriver(pin_rpwm=PIN_MOTOR_RPWM, pin_lpwm=PIN_MOTOR_LPWM)
        print("  --drive ON: the 'd' command will MOVE the car. Clear the track.")

    def drive_straight() -> None:
        if motor is None:
            print("      Not enabled. Restart with --drive to allow motion.")
            return
        print(f"      driving {args.seconds:.1f} s at {args.cruise:.0f}% - watch the drift")
        try:
            if args.kick > 0:
                motor.kick(args.kick, 0.25)
            motor.set_speed(args.cruise)
            time.sleep(args.seconds)
        finally:
            motor.brake()
        print("      stopped. Drifted LEFT -> press r. Drifted RIGHT -> press l.")

    print("=" * 62)
    print("  STEERING TRIM  -  only the servo moves, the motor stays off")
    print(f"  logical centre {SERVO_CENTER_ANGLE:.0f} deg, "
          f"inverted={sd.STEERING_INVERTED}")
    print("  number / + / - / l / r / d / 0 / s / q     (see --help)")
    print("=" * 62)

    def apply(t: float) -> None:
        sd.SERVO_TRIM_DEG = t
        steering.center()
        print(f"  trim {t:+.1f} deg  ->  servo is being written "
              f"{steering._physical(SERVO_CENTER_ANGLE):.1f} deg")

    apply(trim)
    while True:
        try:
            raw = input("  trim> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if raw in ("q", "quit", "exit"):
            break
        if raw == "s":
            print()
            print(f"  >>> SERVO_TRIM_DEG = {trim:.1f}")
            print("  Tell Claude this number, or set it in TMR2026/config.py")
            print()
            continue
        if raw == "d":
            drive_straight()
            continue
        if raw == "0":
            trim = 0.0
        elif raw == "+":
            trim += 0.5
        elif raw == "-":
            trim -= 0.5
        elif raw == "l":
            trim -= 2.0
        elif raw == "r":
            trim += 2.0
        elif raw == "":
            continue
        else:
            try:
                trim = float(raw.replace(",", "."))
            except ValueError:
                print("      Type a number, or + - l r 0 s q")
                continue
        span = SERVO_MAX_ANGLE - SERVO_MIN_ANGLE
        if abs(trim) > span / 2:
            print(f"      {trim:+.1f} is larger than half the steering range "
                  f"({span/2:.0f} deg). That is not a trim, something is "
                  f"mechanically wrong -- check the linkage and the servo horn.")
        apply(trim)

    steering.center()
    if motor is not None:
        motor.brake()
        motor.cleanup()
    print()
    print(f"  final trim: SERVO_TRIM_DEG = {trim:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
