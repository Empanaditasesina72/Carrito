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

Commands:
    <number>   set the trim to that value in degrees (e.g. -4, 2.5)
    + / -      step by 0.5 deg
    l / r      nudge 2 deg left / right
    0          back to no trim
    s          show the final value to put in config.py
    q          quit
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SERVO_CENTER_ANGLE, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE


def main() -> int:
    import hardware.steering_driver as sd
    from hardware.steering_driver import SteeringDriver

    steering = SteeringDriver()
    trim = float(sd.SERVO_TRIM_DEG)

    print("=" * 62)
    print("  STEERING TRIM  -  only the servo moves, the motor stays off")
    print(f"  logical centre {SERVO_CENTER_ANGLE:.0f} deg, "
          f"inverted={sd.STEERING_INVERTED}")
    print("  number / + / - / l / r / 0 / s / q     (see --help)")
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
    print()
    print(f"  final trim: SERVO_TRIM_DEG = {trim:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
