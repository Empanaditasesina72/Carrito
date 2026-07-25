"""Closed-form geometry of the camera, the road and the steering.

Derives, from the measured values in config.py alone, the numbers the paper
must report and the numbers needed to calibrate the vehicle on the physical
track. Runs anywhere: no camera, no GPIO, no network.

What it answers:
  1. Camera field of view and the strip of ground it actually sees.
  2. Which lane lines fall inside the frame near the car -- this determines
     which pair of lines the sliding-window histogram locks onto, and therefore
     the correct LanePipeline right_bias.
  3. Apparent size in pixels of the STOP sign versus distance (the pinhole
     model used by every detector path).
  4. Ackermann steering geometry and the minimum turning radius.

Usage (from TMR2026/):
    python tools/track_geometry.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOCAL_LENGTH_PX,
    CAMERA_HEIGHT_M, CAMERA_TILT_DEG,
    WHEELBASE, TRACK_WIDTH, CAR_LENGTH, CAR_WIDTH,
    MAX_STEERING_ANGLE_DEG,
    SERVO_CENTER_ANGLE, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE,
    ROAD_TOTAL_LENGTH_M, ROAD_LEFT_TO_DASHED_M, ROAD_DASHED_TO_RIGHT_M,
    ROAD_STOP_SIGN_AT_M, ROAD_PARKING_GAP_M, LANE_WIDTH_M,
    STOP_SIGN_REAL_HEIGHT_M, STOP_SIGN_TOTAL_HEIGHT_M,
    STOP_BRAKE_START_MM, STOP_TARGET_MM, PARK_SIDE,
)


def ground_distance(pixel_row: float) -> float | None:
    """Ground distance ahead of the camera seen at an image row (None = sky)."""
    ray_below_axis = math.atan((pixel_row - CAMERA_HEIGHT / 2.0)
                               / CAMERA_FOCAL_LENGTH_PX)
    angle = math.radians(CAMERA_TILT_DEG) + ray_below_axis
    if angle <= 1e-6:
        return None
    return CAMERA_HEIGHT_M / math.tan(angle)


def half_width_at(distance_m: float, hfov_deg: float) -> float:
    """Approximate lateral half-coverage of the frame at a ground distance."""
    return distance_m * math.tan(math.radians(hfov_deg / 2.0))


def image_row(distance_m: float, height_m: float) -> float:
    """Image row where a point at (distance, height above ground) projects."""
    below_horizon = math.atan((CAMERA_HEIGHT_M - height_m) / distance_m)
    below_axis = below_horizon - math.radians(CAMERA_TILT_DEG)
    return CAMERA_HEIGHT / 2.0 + CAMERA_FOCAL_LENGTH_PX * math.tan(below_axis)


def sign_exit_distance() -> float:
    """Closest distance at which the octagon is still fully inside the frame."""
    foot = STOP_SIGN_TOTAL_HEIGHT_M - STOP_SIGN_REAL_HEIGHT_M
    limit = math.radians(CAMERA_TILT_DEG) + math.atan(
        (CAMERA_HEIGHT / 2.0) / CAMERA_FOCAL_LENGTH_PX)
    return (CAMERA_HEIGHT_M - foot) / math.tan(limit)


def main() -> int:
    vfov = 2.0 * math.degrees(math.atan(CAMERA_HEIGHT / 2.0 / CAMERA_FOCAL_LENGTH_PX))
    hfov = 2.0 * math.degrees(math.atan(CAMERA_WIDTH / 2.0 / CAMERA_FOCAL_LENGTH_PX))

    d_near = ground_distance(CAMERA_HEIGHT)
    d_mid = ground_distance(CAMERA_HEIGHT / 2.0)
    d_far = ground_distance(0.0)

    print("=" * 66)
    print("  1. CAMERA")
    print("=" * 66)
    print(f"  resolution        : {CAMERA_WIDTH} x {CAMERA_HEIGHT} px")
    print(f"  focal length      : {CAMERA_FOCAL_LENGTH_PX:.0f} px")
    print(f"  field of view     : {hfov:.1f} deg horizontal, {vfov:.1f} deg vertical")
    print(f"  mounting          : {CAMERA_HEIGHT_M * 100:.0f} cm high, "
          f"{CAMERA_TILT_DEG:.0f} deg down")
    print(f"  ground seen       : from {d_near * 100:.0f} cm (bottom row) "
          f"to {'horizon' if d_far is None else f'{d_far * 100:.0f} cm'} (top row)")
    print(f"  frame centre at   : {d_mid * 100:.0f} cm ahead")

    print()
    print("=" * 66)
    print("  2. ROAD AND WHICH LINES THE DETECTOR SEES")
    print("=" * 66)
    print(f"  total length      : {ROAD_TOTAL_LENGTH_M * 100:.0f} cm")
    print(f"  left solid -> dashed  : {ROAD_LEFT_TO_DASHED_M * 100:.1f} cm")
    print(f"  dashed -> right solid : {ROAD_DASHED_TO_RIGHT_M * 100:.1f} cm")
    print(f"  full width        : {LANE_WIDTH_M * 100:.1f} cm")
    print(f"  car width         : {CAR_WIDTH * 100:.1f} cm")

    lane_centre_from_dashed = ROAD_DASHED_TO_RIGHT_M / 2.0
    offsets = {
        "left solid":  -(ROAD_LEFT_TO_DASHED_M + lane_centre_from_dashed),
        "dashed centre": -lane_centre_from_dashed,
        "right solid": +lane_centre_from_dashed,
    }

    print()
    print(f"  Assuming the car drives centred in the RIGHT lane, the camera axis")
    print(f"  sits {lane_centre_from_dashed * 100:.1f} cm right of the dashed line.")
    print(f"  Lateral offsets from the camera axis and first distance in frame:")
    visible_near = []
    for name, off in offsets.items():
        needed = abs(off) / math.tan(math.radians(hfov / 2.0))
        in_near = needed <= d_near
        if in_near:
            visible_near.append(name)
        flag = "IN FRAME at the bottom row" if in_near else \
               f"only from {needed * 100:.0f} cm ahead"
        print(f"    {name:<15} {off * 100:+6.1f} cm   {flag}")

    print()
    print(f"  Frame half-coverage: {half_width_at(d_near, hfov) * 100:.1f} cm "
          f"at {d_near * 100:.0f} cm, "
          f"{half_width_at(d_mid, hfov) * 100:.1f} cm at {d_mid * 100:.0f} cm")

    if len(visible_near) >= 2:
        pair = visible_near[-2:]
        if "left solid" in pair:
            span = LANE_WIDTH_M
            bias = (ROAD_LEFT_TO_DASHED_M + lane_centre_from_dashed) / span
        else:
            span = ROAD_DASHED_TO_RIGHT_M
            bias = 0.50
        print()
        print(f"  => The histogram will lock onto: {pair[0]} + {pair[1]}")
        print(f"  => Their separation is {span * 100:.1f} cm")
        print(f"  => Start LanePipeline right_bias at {bias:.2f} "
              f"to sit in the middle of the right lane")
    else:
        print()
        print("  => WARNING: fewer than two lines in frame near the car; the")
        print("     pipeline will run in single-line mode (confidence 0.5).")

    print()
    print("=" * 66)
    print("  3. STOP SIGN (pinhole model)")
    print("=" * 66)
    print(f"  octagon height    : {STOP_SIGN_REAL_HEIGHT_M * 100:.1f} cm")
    print(f"  total with post   : {STOP_SIGN_TOTAL_HEIGHT_M * 100:.1f} cm")
    print(f"  placed at         : {ROAD_STOP_SIGN_AT_M * 100:.0f} cm from the start")
    print(f"  runway before it  : {ROAD_STOP_SIGN_AT_M * 100:.0f} cm "
          f"({(ROAD_STOP_SIGN_AT_M / CAR_LENGTH):.1f} car lengths)")
    foot = STOP_SIGN_TOTAL_HEIGHT_M - STOP_SIGN_REAL_HEIGHT_M
    print()
    print("  distance -> apparent height and where it lands in the frame")
    print("  (octagon spans rows top..bottom; frame is 0.."
          f"{CAMERA_HEIGHT})")
    for d_mm in (1500, 1000, STOP_BRAKE_START_MM, 500, STOP_TARGET_MM):
        d_m = d_mm / 1000.0
        h_px = CAMERA_FOCAL_LENGTH_PX * STOP_SIGN_REAL_HEIGHT_M / d_m
        r_top = image_row(d_m, STOP_SIGN_TOTAL_HEIGHT_M)
        r_bot = image_row(d_m, foot)
        inside = 0.0 <= r_top and r_bot <= CAMERA_HEIGHT
        tag = ""
        if d_mm == STOP_BRAKE_START_MM:
            tag = "  <- braking starts"
        elif d_mm == STOP_TARGET_MM:
            tag = "  <- target stop"
        print(f"    {d_mm:>5} mm   {h_px:6.1f} px   rows {r_top:5.0f}..{r_bot:5.0f}"
              f"   {'in frame' if inside else 'CLIPPED'}{tag}")

    d_exit = sign_exit_distance()
    print()
    print(f"  The octagon leaves the bottom of the frame below "
          f"{d_exit * 1000:.0f} mm.")
    if d_exit < STOP_TARGET_MM / 1000.0:
        print(f"  => OK: the car stops at {STOP_TARGET_MM} mm, "
              f"{STOP_TARGET_MM - d_exit * 1000:.0f} mm of margin. The detector")
        print(f"     keeps the sign in view for the whole braking phase.")
    else:
        print(f"  => PROBLEM: the sign is lost before reaching the "
              f"{STOP_TARGET_MM} mm setpoint;")
        print(f"     raise the sign or the camera, or increase STOP_TARGET_MM.")

    print()
    print("=" * 66)
    print("  4. STEERING (Ackermann)")
    print("=" * 66)
    r_min = WHEELBASE / math.tan(math.radians(MAX_STEERING_ANGLE_DEG))
    servo_span = min(SERVO_CENTER_ANGLE - SERVO_MIN_ANGLE,
                     SERVO_MAX_ANGLE - SERVO_CENTER_ANGLE)
    print(f"  wheelbase         : {WHEELBASE * 100:.1f} cm")
    print(f"  wheel track       : {TRACK_WIDTH * 100:.1f} cm")
    print(f"  body              : {CAR_LENGTH * 100:.1f} x {CAR_WIDTH * 100:.1f} cm")
    r_servo = WHEELBASE / math.tan(math.radians(servo_span))
    print(f"  max wheel angle   : {MAX_STEERING_ANGLE_DEG:.1f} deg "
          f"(MAX_STEERING_ANGLE_DEG)")
    print(f"  min turn radius   : {r_min:.3f} m  (centreline)")
    print(f"  servo authority   : {SERVO_MIN_ANGLE:.0f}-{SERVO_MAX_ANGLE:.0f} deg "
          f"= +/-{servo_span:.0f} deg about {SERVO_CENTER_ANGLE:.0f}")
    print(f"  radius at +/-{servo_span:.0f} deg : {r_servo:.3f} m")
    if abs(servo_span - MAX_STEERING_ANGLE_DEG) > 1.0:
        print(f"  => NOTE: the servo reaches {servo_span:.0f} deg but "
              f"MAX_STEERING_ANGLE_DEG says {MAX_STEERING_ANGLE_DEG:.0f} deg.")
        print(f"     Unless the linkage geometrically amplifies servo travel, the")
        print(f"     achievable radius is {r_servo:.3f} m, not {r_min:.3f} m. State")
        print(f"     the measured wheel angle at full servo lock in the paper.")

    if CAR_LENGTH < WHEELBASE:
        print()
        print(f"  => INCONSISTENT: wheelbase {WHEELBASE * 100:.1f} cm exceeds "
              f"CAR_LENGTH {CAR_LENGTH * 100:.1f} cm.")
        print(f"     Axle-to-axle cannot be longer than the vehicle. Either")
        print(f"     CAR_LENGTH is the chassis only (wheels overhang it), or one")
        print(f"     of the two was mismeasured. Re-check before publishing.")
    print()
    print(f"  Lane width is {LANE_WIDTH_M * 100:.1f} cm and the minimum radius is "
          f"{r_min * 100:.0f} cm,")
    print(f"  so a full-lock turn needs {r_min / LANE_WIDTH_M:.1f} lane widths of room:")
    print(f"  keep the physical runs on the straight and steer gently.")

    print()
    print("=" * 66)
    print("  5. PARKING")
    print("=" * 66)
    print(f"  bay side          : {PARK_SIDE}")
    print(f"  slot width        : {ROAD_PARKING_GAP_M * 100:.1f} cm")
    print(f"  car width         : {CAR_WIDTH * 100:.1f} cm")
    margin = (ROAD_PARKING_GAP_M - CAR_WIDTH) / 2.0
    print(f"  clearance         : {margin * 100:.1f} cm per side")
    if margin < 0.03:
        print("  => VERY TIGHT: expect contact; widen the slot if possible.")
    else:
        print("  => Feasible for a perpendicular (battery) entry.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
