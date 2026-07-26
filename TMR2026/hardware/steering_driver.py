"""MG90s servo on a PCA9685 board (steering).

Alternative I2C bus: SDA=GPIO0, SCL=GPIO1
    i2c = busio.I2C(board.D1, board.D0)

Ackermann geometry implemented:
  The RC car has both front wheels mechanically linked to a single servo.
  We use the central (bicycle-model) angle and convert it to the servo PWM
  pulse.

  Turning radius:   R = Wheelbase / tan(delta)
  Inner-wheel angle: arctan(L / (R - W/2))
  Outer-wheel angle: arctan(L / (R + W/2))

  Since the servo moves both wheels at once we use the central angle delta
  directly (a valid simplification for 1:10 scale and low speeds).
"""

import math
import adafruit_pca9685
from adafruit_motor import servo as adafruit_servo
from adafruit_extended_bus import ExtendedI2C

from config import (
    PCA9685_PWM_FREQ,
    SERVO_CHANNEL,
    SERVO_MIN_PULSE_US,
    SERVO_MAX_PULSE_US,
    SERVO_CENTER_ANGLE,
    SERVO_MIN_ANGLE,
    SERVO_MAX_ANGLE,
    WHEELBASE,
    TRACK_WIDTH,
    MAX_STEERING_ANGLE_DEG,
)

try:
    from config import STEERING_INVERTED
except ImportError:
    STEERING_INVERTED = False

try:
    from config import SERVO_TRIM_DEG
except ImportError:
    SERVO_TRIM_DEG = 0.0


class SteeringDriver:
    """Steering control with Ackermann geometry over a PCA9685."""

    def __init__(self):
        i2c = ExtendedI2C(3)
        pca = adafruit_pca9685.PCA9685(i2c)
        pca.frequency = PCA9685_PWM_FREQ

        self._servo = adafruit_servo.Servo(
            pca.channels[SERVO_CHANNEL],
            min_pulse=SERVO_MIN_PULSE_US,
            max_pulse=SERVO_MAX_PULSE_US,
        )
        self._current_angle = SERVO_CENTER_ANGLE
        self.center()

    def center(self):
        """Wheels straight ahead (geometric centre plus the mechanical trim)."""
        self._servo.angle = self._physical(SERVO_CENTER_ANGLE)
        self._current_angle = SERVO_CENTER_ANGLE

    def _physical(self, angle_deg: float) -> float:
        """
        Logical angle -> angle actually written to the servo.

        Applies the steering inversion and then SERVO_TRIM_DEG, which absorbs the
        mechanical offset between the servo's electrical centre and the angle at
        which the wheels really point straight. Without it a logical 90 deg can
        still pull to one side, and the PID has no way to know: it is told the
        lane error is zero, outputs zero correction, and the car drifts.

        The trim is applied AFTER the inversion so it is a fixed offset in servo
        space, independent of which way the linkage is mounted.
        """
        physical = (2.0 * SERVO_CENTER_ANGLE - angle_deg) if STEERING_INVERTED else angle_deg
        return max(0.0, min(180.0, physical + SERVO_TRIM_DEG))

    def set_angle(self, angle_deg: float):
        """
        Move the servo to the given angle.

        Parameters
        ----------
        angle_deg : float
            Range [SERVO_MIN_ANGLE, SERVO_MAX_ANGLE].
            90 deg = straight, <90 = left, >90 = right.

        If `STEERING_INVERTED=True` the logical angle is flipped when writing
        to the servo, but `current_angle` returns the logical value expected
        by consumers (signals/PID/etc.).
        """
        angle_deg = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, float(angle_deg)))
        self._servo.angle = self._physical(angle_deg)
        self._current_angle = angle_deg

    def steer_from_error(self, lane_error_px: float, kp: float = 0.09) -> float:
        """
        Convert the lane error (pixels) into a servo angle.
        Uses simple proportional control -- the PID calls it externally and
        passes the already-computed correction as `steer_delta_deg`.

        Returns the applied angle.
        """
        delta = kp * lane_error_px
        angle = SERVO_CENTER_ANGLE + delta
        self.set_angle(angle)
        return self._current_angle

    def set_steering_angle(self, steering_deg: float):
        """
        Ackermann interface: receives the CENTRAL steering angle
        (+ = right, - = left) in degrees and converts it to a servo
        position.

        The exact Ackermann correction for each wheel is computed
        internally for reference, although the physical servo applies
        only the central angle.
        """
        steering_deg = max(
            -MAX_STEERING_ANGLE_DEG,
            min(MAX_STEERING_ANGLE_DEG, steering_deg)
        )

        inner, outer = self._ackermann_angles(steering_deg)

        servo_angle = SERVO_CENTER_ANGLE + steering_deg
        self.set_angle(servo_angle)
        return inner, outer

    @staticmethod
    def _ackermann_angles(
        center_deg: float,
    ) -> tuple[float, float]:
        """
        Compute the individual wheel angles (pure Ackermann model).

        Returns
        -------
        (inner_deg, outer_deg)
            inner = wheel on the turn side (tighter)
            outer = wheel opposite the turn (wider)
        """
        if abs(center_deg) < 0.1:
            return 0.0, 0.0

        delta = math.radians(center_deg)
        R = WHEELBASE / math.tan(delta)

        sign = math.copysign(1, center_deg)
        R_inner = abs(R) - sign * TRACK_WIDTH / 2
        R_outer = abs(R) + sign * TRACK_WIDTH / 2

        inner_deg = math.degrees(math.atan(WHEELBASE / R_inner)) * sign if R_inner != 0 else 90.0 * sign
        outer_deg = math.degrees(math.atan(WHEELBASE / R_outer)) * sign if R_outer != 0 else 0.0

        return inner_deg, outer_deg

    @staticmethod
    def turning_radius(steering_deg: float) -> float:
        """Turning radius in metres for a central angle (bicycle model)."""
        if abs(steering_deg) < 0.1:
            return float("inf")
        return WHEELBASE / math.tan(math.radians(abs(steering_deg)))

    @property
    def current_angle(self) -> float:
        return self._current_angle

    @property
    def steering_deviation(self) -> float:
        """Deviation from centre in degrees (+ right, - left)."""
        return self._current_angle - SERVO_CENTER_ANGLE
