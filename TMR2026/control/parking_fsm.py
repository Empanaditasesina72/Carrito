"""Perpendicular (battery) PARKING manoeuvre.

State sub-machine for Test 3 of the "Sim2Real Validation" specification:
   VISION -> AUTONOMOUS -> PARKING_SEARCH -> PARKING_MANEUVER -> PARKED

Design:
  - PARKING_SEARCH: drives slowly along the lane looking for the gap. It is
    detected with the side/front ToF (or by time if there is no sensor).
  - PARKING_MANEUVER: open-loop, time-based manoeuvre (just like the car's
    real parking): steers towards config.py:PARK_SIDE and drives in
    perpendicular to the bay, then straightens.
  - PARKED: motor at 0, parked.

Guarantees (like the rest of the project):
  - Uses time.monotonic(), NEVER time.sleep() -- the loop never blocks.
  - brake() is an instantaneous cut; it is not modified.

Timings are calibratable from config.py (PARK_*). If absent, defaults apply.
"""

import time
from enum import Enum, auto

try:
    from config import (
        PARK_SIDE, PARK_SEARCH_SPEED, PARK_MANEUVER_SPEED,
        PARK_REVERSE_LOCK_SEC, PARK_REVERSE_STRAIGHT_SEC,
        SERVO_CENTER_ANGLE, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE,
    )
except ImportError:
    PARK_SIDE = "right"
    PARK_SEARCH_SPEED = 15
    PARK_MANEUVER_SPEED = 12
    PARK_REVERSE_LOCK_SEC = 2.5
    PARK_REVERSE_STRAIGHT_SEC = 1.0
    SERVO_CENTER_ANGLE = 90.0
    SERVO_MIN_ANGLE = 58.0
    SERVO_MAX_ANGLE = 122.0


class ParkingState(Enum):
    PARKING_SEARCH   = auto()
    PARKING_MANEUVER = auto()
    PARKED           = auto()


class ParkingFSM:
    """
    Battery parking. Usage::

        pk = ParkingFSM(motor, steering)
        pk.activate()
        while running:
            pk.lidar_mm = sensor.front_mm   # optional (gap detection)
            pk.update(dt)                    # 50 Hz, non-blocking
            if pk.state == ParkingState.PARKED: break
    """

    SEARCH_MIN_S   = 1.5
    SEARCH_MAX_S   = 4.0
    TURN_IN_S      = 2.2
    STRAIGHTEN_S   = 1.2
    GAP_FRONT_MM   = 600

    SEARCH_SPEED   = float(PARK_SEARCH_SPEED)
    MANEUVER_SPEED = float(PARK_MANEUVER_SPEED)
    TURN_IN_ANGLE  = SERVO_MAX_ANGLE if PARK_SIDE == "right" else SERVO_MIN_ANGLE

    def __init__(self, motor, steering):
        self.motor = motor
        self.steering = steering

        self.lidar_mm = None
        self._state = ParkingState.PARKING_SEARCH
        self._t_state = 0.0
        self._active = False
        self._man_phase = 0

    def activate(self):
        self._state = ParkingState.PARKING_SEARCH
        self._t_state = time.monotonic()
        self._man_phase = 0
        self._active = True
        print(f"[PARK] Parking ENABLED -> PARKING_SEARCH (bay on the {PARK_SIDE})")

    def deactivate(self):
        self._active = False
        self.motor.brake()
        self.steering.set_angle(SERVO_CENTER_ANGLE)

    @property
    def state(self) -> ParkingState:
        return self._state

    @property
    def done(self) -> bool:
        return self._state == ParkingState.PARKED

    def _elapsed(self) -> float:
        return time.monotonic() - self._t_state

    def _go(self, new_state: ParkingState):
        print(f"[PARK] {self._state.name} -> {new_state.name}")
        self._state = new_state
        self._t_state = time.monotonic()

    def update(self, dt: float):
        if not self._active:
            return

        if self._state == ParkingState.PARKING_SEARCH:
            self._do_search()
        elif self._state == ParkingState.PARKING_MANEUVER:
            self._do_maneuver()
        elif self._state == ParkingState.PARKED:
            self.motor.brake()
            self.steering.set_angle(SERVO_CENTER_ANGLE)

    def _do_search(self):
        self.steering.set_angle(SERVO_CENTER_ANGLE)
        self.motor.set_speed(self.SEARCH_SPEED)

        gap = (self._elapsed() >= self.SEARCH_MIN_S
               and self.lidar_mm is not None
               and self.lidar_mm >= self.GAP_FRONT_MM)
        if gap or self._elapsed() >= self.SEARCH_MAX_S:
            self._man_phase = 0
            self._go(ParkingState.PARKING_MANEUVER)

    def _do_maneuver(self):
        if self._man_phase == 0:
            self.steering.set_angle(self.TURN_IN_ANGLE)
            self.motor.set_speed(self.MANEUVER_SPEED)
            if self._elapsed() >= self.TURN_IN_S:
                self._man_phase = 1
                self._t_state = time.monotonic()
        elif self._man_phase == 1:
            self.steering.set_angle(SERVO_CENTER_ANGLE)
            self.motor.set_speed(self.MANEUVER_SPEED * 0.7)
            if self._elapsed() >= self.STRAIGHTEN_S:
                self.motor.brake()
                self.steering.set_angle(SERVO_CENTER_ANGLE)
                self._go(ParkingState.PARKED)
                print("[PARK] Battery-parked (PARKED)")
