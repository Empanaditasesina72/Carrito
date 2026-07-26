"""Autonomous Finite-State Machine for the TMR 2026 vehicle (5 states).

States and transitions:

    CRUCERO (cruise, max speed)
        | sign detected by YOLO
        v
    PRECAUCION (caution, 20% speed)
        | lidar <= 30 cm
        v
    FRENADO (braking, motor = 0, instantaneous)
        v
    ESPERA (wait, motor = 0, exactly 5 s)
        | 5.0 s elapsed
        v
    REANUDAR (resume, own soft-start)
        | ramp complete
        v
    CRUCERO

Key guarantees:
  - FRENADO -> motor.brake() = duty EXACTLY 0 (no 5% residual).
  - ESPERA uses time.monotonic() -- it never blocks the vision thread.
  - REANUDAR applies its own soft-start (independent of the motor ramp).
  - 3 s cooldown after REANUDAR -> ignores the same sign while passing it.
  - Steering (servo) is updated in EVERY state without exception.
"""

import time
from enum import Enum, auto
from typing import Optional

try:
    from hardware.signals import SignalMode
except Exception:
    class SignalMode:
        OFF = "OFF"; LEFT = "LEFT"; RIGHT = "RIGHT"; HAZARD = "HAZARD"

try:
    from config import LANE_MIN_CONFIDENCE as _CFG_LANE_MIN_CONF
except ImportError:
    _CFG_LANE_MIN_CONF = 0.20

try:
    from config import (
        SERVO_CENTER_ANGLE as _CFG_SERVO_CENTER,
        SERVO_MIN_ANGLE    as _CFG_SERVO_MIN,
        SERVO_MAX_ANGLE    as _CFG_SERVO_MAX,
    )
except ImportError:
    _CFG_SERVO_CENTER, _CFG_SERVO_MIN, _CFG_SERVO_MAX = 90.0, 58.0, 122.0


class FSMState(Enum):
    CRUCERO    = auto()
    PRECAUCION = auto()
    FRENADO    = auto()
    ESPERA     = auto()
    REANUDAR   = auto()


class AutonomousFSM:
    """TMR 2026 autonomous controller -- Finite-State Machine.

    Requires a MotorDriver, a SteeringDriver and a PIDController.

    Usage::

        fsm = AutonomousFSM(motor, steering, pid)
        fsm.activate()
        while running:
            fsm.lane_error   = lane_result.error_px
            fsm.lane_conf    = lane_result.confidence
            fsm.lidar_mm     = sensor.front_mm
            fsm.sign_visible = sign_detector.has_any_sign()
            fsm.update(dt)   # call at 50 Hz
        fsm.deactivate()
    """

    MAX_AUTO_PWM    = 42.0
    PRECAUCION_PWM  = 20.0
    RESUME_STEP_PWM = 1.5

    LIDAR_STOP_MM   = 300
    ESPERA_S        = 5.0
    COOLDOWN_S      = 3.0
    MIN_LANE_CONF   = _CFG_LANE_MIN_CONF

    SERVO_CENTER    = _CFG_SERVO_CENTER
    SERVO_MIN       = _CFG_SERVO_MIN
    SERVO_MAX       = _CFG_SERVO_MAX

    SIGNAL_DIR_THRESH_DEG = 12.0

    SIGN_BBOX_STOP_MM = 320

    def __init__(self, motor, steering, pid, signals=None, brake_light=None):
        """
        Parameters
        ----------
        motor       : MotorDriver
        steering    : SteeringDriver
        pid         : PIDController (setpoint=0, output=correction angle in degrees)
        signals     : TurnSignals (optional) -- turn signals / hazard
        brake_light : BrakeLight (optional) -- brake light
        """
        self.motor       = motor
        self.steering    = steering
        self.pid         = pid
        self.signals     = signals
        self.brake_light = brake_light

        self.lane_error:      float           = 0.0
        self.lane_conf:       float           = 0.0
        self.lidar_mm:        Optional[float] = None
        self.sign_visible:    bool            = False
        self.sign_distance_mm:Optional[float] = None

        self._state          = FSMState.CRUCERO
        self._espera_start   = 0.0
        self._cooldown_until = 0.0
        self._resume_speed   = 0.0

        self._active = False


    def activate(self) -> None:
        """Activate autonomous mode."""
        self.pid.reset()
        self._state        = FSMState.CRUCERO
        self._resume_speed = 0.0
        self._active       = True
        self._apply_lights()
        print("[FSM] Autonomous mode ENABLED")

    def deactivate(self) -> None:
        """Brake and disable autonomous mode."""
        self._active = False
        self.motor.brake()
        self.steering.center()
        if self.signals is not None:
            self.signals.set_mode(SignalMode.OFF)
        if self.brake_light is not None:
            self.brake_light.off()
        print("[FSM] Autonomous mode DISABLED")

    @property
    def state(self) -> FSMState:
        return self._state


    def update(self, dt: float) -> None:
        """
        Call ONCE per main-loop iteration (50 Hz recommended).
        dt: elapsed time in seconds since the previous call.

        ALWAYS updates the servo, even while the motor is stopped.
        """
        if not self._active:
            if self.signals is not None:
                self.signals.tick()
            return

        self._apply_steering(dt)

        match self._state:
            case FSMState.CRUCERO:
                self._do_crucero()
            case FSMState.PRECAUCION:
                self._do_precaucion()
            case FSMState.FRENADO:
                self._do_frenado()
            case FSMState.ESPERA:
                self._do_espera()
            case FSMState.REANUDAR:
                self._do_reanudar()

        self._apply_lights()

        if self.signals is not None:
            self.signals.tick()


    def _do_crucero(self) -> None:
        if self.lane_conf < self.MIN_LANE_CONF:
            self.motor.brake()
            return

        if self.sign_visible and time.monotonic() >= self._cooldown_until:
            self._transition(FSMState.PRECAUCION)
            return

        self.motor.set_speed(self.MAX_AUTO_PWM)

    def _do_precaucion(self) -> None:
        if not self.sign_visible:
            self._transition(FSMState.CRUCERO)
            return

        lidar_close = (self.lidar_mm is not None
                       and self.lidar_mm <= self.LIDAR_STOP_MM)
        bbox_close  = (self.sign_distance_mm is not None
                       and self.sign_distance_mm <= self.SIGN_BBOX_STOP_MM)

        if lidar_close or bbox_close:
            source = "lidar" if lidar_close else f"camera {self.sign_distance_mm:.0f}mm"
            print(f"[FSM] Braking ({source})")
            self._transition(FSMState.FRENADO)
            return

        self.motor.set_speed(self.PRECAUCION_PWM)

    def _do_frenado(self) -> None:
        self.motor.brake()
        self._transition(FSMState.ESPERA)

    def _do_espera(self) -> None:
        self.motor.brake()

        elapsed = time.monotonic() - self._espera_start
        if elapsed >= self.ESPERA_S:
            self._transition(FSMState.REANUDAR)
            print(f"[FSM] ESPERA complete ({elapsed:.2f} s) -> REANUDAR")

    def _do_reanudar(self) -> None:
        self._resume_speed = min(
            self._resume_speed + self.RESUME_STEP_PWM,
            self.MAX_AUTO_PWM,
        )
        self.motor.set_speed(self._resume_speed)

        if self._resume_speed >= self.MAX_AUTO_PWM:
            self._transition(FSMState.CRUCERO)


    def _apply_steering(self, dt: float) -> None:
        """
        Compute the PID correction and apply it to the servo.
        Runs in every state -- even with the motor stopped the car must
        point in the correct direction.
        """
        if self._state in (FSMState.FRENADO, FSMState.ESPERA):
            self.pid.reset()

        if self.lane_conf >= self.MIN_LANE_CONF:
            correction = self.pid.compute(self.lane_error, dt)
        else:
            correction = 0.0

        # MINUS, not plus. PIDController computes `setpoint - measurement`, so
        # handing it the lane error already negates it: a positive lane_error
        # comes back as a negative correction. Adding that to SERVO_CENTER
        # steered the car the WRONG WAY and the two negations have to cancel.
        #
        # The geometry, measured on the car 2026-07-26:
        #   lane_pipeline sets error_px = lane_cx - frame_cx, so error > 0 means
        #   the lane centre appears RIGHT of the image centre, which means the car
        #   sits LEFT of the lane -- and it must steer RIGHT to recover. Steering
        #   right is a logical angle above SERVO_CENTER (90 = straight, <90 = left,
        #   >90 = right; confirmed physically -- logical 58 turns the wheels left --
        #   and independently by _apply_lights below, which lights LEFT when
        #   current_angle - SERVO_CENTER is negative).
        #
        # With `+` a car displaced left was commanded further left, so the first
        # correction drove it out of the lane instead of back into it.
        # SteeringDriver.steer_from_error() has always carried the right
        # convention (angle = centre + kp * error) but nothing calls it, which is
        # why the contradiction sat here unnoticed.
        angle = self.SERVO_CENTER - correction
        angle = max(self.SERVO_MIN, min(self.SERVO_MAX, angle))
        self.steering.set_angle(angle)

    def _transition(self, new_state: FSMState) -> None:
        old = self._state
        self._state = new_state
        print(f"[FSM] {old.name} -> {new_state.name}")

        if new_state == FSMState.FRENADO:
            self.motor.brake()

        elif new_state == FSMState.ESPERA:
            self._espera_start = time.monotonic()
            self.motor.brake()

        elif new_state == FSMState.REANUDAR:
            self._resume_speed   = 0.0
            self._cooldown_until = time.monotonic() + self.COOLDOWN_S
            print(f"[FSM] Cooldown active for {self.COOLDOWN_S:.1f} s")
            self.pid.reset()

        elif new_state == FSMState.CRUCERO:
            self.pid.reset()

        self._apply_lights()

    def _apply_lights(self) -> None:
        """
        State -> lights mapping (called every tick so the turn signals
        follow the servo angle in CRUCERO/REANUDAR):
          CRUCERO    -> signals LEFT/RIGHT/OFF by angle,  brake OFF
          PRECAUCION -> signals HAZARD,                   brake OFF
          FRENADO    -> signals HAZARD,                   brake ON
          ESPERA     -> signals HAZARD,                   brake ON
          REANUDAR   -> signals LEFT/RIGHT/OFF by angle,  brake OFF
        """
        if self.signals is not None:
            if self._state in (FSMState.PRECAUCION, FSMState.FRENADO, FSMState.ESPERA):
                self.signals.set_mode(SignalMode.HAZARD)
            elif self._state in (FSMState.CRUCERO, FSMState.REANUDAR):
                deviation = self.steering.current_angle - self.SERVO_CENTER
                if   deviation < -self.SIGNAL_DIR_THRESH_DEG:
                    self.signals.set_mode(SignalMode.LEFT)
                elif deviation > +self.SIGNAL_DIR_THRESH_DEG:
                    self.signals.set_mode(SignalMode.RIGHT)
                else:
                    self.signals.set_mode(SignalMode.OFF)
            else:
                self.signals.set_mode(SignalMode.OFF)

        if self.brake_light is not None:
            if self._state in (FSMState.FRENADO, FSMState.ESPERA):
                self.brake_light.on()
            else:
                self.brake_light.off()
