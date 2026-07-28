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
    from config import STEER_HEADING_GAIN as _CFG_STEER_HEADING_GAIN
except ImportError:
    _CFG_STEER_HEADING_GAIN = 0.0

try:
    from config import (LANE_DEADBAND_PX as _CFG_DEADBAND,
                        LANE_DEADBAND_EXIT_PX as _CFG_DEADBAND_EXIT)
except ImportError:
    _CFG_DEADBAND, _CFG_DEADBAND_EXIT = 0.0, 0.0

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
    HEADING_GAIN    = _CFG_STEER_HEADING_GAIN
    DEADBAND_PX      = _CFG_DEADBAND
    DEADBAND_EXIT_PX = _CFG_DEADBAND_EXIT
    MIN_SERVO_DELTA  = 0.5   # deg; skip writes smaller than this (anti-jitter)

    SERVO_CENTER    = _CFG_SERVO_CENTER
    SERVO_MIN       = _CFG_SERVO_MIN
    SERVO_MAX       = _CFG_SERVO_MAX

    SIGNAL_DIR_THRESH_DEG = 12.0

    SIGN_BBOX_STOP_MM = 320

    # If the sign disappears while the last known distance was this close, treat
    # it as ARRIVAL and brake -- do not resume cruise. Measured 2026-07-27: at
    # under ~30 cm the octagon sits ~45 deg off the camera axis (HFOV is +/-33)
    # and leaves the frame, so `sign_visible` drops exactly when the car reaches
    # it. The old behaviour then transitioned back to CRUCERO and ACCELERATED
    # into the sign; one trial ended 140 mm from it without ever braking.
    SIGN_LOST_NEAR_MM = 600

    # How long the sign must stay UNSEEN before PRECAUCION gives up and returns
    # to cruise. Without it a detection sitting near its 0.55 gate -- 0.56-0.57
    # is what this sign reads at 1 m -- toggles the state every few frames, and
    # the log fills with CRUCERO<->PRECAUCION while the car surges and crawls.
    SIGN_LOST_GRACE_S = 0.6

    # Seconds to keep rolling AFTER the sign is lost at close range, before
    # braking. This exists because of a geometric limit, not a tuning choice.
    #
    # The sign stands beside the lane, roughly 28 cm off the camera axis, and the
    # lens covers about +-33 deg. So it leaves the frame at
    #     d = 0.28 / tan(33 deg) ~= 43 cm
    # and SIGN_BBOX_STOP_MM = 320 can never fire: the car is blind to the sign
    # before it ever gets that close. Both measured runs braked on the lost-sign
    # rule instead, at 362 and 412 mm against a 270 mm setpoint -- a +117 mm mean
    # error that no threshold change can remove, because the threshold is never
    # reached.
    #
    # Coasting closes that gap by dead reckoning over the blind stretch. At the
    # measured 15 cm/s (20 % duty advanced 20.8 cm in 1.4 s), 0.6 s is ~9 cm,
    # taking the mean from 387 to roughly 300 mm. Set 0.0 to brake immediately.
    #
    # CUT 0.6 -> 0.15 on 2026-07-28 for the city track, where the sign was moved
    # against the lane edge. Post centre is now ~20 cm off the camera axis, not
    # 28, so with tan(HFOV/2) = 320/490 the octagon centre only leaves the frame
    # at 0.20 * 1.531 = 306 mm -- past the 320 mm gate. The blind stretch shrank
    # from ~13 cm to ~3.6 cm, and 0.6 s would now dead-reckon ~13 cm straight
    # INTO the sign. 0.15 s is that 3.6 cm at the PRECAUCION speed.
    SIGN_LOST_COAST_S = 0.15

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
        self.lane_heading:    float           = 0.0
        self.lidar_mm:        Optional[float] = None
        self.sign_visible:    bool            = False
        self.sign_distance_mm:Optional[float] = None

        self._state          = FSMState.CRUCERO
        self._espera_start   = 0.0
        self._last_sign_mm: Optional[float] = None
        self._correcting     = False   # deadband hysteresis state
        self._sign_lost_at: Optional[float] = None
        self._coast_until: Optional[float] = None
        self._last_cmd_angle = self.SERVO_CENTER
        self._cooldown_until = 0.0
        self._resume_speed   = 0.0

        self._active = False


    def activate(self) -> None:
        """Activate autonomous mode."""
        self.pid.reset()
        self._state        = FSMState.CRUCERO
        self._resume_speed = 0.0
        self._last_sign_mm = None
        self._sign_lost_at = None
        self._coast_until  = None
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
        if self.sign_distance_mm is not None:
            self._last_sign_mm = self.sign_distance_mm

        if self.sign_visible:
            self._sign_lost_at = None
        elif self._sign_lost_at is None:
            self._sign_lost_at = time.monotonic()

        lost_long_enough = (self._sign_lost_at is not None and
                            time.monotonic() - self._sign_lost_at
                            >= self.SIGN_LOST_GRACE_S)

        if not self.sign_visible and lost_long_enough:
            near = (self._last_sign_mm is not None
                    and self._last_sign_mm <= self.SIGN_LOST_NEAR_MM)
            if near:
                if self.SIGN_LOST_COAST_S > 0.0 and self._coast_until is None:
                    self._coast_until = (time.monotonic()
                                         + self.SIGN_LOST_COAST_S)
                    print(f"[FSM] Sign lost at {self._last_sign_mm:.0f}mm -> "
                          f"coasting {self.SIGN_LOST_COAST_S:.2f}s "
                          f"(blind, sign is outside the lens)")
                if self._coast_until is not None:
                    if time.monotonic() < self._coast_until:
                        self.motor.set_speed(self.PRECAUCION_PWM)
                        return
                    print(f"[FSM] Coast done -> braking")
                self._transition(FSMState.FRENADO)
            else:
                self._last_sign_mm = None
                self._sign_lost_at = None
                self._coast_until  = None
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
            # Forget the last close-range reading: the stop is served, and a
            # stale "near" memory would re-brake the car the moment the passed
            # sign flickers during the cooldown drive-by.
            self._last_sign_mm = None
            self._coast_until  = None
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

        # Deadband with hysteresis. Inside the band the servo holds CENTRE --
        # with SERVO_TRIM_DEG calibrated, centre is physically straight -- and
        # the PID stays reset so no stale integral fires on re-engage. The car
        # is allowed to wander 25 px (3.7 cm); only when it leaves that band
        # does the controller engage, and it then keeps correcting until the
        # error is back under the inner 12 px so the boundary cannot chatter.
        abs_err = abs(self.lane_error)
        if self.DEADBAND_PX > 0.0 and self.lane_conf >= self.MIN_LANE_CONF:
            if self._correcting:
                if abs_err <= self.DEADBAND_EXIT_PX:
                    self._correcting = False
            elif abs_err > self.DEADBAND_PX:
                self._correcting = True

            if not self._correcting:
                self.pid.reset()
                self._write_angle(self.SERVO_CENTER)
                return

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

        # Heading feedforward (Stanley-style second term). Lateral error alone
        # cannot damp a heading disturbance: the controller only reacts after the
        # car has already translated sideways, so a run that starts with any yaw
        # -- or picks one up from a trim bias -- weaves or drifts out before the
        # lateral term converges. lane_heading > 0 means the lines lean right in
        # the BEV, i.e. the nose points left of the road, i.e. steer right --
        # which in the verified convention is an angle ABOVE centre, hence `+`.
        #
        # STEER_HEADING_GAIN defaults to 0.0 (term disabled) because the sign of
        # the heading estimate has not yet been confirmed on the car, and with
        # the sign wrong this is positive feedback that would throw the car off
        # the track. To enable: place the car rotated ~15 deg on the lane, run
        # tools/diag_track.py, confirm the reported heading sign matches the
        # rotation (nose left => heading positive), then set the gain (start 1.5).
        if self.lane_conf >= self.MIN_LANE_CONF and self.HEADING_GAIN > 0.0:
            angle += self.HEADING_GAIN * self.lane_heading
        angle = max(self.SERVO_MIN, min(self.SERVO_MAX, angle))
        self._write_angle(angle)

    def _write_angle(self, angle: float) -> None:
        """Write the servo only when the command moved enough to matter.
        Sub-half-degree chatter wears the servo and wobbles the car without
        changing the trajectory."""
        if abs(angle - self._last_cmd_angle) < self.MIN_SERVO_DELTA:
            return
        self.steering.set_angle(angle)
        self._last_cmd_angle = angle

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
