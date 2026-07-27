"""IBT-2 motor control with anti voltage-sag soft-start (TMR 2026).

Pi 5 COMPATIBILITY NOTE:
  RPi.GPIO does not support hardware PWM on the Pi 5 (kernel 6.1+).
  This module tries RPi.GPIO (software PWM) first.
  If that fails it automatically uses lgpio (native Pi 5 hardware PWM).
  On Pi 4 / Pi 3, RPi.GPIO works normally.

IBT-2 wiring:
  RPWM -> GPIO 18  (forward)
  LPWM -> GPIO 13  (reverse)
  R_EN + L_EN -> physical 3.3V (always enabled, no GPIO control)

Soft-Start:
  An internal thread (50 Hz) raises the duty by at most _SLEW_UP % per tick.
  Ramp-down is 4x faster (safety).
  brake() cuts INSTANTANEOUSLY to 0 without going through the ramp.
"""

import threading
import time
from typing import Optional

# lgpio FIRST. On a Pi 5 RPi.GPIO has no hardware PWM and falls back to a
# Python-timed software one, whose effective duty collapses when the CPU is
# busy -- and this vehicle runs YOLO on the CPU. Measured 2026-07-27: driving
# the H-bridge directly through lgpio at 45 % moved the car, while the same 45 %
# through MotorDriver on RPi.GPIO left it stationary for seven seconds with the
# detector running. The vehicle only ever moved in tests where nothing else was
# competing for the core.
#
# lgpio drives the BCM2712 PWM peripheral, so its duty is immune to load.
# CLAUDE.md already documented lgpio as the intended path with RPi.GPIO as the
# fallback; this module had the order inverted.
try:
    import lgpio as _lgpio
    _lgpio.gpiochip_close(_lgpio.gpiochip_open(4))   # prove chip 4 is reachable
    _BACKEND = "lgpio"
except Exception:
    try:
        import RPi.GPIO as _GPIO
        _GPIO.setmode(_GPIO.BCM)
        _GPIO.setwarnings(False)
        _BACKEND = "RPi.GPIO"
        print("[MOTOR] WARNING: lgpio no disponible -> PWM POR SOFTWARE. "
              "La potencia real caera cuando el CPU este cargado.")
    except (ImportError, RuntimeError):
        _BACKEND = "mock"

_PWM_FREQ = 1_000


class MotorDriver:
    """
    High-level interface for the IBT-2 H-bridge.

    Usage::

        m = MotorDriver()
        m.set_speed(35.0)   # 35 % power forward
        m.brake()           # instantaneous cut to 0
        m.cleanup()
    """

    MAX_DUTY   = 100.0
    _SLEW_UP   = 2.0
    _SLEW_DOWN = 8.0
    _TICK_S    = 0.02

    def __init__(self, pin_rpwm: int = 18, pin_lpwm: int = 13):
        self._pin_r = pin_rpwm
        self._pin_l = pin_lpwm
        self._current = 0.0
        self._target  = 0.0
        self._lock    = threading.Lock()
        self._running = True

        self._init_hw()

        self._thread = threading.Thread(
            target=self._ramp_loop, name="MotorRamp", daemon=True
        )
        self._thread.start()
        print(f"[MOTOR] Backend: {_BACKEND}  RPWM=GPIO{pin_rpwm}  LPWM=GPIO{pin_lpwm}")


    def set_speed(self, duty: float) -> None:
        """
        Set the target speed. The internal ramp reaches it gradually
        (soft-start). duty in [-MAX_DUTY, +MAX_DUTY];
        positive = forward, negative = reverse.
        """
        duty = max(-self.MAX_DUTY, min(self.MAX_DUTY, float(duty)))
        with self._lock:
            self._target = duty

    def brake(self) -> None:
        """
        Instantaneous brake: cuts the PWM to EXACTLY 0.
        Bypasses the ramp. Call this in FRENADO/ESPERA.
        """
        with self._lock:
            self._target  = 0.0
            self._current = 0.0
        self._apply_hw(0.0)

    def kick(self, duty: float, seconds: float = 0.25) -> None:
        """
        Break static friction with a brief high-duty pulse, bypassing the ramp.

        A loaded vehicle at a low cruise duty can stall on start: the PWM is
        applied, the motor hums, and the wheels never turn, because the duty
        that sustains motion is well below the one needed to start it. This
        forces the duty for a moment so the car is already rolling when normal
        control takes over -- rolling friction is far lower than the static
        friction that blocked it, so the low cruise duty is enough to keep it
        going even though it could not start it.

        Blocks for `seconds`. Leaves the target where the pulse put it; the
        next set_speed() or brake() takes over from there.
        """
        duty = max(-self.MAX_DUTY, min(self.MAX_DUTY, float(duty)))
        with self._lock:
            self._target  = duty
            self._current = duty
        self._apply_hw(duty)
        time.sleep(max(0.0, float(seconds)))

    @property
    def current_duty(self) -> float:
        """Current duty cycle (the one being applied to the H-bridge)."""
        with self._lock:
            return self._current

    def cleanup(self) -> None:
        """Release GPIO. Call on shutdown."""
        self._running = False
        self.brake()
        time.sleep(self._TICK_S * 2)
        if _BACKEND == "RPi.GPIO":
            try:
                self._pwm_r.stop()
                self._pwm_l.stop()
                _GPIO.cleanup([self._pin_r, self._pin_l])
            except Exception:
                pass
        elif _BACKEND == "lgpio":
            try:
                _lgpio.gpiochip_close(self._h)
            except Exception:
                pass


    def _init_hw(self) -> None:
        if _BACKEND == "RPi.GPIO":
            _GPIO.setup(self._pin_r, _GPIO.OUT)
            _GPIO.setup(self._pin_l, _GPIO.OUT)
            self._pwm_r = _GPIO.PWM(self._pin_r, _PWM_FREQ)
            self._pwm_l = _GPIO.PWM(self._pin_l, _PWM_FREQ)
            self._pwm_r.start(0)
            self._pwm_l.start(0)

        elif _BACKEND == "lgpio":
            self._h = _lgpio.gpiochip_open(4)
            _lgpio.gpio_claim_output(self._h, self._pin_r)
            _lgpio.gpio_claim_output(self._h, self._pin_l)
            _lgpio.tx_pwm(self._h, self._pin_r, _PWM_FREQ, 0)
            _lgpio.tx_pwm(self._h, self._pin_l, _PWM_FREQ, 0)


    def _ramp_loop(self) -> None:
        """Move _current toward _target with a ramp limit."""
        while self._running:
            with self._lock:
                target  = self._target
                current = self._current

            diff = target - current
            if diff > 0:
                step = min(diff,  self._SLEW_UP)
            elif diff < 0:
                step = max(diff, -self._SLEW_DOWN)
            else:
                step = 0.0

            if step != 0.0:
                new_duty = current + step
                with self._lock:
                    self._current = new_duty
                self._apply_hw(new_duty)

            time.sleep(self._TICK_S)


    def _apply_hw(self, duty: float) -> None:
        """Write the duty cycle to the H-bridge. No ramp."""
        r = max(0.0, min(100.0,  duty))
        l = max(0.0, min(100.0, -duty))

        if _BACKEND == "RPi.GPIO":
            self._pwm_r.ChangeDutyCycle(r)
            self._pwm_l.ChangeDutyCycle(l)

        elif _BACKEND == "lgpio":
            _lgpio.tx_pwm(self._h, self._pin_r, _PWM_FREQ, r)
            _lgpio.tx_pwm(self._h, self._pin_l, _PWM_FREQ, l)
