"""Generic PID controller with anti-windup.

Features:
  - Anti-windup via integrator clamping.
  - Derivative on measurement (not on error) to avoid "derivative kick"
    when the setpoint changes abruptly.
  - Hot reset without losing the previous I/D state.
"""

import time


class PIDController:
    """Real-time discrete PID.

    Parameters
    ----------
    kp, ki, kd : float
        Proportional, integral and derivative gains.
    setpoint : float
        Target value (can be changed at runtime).
    output_limits : (min, max)
        Output saturation.
    integral_limits : (min, max)
        Integral accumulator limits (anti-windup).
    derivative_on_measurement : bool
        If True, the derivative is computed on the measurement (recommended).
        If False, it is computed on the error (classic behaviour).
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        setpoint: float = 0.0,
        output_limits: tuple[float, float] = (-100.0, 100.0),
        integral_limits: tuple[float, float] = (-50.0, 50.0),
        derivative_on_measurement: bool = True,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self.derivative_on_measurement = derivative_on_measurement

        self._integral   = 0.0
        self._last_input = 0.0
        self._last_error = 0.0
        self._last_time  = time.monotonic()
        # False until the first compute() after construction or reset(), so the
        # derivative term can be seeded from the real measurement instead of
        # differentiating against a stale zero. See compute().
        self._primed     = False

        self.last_error  = 0.0
        self.last_p      = 0.0
        self.last_i      = 0.0
        self.last_d      = 0.0
        self.last_output = 0.0

    def compute(self, measurement: float, dt: float | None = None) -> float:
        """
        Compute the PID output.

        Parameters
        ----------
        measurement : float
            Current value of the controlled variable.
        dt : float, optional
            Time interval in seconds. If None, the real time elapsed since
            the previous call is used.

        Returns
        -------
        float
            Saturated controller output.
        """
        now = time.monotonic()
        if dt is None:
            dt = now - self._last_time
        if dt <= 0:
            dt = 1e-4
        self._last_time = now

        error = self.setpoint - measurement

        p_out = self.kp * error

        self._integral += error * dt
        self._integral = max(
            self.integral_limits[0],
            min(self.integral_limits[1], self._integral)
        )
        i_out = self.ki * self._integral

        # Seed the derivative history on the first call after construction or
        # reset(). Without this, _last_input is 0.0 and the first sample is
        # differentiated against it: at the steering gains (kd=0.025) a 50 px lane
        # error one tick after a reset produced -0.025 * 50/0.02 = -62 deg, which
        # saturates the output and slams the servo to a limit for one tick. That
        # fires on every brake-and-resume, because _apply_steering() resets the PID
        # throughout FRENADO and ESPERA. A derivative of zero on the first sample
        # is the correct reading: no history means no measured rate of change.
        if not self._primed:
            self._last_input = measurement
            self._last_error = error
            self._primed = True

        if self.derivative_on_measurement:
            d_out = -self.kd * (measurement - self._last_input) / dt
            self._last_input = measurement
        else:
            d_out = self.kd * (error - self._last_error) / dt
            self._last_error = error

        output = p_out + i_out + d_out
        output = max(self.output_limits[0], min(self.output_limits[1], output))

        self.last_error  = error
        self.last_p      = p_out
        self.last_i      = i_out
        self.last_d      = d_out
        self.last_output = output

        return output

    def reset(self):
        """Reset the internal state without changing the gains."""
        self._integral   = 0.0
        self._last_input = 0.0
        self._last_error = 0.0
        self._last_time  = time.monotonic()
        self._primed     = False
        self.last_error  = 0.0
        self.last_p      = 0.0
        self.last_i      = 0.0
        self.last_d      = 0.0
        self.last_output = 0.0

    def update_gains(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
