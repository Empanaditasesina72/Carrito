"""Is the vehicle actually moving? Answered from the camera, with no encoder.

This exists because a stalled run looks like a PERFECT run to every other metric.
Measured on the car 2026-07-27, a 4 s closed-loop test reported lane-error
standard deviation of 0.1 px, drift of 0.0 px/s and a servo frozen at 93.1 deg --
numbers that read as flawless tracking and in fact meant the motor never broke
static friction. The scene did not change, so the error did not change. Without
this check that run was averaged in as a success.

The vehicle has no encoder and no IMU, so motion is inferred from the frame
itself: a moving camera changes its image, a stationary one changes only by
sensor noise. The threshold is not a constant -- noise depends on analogue gain,
which the adaptive exposure moves between 1 and 16 -- so the baseline is measured
from the vehicle's own stationary frames at the start of each run and the
decision is made relative to that.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import cv2
import numpy as np


class MotionCheck:
    """Rolling frame-difference motion detector.

    Usage::

        mc = MotionCheck()
        # before the motor starts, while the car is definitely still:
        while calibrating:
            mc.calibrate(cam.get_frame())
        mc.finish_calibration()
        # then, every tick:
        moving = mc.update(frame)
    """

    #: frames are reduced to this before differencing -- enough to see the track
    #: slide past, cheap enough to run every tick, and the downscale averages
    #: away most of the per-pixel noise that the gain would otherwise dominate.
    SMALL = (80, 60)

    #: motion is declared when the rolling difference exceeds
    #: baseline * FACTOR + FLOOR. FLOOR covers the case where the baseline is
    #: measured in an unusually quiet moment and would otherwise be near zero.
    FACTOR = 2.5
    FLOOR = 1.0

    #: seconds of history the decision is averaged over. Long enough that one
    #: noisy frame cannot flip it, short enough to catch a stall in well under a
    #: second of travel.
    WINDOW_S = 0.4

    #: how far back the comparison frame is taken from. THIS is what separates
    #: driving from shaking, and getting it wrong produced a false "SE MOVIO" on
    #: a car that never left its parking spot: consecutive frames differ just as
    #: much when the chassis vibrates under motor current as when the vehicle
    #: translates, so a frame-to-frame difference reports motion for a motor that
    #: is only buzzing. Against a frame from a second ago the two are nothing
    #: alike -- 0.3 m/s moves the scene 30 cm, while vibration returns to exactly
    #: where it started.
    LOOKBACK_S = 1.0

    def __init__(self, factor: float = FACTOR, floor: float = FLOOR,
                 window_s: float = WINDOW_S):
        self._factor = factor
        self._floor = floor
        self._window_s = window_s

        self._prev: Optional[np.ndarray] = None
        self._ref: deque[tuple[float, np.ndarray]] = deque()
        self._hist: deque[tuple[float, float]] = deque()
        self._cal: list[float] = []
        self._baseline: Optional[float] = None
        self.last_diff = 0.0

    # -- calibration -------------------------------------------------------

    def calibrate(self, frame: Optional[np.ndarray]) -> None:
        """Feed a frame taken while the vehicle is KNOWN to be stationary."""
        d = self._diff(frame)
        if d is not None:
            self._cal.append(d)

    def finish_calibration(self) -> float:
        """Freeze the stationary baseline. Returns it."""
        # Median, not mean: a single frame captured while someone's hand was
        # still in shot should not set the floor for the whole run.
        self._baseline = float(np.median(self._cal)) if self._cal else 0.0
        return self._baseline

    @property
    def threshold(self) -> float:
        base = self._baseline if self._baseline is not None else 0.0
        return base * self._factor + self._floor

    # -- runtime -----------------------------------------------------------

    def update(self, frame: Optional[np.ndarray]) -> bool:
        """Feed a frame; return True while the vehicle appears to be moving."""
        d = self._diff(frame)
        now = time.monotonic()
        if d is not None:
            self.last_diff = d
            self._hist.append((now, d))
        while self._hist and now - self._hist[0][0] > self._window_s:
            self._hist.popleft()

        if not self._hist:
            return True          # no evidence either way: do not cry stall
        mean = float(np.mean([v for _, v in self._hist]))
        return mean > self.threshold

    def reset(self) -> None:
        """Forget the rolling history, keeping the calibrated baseline."""
        self._hist.clear()
        self._ref.clear()
        self._prev = None

    # -- internals ---------------------------------------------------------

    def _diff(self, frame: Optional[np.ndarray]) -> Optional[float]:
        """Difference against a frame LOOKBACK_S old, not the previous one."""
        if frame is None:
            return None
        small = cv2.resize(frame, self.SMALL, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        now = time.monotonic()
        self._ref.append((now, gray))
        # keep just enough history to always have one frame at least
        # LOOKBACK_S old available
        while len(self._ref) > 2 and now - self._ref[1][0] >= self.LOOKBACK_S:
            self._ref.popleft()

        ref_t, ref = self._ref[0]
        if now - ref_t < self.LOOKBACK_S * 0.5:
            return None            # not enough history yet to judge
        return float(np.mean(np.abs(gray - ref)))
