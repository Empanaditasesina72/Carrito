"""Pi AI Camera (Picamera2) capture thread for the TMR 2026 vehicle.

Features:
  - RGB888 format -> already BGR-ordered, NO conversion (measured; see
    _capture_loop).
  - AE/AWB lock after a warm-up period (removes flicker).
  - Daemon thread: the main loop never waits for the previous frame.
  - Configurable resolution and FPS.
"""

import threading
import time
from typing import Optional

import cv2
import numpy as np

try:
    from config import (
        CAMERA_AWB_MODE,
        CAMERA_CONTRAST,
        CAMERA_SATURATION,
        CAMERA_SHARPNESS,
        CAMERA_DENOISE,
        CAMERA_EXPOSURE_US,
        CAMERA_GAIN,
        CAMERA_ADAPT_ENABLED,
        CAMERA_ADAPT_INTERVAL_S,
        CAMERA_ADAPT_V_LO,
        CAMERA_ADAPT_V_HI,
        CAMERA_ADAPT_CLIP_MAX,
        CAMERA_ADAPT_EXP_MAX_US,
        CAMERA_ADAPT_EXP_MIN_US,
        CAMERA_ADAPT_GAIN_MAX,
        CAMERA_ADAPT_GAIN_MIN,
    )
except ImportError:
    CAMERA_AWB_MODE   = 4
    CAMERA_CONTRAST   = 1.5
    CAMERA_SATURATION = 1.8
    CAMERA_SHARPNESS  = 4.0
    CAMERA_DENOISE    = 2
    CAMERA_EXPOSURE_US = None
    CAMERA_GAIN        = None
    CAMERA_ADAPT_ENABLED    = True
    CAMERA_ADAPT_INTERVAL_S = 1.5
    CAMERA_ADAPT_V_LO       = 45.0
    CAMERA_ADAPT_V_HI       = 95.0
    CAMERA_ADAPT_CLIP_MAX   = 3.0
    CAMERA_ADAPT_EXP_MAX_US = 33000
    CAMERA_ADAPT_EXP_MIN_US = 200
    CAMERA_ADAPT_GAIN_MAX   = 16.0
    CAMERA_ADAPT_GAIN_MIN   = 1.0


class CameraStream:
    """
    Captures frames from the Pi AI Camera in a separate thread.

    Usage::

        cam = CameraStream(width=640, height=480, fps=30)
        cam.start()
        frame = cam.get_frame()   # BGR, ready for OpenCV
        cam.stop()
    """

    # Seconds to wait after locking exposure/gain before serving frames, so the
    # sensor has adopted the manual values. See start().
    CONTROL_SETTLE_S = 0.6

    def __init__(
        self,
        width:        int   = 640,
        height:       int   = 480,
        fps:          int   = 30,
        awb_warmup_s: float = 2.0,
    ):
        self._w          = width
        self._h          = height
        self._fps        = fps
        self._warmup_s   = awb_warmup_s

        self._frame: Optional[np.ndarray] = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._ready = threading.Event()

        # Current sensor state, tracked so _adapt_exposure() can scale it rather
        # than re-reading metadata (which lags a write by ~0.5 s).
        self._exp   = 10_000.0
        self._gain  = 2.0
        self._last_adapt         = 0.0
        self._adapt_settle_until = 0.0

        from picamera2 import Picamera2
        self._picam2 = Picamera2()

        cfg = self._picam2.create_preview_configuration(
            main={
                "format": "RGB888",
                "size":   (width, height),
            },
            controls={
                "FrameDurationLimits": (1_000_000 // fps, 1_000_000 // fps),
                "AeEnable":            True,
                "AwbEnable":           True,
                "AwbMode":             CAMERA_AWB_MODE,
                "Contrast":            CAMERA_CONTRAST,
                "Saturation":          CAMERA_SATURATION,
                "Sharpness":           CAMERA_SHARPNESS,
                "NoiseReductionMode":  CAMERA_DENOISE,
            },
        )
        self._picam2.configure(cfg)


    def start(self) -> None:
        """Start the camera, wait for AE/AWB to settle and launch the thread."""
        self._picam2.start()
        print(f"[CAM] Settling AE/AWB ({self._warmup_s:.1f} s)...")
        time.sleep(self._warmup_s)
        self._lock_ae_awb()
        # The sensor needs a few frames to actually adopt manual exposure/gain.
        # Measured 2026-07-26: immediately after the lock the metadata still read
        # gain 6.52 with 14.2% of the frame clipped; 0.5 s later it read the
        # configured 4.00 at 2.7%. Without this wait the first ~15 frames are
        # served at the wrong exposure, which is enough to make a short
        # diagnostic run draw the wrong conclusion.
        time.sleep(self.CONTROL_SETTLE_S)
        self._stop.clear()
        threading.Thread(
            target=self._capture_loop,
            name="CameraStream",
            daemon=True,
        ).start()
        self._ready.wait(timeout=5.0)
        print("[CAM] Ready.")

    def stop(self) -> None:
        self._stop.set()
        time.sleep(0.1)
        self._picam2.stop()


    def get_frame(self) -> Optional[np.ndarray]:
        """
        Return the most recent BGR frame. Never blocks.
        Returns None if the camera has not captured any frame yet.
        """
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def recalibrate(self) -> None:
        """Re-calibrate AE/AWB (use on lighting changes)."""
        self._picam2.set_controls({"AeEnable": True, "AwbEnable": True})
        time.sleep(self._warmup_s)
        self._lock_ae_awb()
        time.sleep(self.CONTROL_SETTLE_S)


    def _lock_ae_awb(self) -> None:
        """Lock exposure and white balance, honouring the config overrides.

        CAMERA_EXPOSURE_US / CAMERA_GAIN win over whatever auto-exposure picked;
        either can be None to keep the AE result for that one control.

        This used to ignore both and always lock the AE result, which made
        config.py lie about the production path: on 2026-07-26 config asked for
        gain 4.0 and the sensor was measured running at 7.47, the value AE chose.
        AE meters the whole frame, and the frame is mostly dark track, so it
        pushes exposure up until the one bright object that matters -- the red
        STOP octagon -- clips. At gain 7.47 that cost 13.9% of the frame to
        clipping and `stop` dropped off the detector's output entirely at imgsz
        320; at the configured 4.0 the same sign reads 0.784. Use
        tools/tune_exposure.py to pick the values.
        """
        try:
            meta   = self._picam2.capture_metadata()
            exp    = meta.get("ExposureTime")
            gain   = meta.get("AnalogueGain")
            cgains = meta.get("ColourGains")

            src = "AE"
            if CAMERA_EXPOSURE_US is not None:
                exp = int(CAMERA_EXPOSURE_US)
                src = "config"
            if CAMERA_GAIN is not None:
                gain = float(CAMERA_GAIN)
                src = "config"

            ctrl: dict = {"AeEnable": False}
            if exp    is not None: ctrl["ExposureTime"] = int(exp)
            if gain   is not None: ctrl["AnalogueGain"] = float(gain)
            if cgains is not None:
                ctrl["AwbEnable"]   = False
                ctrl["ColourGains"] = tuple(cgains)

            self._picam2.set_controls(ctrl)
            self._exp  = float(exp)  if exp  is not None else 10_000.0
            self._gain = float(gain) if gain is not None else 2.0
            print(f"[CAM] AE/AWB locked ({src}) - exp={exp} us  gain={gain:.2f}")
        except Exception as e:
            print(f"[CAM] Could not lock AE/AWB: {e}")


    def _adapt_exposure(self, frame: np.ndarray) -> None:
        """Re-aim exposure at what the vision stack needs, not at mid-grey.

        The sensor's own AE cannot do this job. It averages the whole frame, the
        frame is mostly dark plastic track, so it opens up until the one bright
        object that matters clips: measured in daylight, the red STOP octagon hit
        S 23.7 with 100 % of its pixels at 255 and `stop` confidence read 0.000. But
        a pinned value cannot survive the day either -- the same 33 ms / gain 4.0
        that gave 100 % lane confidence at 14:00 left a completely empty mask by
        15:40, after a 3.4x drop in scene brightness.

        So the loop is closed on both objectives at once: raise the light while the
        frame is too dark for the white lines to clear their threshold, and back off
        whenever clipping starts to threaten the sign's red. Clipping wins over the
        brightness band, because a clipped sign cannot be recovered downstream while
        a slightly dark lane still can (the lane threshold is adaptive too).

        Total light is exposure x gain, so the correction is applied to that product
        and then split with exposure first -- gain only buys brightness at the cost
        of noise, and the detector was trained on frames whose noise was measured at
        sigma ~9.
        """
        if not CAMERA_ADAPT_ENABLED:
            return
        if CAMERA_EXPOSURE_US is not None and CAMERA_GAIN is not None:
            return                      # both pinned by config: respect that

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        v    = float(gray.mean())
        clip = 100.0 * float((gray >= 250).mean())

        if clip > CAMERA_ADAPT_CLIP_MAX:
            factor = max(0.55, 1.0 - (clip - CAMERA_ADAPT_CLIP_MAX) / 25.0)
            why    = f"clip {clip:.1f}%"
        elif v < CAMERA_ADAPT_V_LO:
            target = 0.5 * (CAMERA_ADAPT_V_LO + CAMERA_ADAPT_V_HI)
            factor = min(2.0, target / max(v, 1.0))
            why    = f"dark V {v:.1f}"
        elif v > CAMERA_ADAPT_V_HI:
            target = 0.5 * (CAMERA_ADAPT_V_LO + CAMERA_ADAPT_V_HI)
            factor = max(0.5, target / v)
            why    = f"bright V {v:.1f}"
        else:
            return                      # inside the band, leave the sensor alone

        prod = self._exp * self._gain * factor
        exp  = min(CAMERA_ADAPT_EXP_MAX_US,
                   max(CAMERA_ADAPT_EXP_MIN_US, prod / CAMERA_ADAPT_GAIN_MIN))
        gain = min(CAMERA_ADAPT_GAIN_MAX,
                   max(CAMERA_ADAPT_GAIN_MIN, prod / exp))

        # Skip writes that would not move the sensor, so a scene sitting just
        # outside the band does not generate I2C traffic every cycle.
        if (abs(exp - self._exp) < 0.02 * self._exp
                and abs(gain - self._gain) < 0.05):
            return

        try:
            self._picam2.set_controls({"AeEnable": False,
                                       "ExposureTime": int(exp),
                                       "AnalogueGain": float(gain)})
            print(f"[CAM] adapt ({why}): exp {self._exp:.0f}->{exp:.0f} us  "
                  f"gain {self._gain:.2f}->{gain:.2f}", flush=True)
            self._exp, self._gain = float(exp), float(gain)
            self._adapt_settle_until = time.monotonic() + self.CONTROL_SETTLE_S
        except Exception as e:
            print(f"[CAM] adapt failed: {e}")


    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            # Picamera2's "RGB888" already hands back BGR-ordered bytes. Measured
            # on this camera against the printed red STOP sign: the array as-is
            # reads H=9.4 (red), and after a COLOR_RGB2BGR it reads H=110.8
            # (cyan). Converting swaps red and blue. See vision/imx500_detector.py
            # for the full note.
            bgr = self._picam2.capture_array()

            with self._lock:
                self._frame = bgr

            self._ready.set()

            # Re-aim the exposure periodically. Skipped while the sensor is still
            # adopting the last change: measured, it needs ~0.5 s, and metering a
            # frame from before the change would make the loop chase its own tail.
            now = time.monotonic()
            if (now - self._last_adapt >= CAMERA_ADAPT_INTERVAL_S
                    and now >= self._adapt_settle_until):
                self._last_adapt = now
                self._adapt_exposure(bgr)
