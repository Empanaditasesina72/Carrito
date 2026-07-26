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
    )
except ImportError:
    CAMERA_AWB_MODE   = 4
    CAMERA_CONTRAST   = 1.5
    CAMERA_SATURATION = 1.8
    CAMERA_SHARPNESS  = 4.0
    CAMERA_DENOISE    = 2
    CAMERA_EXPOSURE_US = None
    CAMERA_GAIN        = None


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
            print(f"[CAM] AE/AWB locked ({src}) - exp={exp} us  gain={gain:.2f}")
        except Exception as e:
            print(f"[CAM] Could not lock AE/AWB: {e}")


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
