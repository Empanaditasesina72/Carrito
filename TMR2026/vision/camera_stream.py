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
    )
except ImportError:
    CAMERA_AWB_MODE   = 4
    CAMERA_CONTRAST   = 1.5
    CAMERA_SATURATION = 1.8
    CAMERA_SHARPNESS  = 4.0
    CAMERA_DENOISE    = 2


class CameraStream:
    """
    Captures frames from the Pi AI Camera in a separate thread.

    Usage::

        cam = CameraStream(width=640, height=480, fps=30)
        cam.start()
        frame = cam.get_frame()   # BGR, ready for OpenCV
        cam.stop()
    """

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


    def _lock_ae_awb(self) -> None:
        """Read the current exposure and white balance and lock them."""
        try:
            meta   = self._picam2.capture_metadata()
            exp    = meta.get("ExposureTime")
            gain   = meta.get("AnalogueGain")
            cgains = meta.get("ColourGains")

            ctrl: dict = {"AeEnable": False}
            if exp    is not None: ctrl["ExposureTime"] = exp
            if gain   is not None: ctrl["AnalogueGain"] = gain
            if cgains is not None:
                ctrl["AwbEnable"]   = False
                ctrl["ColourGains"] = tuple(cgains)

            self._picam2.set_controls(ctrl)
            print(f"[CAM] AE/AWB locked - exp={exp} us  gain={gain:.2f}")
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
