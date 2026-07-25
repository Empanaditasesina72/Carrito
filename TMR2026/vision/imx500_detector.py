"""Camera + sign detection on the IMX500 NPU.

The Pi AI Camera (Sony IMX500) has a neural accelerator INSIDE the sensor:
the model runs in the camera and the Pi only receives, per frame, the image
plus the output tensors in the metadata. CPU used for inference: ~0%.

`IMX500CameraStream` merges the two roles that `CameraStream` and
`SignDetector` play in the CPU path, exposing BOTH interfaces:

  As a camera (for LanePipeline / overlay):
      get_frame() -> BGR (golden rule RGB888 -> cv2.COLOR_RGB2BGR)

  As a detector (for the FSM / telemetry):
      get_detections() / has_sign() / has_any_sign() / closest_sign()
      update_frame() is a no-op (the NPU already has the frame; it exists
      only so main.py does not change with the backend).

Same guarantees as the CPU path:
  - Normalized labels: "stop" -> "stop_sign"; only the 7 classes of the
    tmr_signs model (green/left/red/right/stop/straight/yellow).
  - Pinhole distance using the real height of each class.
  - Hysteresis of N consecutive frames before publishing a label.
  - COLOR fallback (red/purple) when the NPU misses the STOP.
  - AE/AWB locked after warm-up (no exposure flicker).

The .rpk is generated with `tools/export_imx500.py` (on the Pi or any Linux).
If the .rpk does not exist, main.py does not even import this module and uses
the CPU path (NCNN) -- see `VehicleTMR._build_vision()`.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from vision.sign_detector import (
    Detection,
    SIGN_REAL_HEIGHT_M,
    CAMERA_FOCAL_LENGTH_PX,
    _detect_red_blob,
)

try:
    from config import (
        CAMERA_AWB_MODE, CAMERA_CONTRAST, CAMERA_SATURATION,
        CAMERA_SHARPNESS, CAMERA_DENOISE, CAMERA_BUFFERS,
        CAMERA_EXPOSURE_US, CAMERA_GAIN,
        STOP_SIGN_REAL_HEIGHT_M,
    )
except ImportError:
    CAMERA_AWB_MODE, CAMERA_CONTRAST, CAMERA_SATURATION = 4, 1.5, 1.0
    CAMERA_SHARPNESS, CAMERA_DENOISE, CAMERA_BUFFERS = 4.0, 2, 6
    CAMERA_EXPOSURE_US, CAMERA_GAIN = None, None
    STOP_SIGN_REAL_HEIGHT_M = 0.085

DEFAULT_LABELS = ("green", "left", "red", "right", "stop", "straight", "yellow")

MIN_BBOX_AREA = 150


def map_raw_detections(
    raw: list[tuple[int, int, int, int, float, int]],
    labels: tuple[str, ...] | list[str],
    conf_min: float,
    min_area: int = MIN_BBOX_AREA,
) -> list[Detection]:
    """
    Convert raw NPU detections -- (x1, y1, x2, y2, score, cls_id) in frame
    pixels -- into `Detection` objects with the same semantics as the CPU
    path: "stop"->"stop_sign" normalization, class/area filter and per-class
    pinhole distance.
    """
    dets: list[Detection] = []
    for x1, y1, x2, y2, score, cls_id in raw:
        if score < conf_min:
            continue
        if not (0 <= cls_id < len(labels)):
            continue
        label = str(labels[cls_id]).strip().lower().replace(" ", "_")
        if label == "stop_sign":
            label = "stop"
        if label not in SIGN_REAL_HEIGHT_M:
            continue

        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))
        if (x2 - x1) * (y2 - y1) < min_area:
            continue

        height_px  = max(1, y2 - y1)
        distance_m = (SIGN_REAL_HEIGHT_M[label] * CAMERA_FOCAL_LENGTH_PX) / height_px
        normalized = "stop_sign" if label == "stop" else label
        dets.append(Detection(normalized, float(score), x1, y1, x2, y2,
                              distance_m=distance_m))
    return dets


class LabelHysteresis:
    """
    Temporal filter identical to SignDetector's: a label is published only
    after appearing in `n_frames` consecutive frames; the largest-area
    detection of the frame (the closest one) is kept.
    """

    def __init__(self, n_frames: int = 3):
        self._n = max(1, n_frames)
        self._consecutive: dict[str, int] = {}
        self._last_raw: dict[str, Detection] = {}

    def update(self, raw_dets: list[Detection]) -> list[Detection]:
        seen: dict[str, Detection] = {}
        for d in raw_dets:
            prev = seen.get(d.label)
            if prev is None or d.area > prev.area:
                seen[d.label] = d

        for label in list(self._consecutive.keys()):
            if label not in seen:
                self._consecutive[label] = 0

        for label, det in seen.items():
            self._consecutive[label] = self._consecutive.get(label, 0) + 1
            self._last_raw[label] = det

        return [self._last_raw[label]
                for label, count in self._consecutive.items()
                if count >= self._n and label in self._last_raw]


class IMX500CameraStream:
    """
    Captures frames and detections from the NPU in a single daemon thread.

    Usage (identical to CameraStream + SignDetector together)::

        cam = IMX500CameraStream("weights/tmr_signs_imx500.rpk")
        cam.start()
        frame = cam.get_frame()          # BGR
        dets  = cam.get_detections()     # [Detection, ...]
        cam.stop()
    """

    HYSTERESIS_FRAMES = 3

    def __init__(
        self,
        rpk_path: str,
        labels_path: Optional[str] = None,
        width:  int = 640,
        height: int = 480,
        fps:    int = 30,
        conf:   float = 0.55,
        awb_warmup_s: float = 2.0,
    ):
        self._w, self._h = width, height
        self._conf       = conf
        self._warmup_s   = awb_warmup_s

        self._frame: Optional[np.ndarray] = None
        self._frame_lock  = threading.Lock()
        self._results:    list[Detection] = []
        self._result_lock = threading.Lock()
        self._hysteresis  = LabelHysteresis(self.HYSTERESIS_FRAMES)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._stopped = False

        from picamera2 import Picamera2
        from picamera2.devices.imx500 import IMX500, NetworkIntrinsics

        print(f"[NPU] Loading model into the IMX500: {rpk_path}")
        self._imx500 = IMX500(rpk_path)
        self._intrinsics = self._imx500.network_intrinsics or NetworkIntrinsics()

        self._labels = self._resolve_labels(labels_path)
        print(f"[NPU] Classes: {list(self._labels)}")

        rate = getattr(self._intrinsics, "inference_rate", None)
        eff_fps = int(min(fps, rate)) if rate else fps

        self._picam2 = Picamera2(self._imx500.camera_num)
        cfg = self._picam2.create_preview_configuration(
            main={
                "format": "RGB888",
                "size":   (width, height),
            },
            controls={
                "FrameRate":           eff_fps,
                "AeEnable":            True,
                "AwbEnable":           True,
                "AwbMode":             CAMERA_AWB_MODE,
                "Contrast":            CAMERA_CONTRAST,
                "Saturation":          CAMERA_SATURATION,
                "Sharpness":           CAMERA_SHARPNESS,
                "NoiseReductionMode":  CAMERA_DENOISE,
            },
            buffer_count=CAMERA_BUFFERS,
        )
        self._picam2.configure(cfg)

    def _resolve_labels(self, labels_path: Optional[str]):
        """labels.txt (export) -> .rpk intrinsics -> dataset order."""
        if labels_path:
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    labels = [ln.strip() for ln in f if ln.strip()]
                if labels:
                    return tuple(labels)
            except OSError:
                pass
        intr_labels = getattr(self._intrinsics, "labels", None)
        if intr_labels:
            return tuple(intr_labels)
        return DEFAULT_LABELS


    def start(self) -> None:
        if self._started:
            return
        self._started = True

        self._picam2.start()
        print(f"[NPU] Settling AE/AWB ({self._warmup_s:.1f} s)...")
        time.sleep(self._warmup_s)
        self._lock_ae_awb()

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="IMX500Stream", daemon=True)
        self._thread.start()
        print("[NPU] Camera + NPU ready (on-chip inference, CPU free).")

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self._picam2.stop()
        except Exception:
            pass


    def get_frame(self) -> Optional[np.ndarray]:
        """Latest BGR frame. Never blocks; None if there is no capture yet."""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None


    def update_frame(self, frame: np.ndarray) -> None:
        """No-op: the NPU receives the frame inside the sensor itself."""

    def get_detections(self) -> list[Detection]:
        with self._result_lock:
            return list(self._results)

    def has_sign(self, label: str) -> bool:
        return any(d.label == label for d in self.get_detections())

    def has_any_sign(self) -> bool:
        return len(self.get_detections()) > 0

    def closest_sign(self, label: Optional[str] = None) -> Optional[Detection]:
        dets = self.get_detections()
        if label is not None:
            dets = [d for d in dets if d.label == label]
        dets = [d for d in dets if d.distance_m is not None]
        return min(dets, key=lambda d: d.distance_m) if dets else None


    def _lock_ae_awb(self) -> None:
        try:
            meta   = self._picam2.capture_metadata()
            exp    = meta.get("ExposureTime")
            gain   = meta.get("AnalogueGain")
            cgains = meta.get("ColourGains")

            if CAMERA_EXPOSURE_US is not None: exp  = int(CAMERA_EXPOSURE_US)
            if CAMERA_GAIN        is not None: gain = float(CAMERA_GAIN)

            ctrl: dict = {"AeEnable": False}
            if exp    is not None: ctrl["ExposureTime"] = exp
            if gain   is not None: ctrl["AnalogueGain"] = gain
            if cgains is not None:
                ctrl["AwbEnable"]   = False
                ctrl["ColourGains"] = tuple(cgains)

            self._picam2.set_controls(ctrl)
            forced = ""
            if CAMERA_EXPOSURE_US is not None or CAMERA_GAIN is not None:
                forced = "  (forced by config)"
            print(f"[NPU] AE/AWB locked - exp={exp} us  gain={gain:.2f}{forced}")
        except Exception as e:
            print(f"[NPU] Could not lock AE/AWB: {e}")


    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                request = self._picam2.capture_request()
                try:
                    rgb      = request.make_array("main")
                    metadata = request.get_metadata()
                finally:
                    request.release()

                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                with self._frame_lock:
                    self._frame = bgr

                raw = self._parse_npu_output(metadata)

                if not any(d.label == "stop_sign" for d in raw):
                    blob = _detect_red_blob(bgr)
                    if blob is not None:
                        x1, y1, x2, y2, _area = blob
                        h_px = max(1, y2 - y1)
                        raw.append(Detection(
                            "stop_sign", 0.55, x1, y1, x2, y2,
                            distance_m=(STOP_SIGN_REAL_HEIGHT_M
                                        * CAMERA_FOCAL_LENGTH_PX) / h_px,
                        ))

                confirmed = self._hysteresis.update(raw)
                with self._result_lock:
                    self._results = confirmed

            except Exception:
                time.sleep(0.01)

    def _parse_npu_output(self, metadata: dict) -> list[Detection]:
        """
        IMX500 tensors -> detections in `main` frame pixels.
        Follows the official picamera2 demo flow for detection models
        (including those exported by Ultralytics in `imx` format):
        boxes/scores/classes + intrinsics flags + coordinate conversion
        through the ISP (`convert_inference_coords`).
        """
        np_outputs = self._imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None or len(np_outputs) < 3:
            return []

        input_w, input_h = self._imx500.get_input_size()

        if getattr(self._intrinsics, "postprocess", "") == "nanodet":
            from picamera2.devices.imx500 import postprocess_nanodet_detection
            from picamera2.devices.imx500.postprocess import scale_boxes
            boxes, scores, classes = postprocess_nanodet_detection(
                outputs=np_outputs[0], conf=self._conf,
                iou_thres=0.65, max_out_dets=10)[0]
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
        else:
            boxes   = np_outputs[0][0]
            scores  = np_outputs[1][0]
            classes = np_outputs[2][0]
            if getattr(self._intrinsics, "bbox_normalization", False):
                boxes = boxes / input_h
            if getattr(self._intrinsics, "bbox_order", "yx") == "xy":
                boxes = boxes[:, [1, 0, 3, 2]]

        raw: list[tuple[int, int, int, int, float, int]] = []
        for box, score, cls in zip(boxes, scores, classes):
            if float(score) < self._conf:
                continue
            x, y, w, h = self._imx500.convert_inference_coords(
                np.asarray(box).flatten(), metadata, self._picam2)
            raw.append((int(x), int(y), int(x + w), int(y + h),
                        float(score), int(cls)))

        return map_raw_detections(raw, self._labels, self._conf)
