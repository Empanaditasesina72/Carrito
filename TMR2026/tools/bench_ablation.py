"""CPU-vs-NPU ablation: what does moving inference into the sensor buy?

All three reviewers asked for a baseline or ablation rather than a single
configuration. This measures the same detector two ways on the same hardware:

  cpu-inference  the YOLO model executing on the Pi 5 ARM cores (NCNN export,
                 falling back to the .pt), timed directly on a fixed frame. No
                 camera needed, so this phase runs under any conditions.
  cpu            the full control loop with the CPU detector thread running.
  npu            the full control loop with inference inside the IMX500.

Inference time for a fixed-size YOLO graph is essentially independent of image
content, so the synthetic frame in cpu-inference is representative; only the
resolution matters, and it is the deployed imgsz.

Run one backend per invocation so the camera is never opened twice:
    python tools/bench_ablation.py --backend cpu-inference
    python tools/bench_ablation.py --backend npu --cycles 600
    python tools/bench_ablation.py --backend cpu --cycles 600

Never writes to the motor.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    SERVO_CENTER_ANGLE, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE,
    USE_IMX500_NPU, IMX500_RPK_PATH, IMX500_LABELS_PATH, IMX500_CONF,
    PID_KP, PID_KI, PID_KD,
)
from vision.lane_pipeline import LanePipeline
from control.pid_controller import PIDController

YOLO_CONF, YOLO_IMGSZ = 0.55, 320


def _temp_c():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], timeout=2).decode()
        return float(out.strip().split("=")[1].split("'")[0])
    except Exception:
        return None


def _cpu_snapshot():
    """Cumulative (busy, total) jiffies across all cores."""
    try:
        with open("/proc/stat", "r") as f:
            parts = [float(x) for x in f.readline().split()[1:]]
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0.0)
        return sum(parts) - idle, sum(parts)
    except Exception:
        return None


def _cpu_percent(before, after):
    if not before or not after:
        return None
    db, dt = after[0] - before[0], after[1] - before[1]
    return 100.0 * db / dt if dt > 0 else None


def _synthetic_frame() -> np.ndarray:
    img = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    cv2.line(img, (210, CAMERA_HEIGHT), (255, CAMERA_HEIGHT // 2), (255, 255, 255), 8)
    cv2.line(img, (430, CAMERA_HEIGHT), (395, CAMERA_HEIGHT // 2), (255, 255, 255), 8)
    return img


def bench_cpu_inference(n: int) -> int:
    from vision.sign_detector import SignDetector
    sd = SignDetector(conf=YOLO_CONF, imgsz=YOLO_IMGSZ)
    model = getattr(sd, "_model", None)
    if model is None:
        print("[ABL] Could not load the CPU model.")
        return 1
    path = getattr(sd, "_model_path", "?")
    print(f"[ABL] CPU model: {path}  imgsz={YOLO_IMGSZ}  conf={YOLO_CONF}")

    frame = _synthetic_frame()
    for _ in range(3):
        model.predict(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)

    t0, c0 = _temp_c(), _cpu_snapshot()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        model.predict(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
        times.append((time.perf_counter() - t) * 1000.0)
    c1, t1 = _cpu_snapshot(), _temp_c()

    a = np.array(times)
    print()
    print("=" * 62)
    print("  CPU INFERENCE (per detection, on the Pi 5 ARM cores)")
    print("=" * 62)
    print(f"  inferences   : {a.size}")
    print(f"  mean/median  : {a.mean():.2f} / {np.median(a):.2f} ms")
    print(f"  p95 / max    : {np.percentile(a,95):.2f} / {a.max():.2f} ms")
    print(f"  max rate     : {1000.0/a.mean():.1f} detections/s")
    print(f"  vs 20 ms budget: {a.mean()/20.0:.1f}x the entire control cycle")
    print(f"  CPU busy     : {_cpu_percent(c0,c1):.1f} % of all cores")
    print(f"  temp         : {t0} -> {t1} C")
    print("=" * 62)
    print()
    print("  For comparison, the IMX500 path costs 0.019 ms of CPU per cycle")
    print("  (tensor parsing only); inference runs on the sensor in parallel.")
    return 0


def bench_loop(backend: str, cycles: int, hz: float) -> int:
    lane = LanePipeline(frame_w=CAMERA_WIDTH, frame_h=CAMERA_HEIGHT, debug=False)
    pid = PIDController(
        kp=PID_KP, ki=PID_KI, kd=PID_KD, setpoint=0.0,
        output_limits=(-(SERVO_CENTER_ANGLE - SERVO_MIN_ANGLE),
                       (SERVO_MAX_ANGLE - SERVO_CENTER_ANGLE)),
        integral_limits=(-25.0, 25.0),
    )

    if backend == "npu":
        if not (USE_IMX500_NPU and os.path.isfile(IMX500_RPK_PATH)):
            print("[ABL] The .rpk is missing; cannot run the NPU backend.")
            return 1
        from vision.imx500_detector import IMX500CameraStream
        cam = IMX500CameraStream(
            rpk_path=IMX500_RPK_PATH, labels_path=IMX500_LABELS_PATH,
            width=CAMERA_WIDTH, height=CAMERA_HEIGHT,
            fps=CAMERA_FPS, conf=IMX500_CONF)
        sign = cam
        label = "IMX500 NPU (on-sensor)"
    else:
        from vision.camera_stream import CameraStream
        from vision.sign_detector import SignDetector
        cam = CameraStream(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS)
        sign = SignDetector(conf=YOLO_CONF, imgsz=YOLO_IMGSZ)
        label = "CPU detector thread"

    cam.start()
    if sign is not cam:
        sign.start()

    t0 = time.monotonic()
    while cam.get_frame() is None and time.monotonic() - t0 < 10:
        time.sleep(0.05)
    if cam.get_frame() is None:
        print("[ABL] No camera frames.")
        return 1

    period = 1.0 / hz
    lat, sign_ms, det_count = [], [], 0
    temp0, cpu0 = _temp_c(), _cpu_snapshot()
    t_last = time.monotonic()

    for i in range(cycles):
        c0 = time.perf_counter()
        frame = cam.get_frame()
        if frame is None:
            time.sleep(period); continue
        c1 = time.perf_counter()
        r = lane.process(frame)
        c2 = time.perf_counter()
        if sign is not cam:
            sign.update_frame(frame)
        _ = sign.has_sign("stop_sign") or sign.has_sign("red")
        c3 = time.perf_counter()
        corr = pid.compute(r.error_px, period)
        _ = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, SERVO_CENTER_ANGLE + corr))
        c4 = time.perf_counter()

        lat.append((c4 - c0) * 1000.0)
        sign_ms.append((c3 - c2) * 1000.0)
        if sign.get_detections():
            det_count += 1

        dt = time.monotonic() - t_last
        if dt < period:
            time.sleep(period - dt)
        t_last = time.monotonic()

    cpu1, temp1 = _cpu_snapshot(), _temp_c()
    cam.stop()
    if sign is not cam:
        sign.stop()

    a = np.array(lat[1:])
    s = np.array(sign_ms[1:])
    deadline = 1000.0 / hz
    print()
    print("=" * 62)
    print(f"  CONTROL LOOP -- {label}")
    print("=" * 62)
    print(f"  cycles       : {a.size}")
    print(f"  latency mean : {a.mean():.2f} ms   median {np.median(a):.2f}")
    print(f"  jitter (std) : {a.std(ddof=1):.2f} ms")
    print(f"  p95 / p99    : {np.percentile(a,95):.2f} / {np.percentile(a,99):.2f} ms")
    print(f"  max          : {a.max():.2f} ms")
    print(f"  under {deadline:.0f} ms  : {100*np.mean(a<=deadline):.2f} %  "
          f"({int((a>deadline).sum())} misses)")
    print(f"  t_sign mean  : {s.mean():.3f} ms  (in-loop cost of sign gating)")
    print(f"  CPU busy     : {_cpu_percent(cpu0,cpu1):.1f} % of all cores")
    print(f"  temp         : {temp0} -> {temp1} C")
    print("=" * 62)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("cpu-inference", "cpu", "npu"),
                    default="cpu-inference")
    ap.add_argument("--cycles", type=int, default=600)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--inferences", type=int, default=60)
    args = ap.parse_args()

    if args.backend == "cpu-inference":
        return bench_cpu_inference(args.inferences)
    return bench_loop(args.backend, args.cycles, args.hz)


if __name__ == "__main__":
    sys.exit(main())
