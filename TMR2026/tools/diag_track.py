"""Headless lane + sign diagnostic. No display, no motors, safe over SSH.

tools/tune_track.py needs an OpenCV window, which means a desktop session. This
tool answers the same question -- does the vehicle see the track the way the
simulator does -- from a plain SSH shell, and writes the intermediate images to
disk so they can be copied off the Pi and inspected.

It reports, per frame, the lane error in pixels, the confidence, and which of
the two lane lines the sliding windows actually locked onto. That last column is
the one that matters on a three-line road: if the histogram picks the dashed
centre line instead of an outer solid line, right_bias has to change.

Reads track_calib.json when present, so it reflects the current calibration.
Never writes to the motor.

Usage (from TMR2026/):
    python tools/diag_track.py
    python tools/diag_track.py --frames 30 --out /tmp/diag
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    USE_IMX500_NPU, IMX500_RPK_PATH, IMX500_LABELS_PATH, IMX500_CONF,
    LANE_RIGHT_BIAS,
)
from vision.lane_pipeline import LanePipeline


def _load_calib() -> dict:
    path = ROOT / "track_calib.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            calib = json.load(f)
        print(f"[DIAG] Using calibration {path}")
        return calib
    except FileNotFoundError:
        print("[DIAG] No track_calib.json yet; using config defaults.")
        return {}
    except Exception as e:
        print(f"[DIAG] Could not read track_calib.json ({e}); using defaults.")
        return {}


def _build_vision():
    if USE_IMX500_NPU and os.path.isfile(IMX500_RPK_PATH):
        try:
            from vision.imx500_detector import IMX500CameraStream
            npu = IMX500CameraStream(
                rpk_path=IMX500_RPK_PATH, labels_path=IMX500_LABELS_PATH,
                width=CAMERA_WIDTH, height=CAMERA_HEIGHT,
                fps=CAMERA_FPS, conf=IMX500_CONF,
            )
            print("[DIAG] Backend: IMX500 NPU (on-sensor inference)")
            return npu, npu
        except Exception as e:
            print(f"[DIAG] NPU unavailable ({e}); falling back to CPU.")

    from vision.camera_stream import CameraStream
    from vision.sign_detector import SignDetector
    cam = CameraStream(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS)
    sign = SignDetector()
    print("[DIAG] Backend: CPU (CameraStream + SignDetector)")
    return cam, sign


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--out", default="/tmp/diag")
    ap.add_argument("--exposure", type=int, default=None,
                    help="force ExposureTime in us (low light)")
    ap.add_argument("--gain", type=float, default=None,
                    help="force AnalogueGain (low light)")
    ap.add_argument("--fps", type=float, default=None,
                    help="force frame rate; lower it to allow a longer exposure")
    ap.add_argument("--interval", type=float, default=None,
                    help="seconds between samples; raise it to watch live while "
                         "moving a sign in front of the camera")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    calib = _load_calib()

    lane = LanePipeline(
        frame_w=CAMERA_WIDTH, frame_h=CAMERA_HEIGHT, debug=True,
        right_bias=calib.get("right_bias", LANE_RIGHT_BIAS),
        roi_frac=calib.get("roi_frac", 0.5),
        hsv_white_lo=calib.get("hsv_white_lo"),
        hsv_white_hi=calib.get("hsv_white_hi"),
    )
    print(f"[DIAG] right_bias={lane._right_bias:.2f}  "
          f"hsv_lo={list(lane.HSV_WHITE_LO)}  hsv_hi={list(lane.HSV_WHITE_HI)}")

    camera, sign_det = _build_vision()
    camera.start()
    if sign_det is not camera:
        sign_det.start()

    import time
    if args.exposure or args.gain or args.fps:
        picam2 = getattr(camera, "_picam2", None)
        if picam2 is None:
            print("[DIAG] Cannot reach the camera handle; overrides ignored.")
        else:
            ctrl: dict = {"AeEnable": False}
            if args.fps:
                dur = int(1e6 / args.fps)
                ctrl["FrameDurationLimits"] = (dur, dur)
            if args.exposure:
                ctrl["ExposureTime"] = args.exposure
            if args.gain:
                ctrl["AnalogueGain"] = args.gain
            picam2.set_controls(ctrl)
            time.sleep(1.5)
            m = picam2.capture_metadata()
            print(f"[DIAG] Overrides applied: exp={m.get('ExposureTime')} us  "
                  f"gain={m.get('AnalogueGain'):.1f}")

    t0 = time.monotonic()
    while camera.get_frame() is None and time.monotonic() - t0 < 10:
        time.sleep(0.05)
    frame = camera.get_frame()
    if frame is None:
        print("[DIAG] ERROR: no frames from the camera.")
        return 1

    interval = args.interval if args.interval else 1.0 / CAMERA_FPS
    print()
    print(f"{'#':>3} {'error_px':>9} {'head':>6} {'conf':>6} {'left_x':>7} "
          f"{'right_x':>8}  {'lines':<11} signs")
    print("-" * 78)

    results, last, best = [], None, 0
    for i in range(args.frames):
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        if sign_det is not camera:
            sign_det.update_frame(frame)
        r = lane.process(frame)
        results.append(r)
        last = (frame, r)
        which = ("both" if r.left_x is not None and r.right_x is not None
                 else "LEFT only" if r.left_x is not None
                 else "RIGHT only" if r.right_x is not None
                 else "NONE")
        lx = f"{r.left_x}" if r.left_x is not None else "-"
        rx = f"{r.right_x}" if r.right_x is not None else "-"
        live = sign_det.get_detections() if hasattr(sign_det, "get_detections") else []
        if live:
            best += 1
            sg = ", ".join(
                f"{d.label} {d.confidence:.0%}"
                f"@{d.distance_m * 100:.0f}cm" if d.distance_m else f"{d.label} {d.confidence:.0%}"
                for d in live[:3])
        else:
            sg = "-"
        print(f"{i:>3} {r.error_px:>+9.1f} {r.heading:>+6.1f} {r.confidence:>6.0%} "
              f"{lx:>7} {rx:>8}  {which:<11} {sg}")
        time.sleep(interval)

    if not results:
        print("[DIAG] ERROR: no frames processed.")
        return 1

    errs = np.array([r.error_px for r in results])
    confs = np.array([r.confidence for r in results])
    both = sum(1 for r in results
               if r.left_x is not None and r.right_x is not None)
    seps = [r.right_x - r.left_x for r in results
            if r.left_x is not None and r.right_x is not None]

    print()
    print("=" * 56)
    print(f"  frames            : {len(results)}")
    print(f"  error_px          : mean {errs.mean():+.1f}, "
          f"std {errs.std():.1f}, range {errs.min():+.0f}..{errs.max():+.0f}")
    print(f"  confidence        : mean {confs.mean():.0%}, "
          f"at 100% in {int((confs >= 1.0).sum())}/{len(results)} frames")
    print(f"  both lines found  : {both}/{len(results)} frames")
    print(f"  frames with signs : {best}/{len(results)}")
    if seps:
        print(f"  line separation   : mean {np.mean(seps):.0f} px "
              f"(pipeline expects 384 px, accepts 230-538)")
        if not (230 <= np.mean(seps) <= 538):
            print("  => OUT OF BAND: the pipeline will drop to single-line mode.")
    fill = None
    if last is not None and last[1].mask_frame is not None:
        mk = last[1].mask_frame
        mk = mk[:, :, 0] if mk.ndim == 3 else mk
        fill = float((mk > 0).mean())
        print(f"  white mask fill   : {fill:.1%} of the bird's-eye view")

    print()
    degenerate = fill is not None and not (0.005 <= fill <= 0.35)
    if degenerate:
        print("  VERDICT: DEGENERATE MASK -- the confidence above is not")
        print("  trustworthy. A usable lane mask covers roughly 1-15 % of the")
        if fill > 0.35:
            print("  view. Here almost everything passes the filter, so the")
            print("  sliding windows locked onto arbitrary bright edges (floor,")
            print("  furniture), not the lane lines. Reduce exposure/gain or")
            print("  tighten S_max; if the illuminant is coloured, no threshold")
            print("  will separate white lines from a bright floor.")
        else:
            print("  view. Here almost nothing passes: too dark, or V_min too high.")
    elif confs.mean() >= 0.9 and errs.std() < 40:
        print("  VERDICT: stable lock. Good enough to drive.")
    elif confs.mean() >= 0.5:
        print("  VERDICT: partial lock. Tune HSV V_min and re-run.")
    else:
        print("  VERDICT: no lock. Check lighting and the white mask image.")
    print("=" * 56)

    frame, r = last
    cv2.imwrite(os.path.join(args.out, "1_frame.png"), frame)
    if r.bev_frame is not None:
        cv2.imwrite(os.path.join(args.out, "2_bev.png"), r.bev_frame)
    if r.mask_frame is not None:
        cv2.imwrite(os.path.join(args.out, "3_mask_white.png"), r.mask_frame)
    cv2.imwrite(os.path.join(args.out, "4_annotated.png"),
                lane.draw_debug(frame, r))

    dets = sign_det.get_detections() if hasattr(sign_det, "get_detections") else []
    print()
    print(f"  signs detected    : {len(dets)}")
    for d in dets:
        dist = f"{d.distance_m * 100:.0f} cm" if d.distance_m else "?"
        print(f"    {d.label:<10} conf {d.confidence:.0%}  at {dist}")
    if not dets:
        print("    (none -- point the camera at the STOP sign to test it)")

    print()
    print(f"[DIAG] Images written to {args.out}/")

    camera.stop()
    if sign_det is not camera:
        sign_det.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
