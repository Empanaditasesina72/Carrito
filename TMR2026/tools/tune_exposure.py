#!/usr/bin/env python3
"""Find the exposure/gain that lets the detector actually see the STOP sign.

Why this exists. On 2026-07-26, with the car in front of the track in daylight,
the lane pipeline reported a perfect lock (100% confidence, both lines, 12/12
frames) while the sign detector found nothing at all in 12 frames. Measuring the
captured frame explained it:

    STOP sign core   B 255  G 231  R 255   S 23.7   100% of pixels clipped
    whole frame                            13.2% of pixels clipped
    track surface    B  61  G  53  R  37   S 104     0% clipped

config.py had CAMERA_GAIN=22.0 and CAMERA_EXPOSURE_US=33000 pinned into it, which
were tuned the previous night under a phone flashlight. In daylight that is
several stops over, so the red octagon blew out to white-pink: saturation
collapsed from the >150 a real red carries down to 23.7, and no amount of model
retraining can recover a channel that is clipped at 255.

Note the failure is silent in exactly the same way the RGB/BGR bug was. The
track is dark plastic and the lane lines are R=G=B, so the lane pipeline is
indifferent to exposure and keeps reporting a healthy lock while the coloured
signs are being destroyed. A green lane verdict is NOT evidence that the camera
is configured correctly.

So: sweep the settings, run the real detector on each, and report numbers rather
than guessing. Judge by what the FSM actually consumes -- the confidence of
`stop` against its 0.55 gate -- with saturation and clipping alongside to show
why a setting wins or loses.

Point the camera at the STOP sign, from a realistic distance, then:

    python TMR2026/tools/tune_exposure.py
    python TMR2026/tools/tune_exposure.py --frames 5 --dist 1.0
    python TMR2026/tools/tune_exposure.py --apply     # write the winner to config.py

Touches no GPIO: camera and YOLO only, safe to run with the motors powered down.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
TMR_ROOT = HERE.parent
sys.path.insert(0, str(TMR_ROOT))

CONFIG_PY = TMR_ROOT / "config.py"
GATE = 0.55

# (label, exposure_us, analogue_gain). None/None = leave the sensor's own AE
# result in place, which is what config.py does when both overrides are unset.
CANDIDATES = [
    ("auto (AE)",        None,   None),
    ("2 ms  g1.0",      2_000,   1.0),
    ("4 ms  g1.0",      4_000,   1.0),
    ("8 ms  g1.0",      8_000,   1.0),
    ("8 ms  g2.0",      8_000,   2.0),
    ("16 ms g1.0",     16_000,   1.0),
    ("16 ms g2.0",     16_000,   2.0),
    ("33 ms g4.0",     33_000,   4.0),
    ("33 ms g22.0",    33_000,  22.0),   # the night setting, for reference
]


def measure(frame: np.ndarray, model, imgsz: int, probe: float):
    """Best `stop` detection in one frame, plus why it looks the way it does."""
    res = model.predict(frame, imgsz=imgsz, conf=probe, verbose=False)[0]
    names = res.names

    best_conf, best_sat, best_clip = 0.0, 0.0, 0.0
    for b in res.boxes:
        if names[int(b.cls.item())] != "stop":
            continue
        c = float(b.conf.item())
        if c <= best_conf:
            continue
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        patch = frame[max(0, y1):y2, max(0, x1):x2]
        if patch.size:
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            best_sat = float(hsv[..., 1].mean())
            best_clip = 100.0 * float((patch.max(axis=2) >= 250).mean())
        best_conf = c

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_clip = 100.0 * float((gray >= 250).mean())
    return best_conf, best_sat, best_clip, frame_clip, float(gray.mean())


def apply_to_config(exp, gain) -> None:
    """Rewrite the two override lines in config.py, preserving everything else."""
    txt = CONFIG_PY.read_text(encoding="utf-8")
    e = "None" if exp is None else str(int(exp))
    g = "None" if gain is None else f"{float(gain):.1f}"
    txt = re.sub(r"^CAMERA_EXPOSURE_US\s*=.*$",
                 f"CAMERA_EXPOSURE_US = {e}", txt, count=1, flags=re.M)
    txt = re.sub(r"^CAMERA_GAIN\s*=.*$",
                 f"CAMERA_GAIN        = {g}", txt, count=1, flags=re.M)
    CONFIG_PY.write_text(txt, encoding="utf-8")
    print(f"\n[TUNE] config.py updated: CAMERA_EXPOSURE_US={e}  CAMERA_GAIN={g}")
    print("[TUNE] None means 'keep the sensor's own AE result', which adapts to "
          "whatever light is present at startup -- the safer default than a "
          "value pinned for one condition.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=4,
                    help="frames averaged per candidate")
    ap.add_argument("--imgsz", type=int, default=320,
                    help="must match SignDetector's imgsz to be meaningful")
    ap.add_argument("--probe", type=float, default=0.10,
                    help="detector threshold; below the gate so near-misses show")
    ap.add_argument("--weights", default=str(TMR_ROOT / "weights" / "tmr_signs.pt"))
    ap.add_argument("--settle", type=float, default=1.2,
                    help="seconds to let the sensor settle after a change")
    ap.add_argument("--apply", action="store_true",
                    help="write the winning setting into config.py")
    ap.add_argument("--dist", type=float, default=None,
                    help="note the sign distance in metres, for the log only")
    args = ap.parse_args()

    from picamera2 import Picamera2
    from ultralytics import YOLO

    print(f"[TUNE] loading {args.weights} ...")
    model = YOLO(args.weights)

    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)},
        controls={"FrameDurationLimits": (33_333, 33_333)},
    )
    picam2.configure(cfg)
    picam2.start()
    print("[TUNE] camera up, letting AE settle ...")
    time.sleep(2.0)

    ae = picam2.capture_metadata()
    ae_exp, ae_gain = ae.get("ExposureTime"), ae.get("AnalogueGain")
    print(f"[TUNE] the sensor's own AE picked exp={ae_exp} us gain={ae_gain:.2f}")
    if args.dist:
        print(f"[TUNE] sign distance noted: {args.dist:.2f} m")

    print(f"\n{'setting':14} {'stop conf':>10} {'sat':>7} {'box clip':>9} "
          f"{'frame clip':>11} {'frame V':>8}")
    print("-" * 64)

    rows = []
    for label, exp, gain in CANDIDATES:
        if exp is None:
            picam2.set_controls({"AeEnable": True})
            time.sleep(args.settle)
            m = picam2.capture_metadata()
            picam2.set_controls({
                "AeEnable": False,
                "ExposureTime": m.get("ExposureTime", ae_exp),
                "AnalogueGain": m.get("AnalogueGain", ae_gain),
            })
            eff = (m.get("ExposureTime"), m.get("AnalogueGain"))
        else:
            picam2.set_controls({"AeEnable": False, "ExposureTime": int(exp),
                                 "AnalogueGain": float(gain)})
            eff = (exp, gain)
        time.sleep(args.settle)

        acc = []
        for _ in range(args.frames):
            # "RGB888" already hands back BGR-ordered bytes on this camera -- no
            # colour conversion here, on purpose. See CLAUDE.md.
            acc.append(measure(picam2.capture_array(), model,
                               args.imgsz, args.probe))
            time.sleep(0.08)

        conf = max(a[0] for a in acc)
        sat = max(a[1] for a in acc)
        bclip = np.mean([a[2] for a in acc])
        fclip = np.mean([a[3] for a in acc])
        fv = np.mean([a[4] for a in acc])

        mark = "  <-- passes gate" if conf >= GATE else ("  (under gate)" if conf else "  NOT SEEN")
        print(f"{label:14} {conf:10.3f} {sat:7.1f} {bclip:8.1f}% "
              f"{fclip:10.1f}% {fv:8.1f}{mark}")
        rows.append((conf, sat, label, eff))

    picam2.stop()

    rows.sort(key=lambda r: (-r[0], -r[1]))
    conf, sat, label, eff = rows[0]
    print("-" * 64)
    print(f"[TUNE] best: {label}  stop conf {conf:.3f}  sign saturation {sat:.1f}")
    if conf < GATE:
        print(f"[TUNE] WARNING: even the best setting is under the {GATE} gate. "
              f"Exposure is then not the whole story -- check that the sign is "
              f"in frame, upright and within ~1.5 m.")

    if args.apply:
        if conf <= 0:
            print("[TUNE] refusing --apply: no setting saw the sign at all, so "
                  "there is nothing to conclude.")
        else:
            apply_to_config(*eff)
    else:
        print(f"[TUNE] re-run with --apply to write this into config.py")


if __name__ == "__main__":
    main()
