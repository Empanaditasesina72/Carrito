#!/usr/bin/env python3
"""Retrain the traffic-sign detector for robustness / track generalization.

Fine-tunes weights/tmr_signs.pt (the validated 7-class model) with a
generalization-focused augmentation recipe, so the detector survives the real
track's distance, lighting and motion blur.

Point --data at TMR2026/datasets/merged_signs.yaml (built by
tools/overnight_signs.py from make_car_domain + synth_signs) rather than at
traffic_lights/data.yaml directly: the raw Roboflow set is 320x240, so training
on it at imgsz 640 is 2x interpolation and does not match what the camera hands
the detector. See the traffic_lights entry in CLAUDE.md for the measurements.

CRITICAL for THIS dataset: the classes include directional arrows
(left / right / straight). Horizontal and vertical flips are therefore DISABLED
(fliplr=flipud=0): a mirrored "left" arrow looks like "right" but keeps the
"left" label, which would poison training. Do not turn flips back on.

Biggest real-world win: add actual track images of the signs (captured with
tools/capture_track.py, labeled, merged into the dataset). Heavy augmentation
on the close-up set alone only goes so far.

GPU: CUDA is set up on this PC (torch 2.12.0+cu126, GTX 1650). Training uses the
GPU automatically (--device defaults to 0 when CUDA is present); this is where
the big speedup is (imgsz 640 conv is compute-bound). If you ever need to
reinstall it (fresh machine / Python 3.14):
    pip install torch==2.12.0+cu126 torchvision==0.27.0+cu126 \
        --index-url https://download.pytorch.org/whl/cu126
(cu128 has no torch 2.12 build; cu126 is the right index for Python 3.14.)

Usage:
    python TMR2026/tools/train_signs.py --epochs 120 --imgsz 640
    python TMR2026/tools/train_signs.py --model weights/tmr_signs.pt --device 0
    python TMR2026/tools/train_signs.py --data traffic_lights/data.yaml --batch 16

After training, copy the best weights and regenerate BOTH deploy exports:
    cp runs/train_signs/.../weights/best.pt TMR2026/weights/tmr_signs.pt
    python TMR2026/tools/export_model.py        # NCNN
    python TMR2026/tools/export_imx500.py        # rpk (on the Pi)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TMR_ROOT = HERE.parent
REPO_ROOT = TMR_ROOT.parent

DEFAULT_DATA = REPO_ROOT / "traffic_lights" / "data.yaml"
DEFAULT_MODEL = TMR_ROOT / "weights" / "tmr_signs.pt"

GENERALIZATION_AUG = dict(
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=8.0, translate=0.12, scale=0.6, shear=2.0, perspective=0.0005,
    flipud=0.0, fliplr=0.0,
    mosaic=1.0, close_mosaic=10, mixup=0.10, copy_paste=0.0,
    erasing=0.4, auto_augment="randaugment",
)

WASHOUT_ROBUST_AUG = dict(
    GENERALIZATION_AUG,
    hsv_s=0.9,
    hsv_v=0.6,
    scale=0.7,
)

# Pairs with the offline degradation in tools/make_car_domain.py. That script
# already supplies the photometric extremes (gain-22 noise, gamma darkening,
# motion blur, resolution loss) that Ultralytics cannot synthesize, so the
# online half concentrates on GEOMETRY: `scale` up to 0.9 plus full mosaic keeps
# shrinking the sign toward the ~28 px it occupies at 1.5 m, which is the size
# the detector keeps failing at.
CAR_DOMAIN_AUG = dict(
    GENERALIZATION_AUG,
    hsv_s=0.9,
    hsv_v=0.6,
    scale=0.9,
    translate=0.15,
    degrees=10.0,
    perspective=0.0008,
    close_mosaic=15,
    mixup=0.05,
    erasing=0.30,
)

RECIPES = {
    "generalization": GENERALIZATION_AUG,
    "washout": WASHOUT_ROBUST_AUG,
    "cardomain": CAR_DOMAIN_AUG,
}

# Names Ultralytics resolves and downloads on its own. Without this list a
# --model that is not a local file silently degrades to yolov8n.pt, so asking
# for yolov8s would quietly train an n and the comparison would be a lie.
HUB_MODELS = {
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="base weights to fine-tune (falls back to yolov8n.pt)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="", help="'0' for GPU, 'cpu', '' = auto")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--recipe", choices=tuple(RECIPES), default="generalization",
                    help="augmentation recipe; 'washout' widens saturation and "
                         "brightness jitter for pale prints and mixed light")
    ap.add_argument("--name", default="train_signs")
    ap.add_argument("--project", default=str(REPO_ROOT / "runs"))
    ap.add_argument("--time", type=float, default=None,
                    help="hard wall-clock cap in HOURS; overrides --epochs when "
                         "it runs out and still writes best.pt. Used by "
                         "tools/overnight_signs.py to keep a queue on schedule.")
    ap.add_argument("--cache", default=None, choices=("ram", "disk"),
                    help="cache images to speed up epochs (ram needs the "
                         "dataset to fit in memory)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not Path(args.data).exists():
        print(f"ERROR: data yaml not found: {args.data}")
        sys.exit(1)

    if Path(args.model).exists() or args.model in HUB_MODELS:
        base = args.model
    else:
        base = "yolov8n.pt"
        print(f"[TRAIN] {args.model} not found -> training from {base}")

    from ultralytics import YOLO
    import torch

    device = args.device
    if device == "":
        device = "0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[TRAIN] WARNING: training on CPU (no CUDA torch). This is slow; "
              "see the GPU note at the top of this file.")

    print(f"[TRAIN] base={base}  data={args.data}  imgsz={args.imgsz}  "
          f"epochs={args.epochs}  device={device}")
    print(f"[TRAIN] flips DISABLED (directional arrow classes)")
    print(f"[TRAIN] recipe={args.recipe}  "
          f"hsv_h={RECIPES[args.recipe]['hsv_h']} "
          f"hsv_s={RECIPES[args.recipe]['hsv_s']} "
          f"hsv_v={RECIPES[args.recipe]['hsv_v']}")
    if args.recipe == "washout":
        print("[TRAIN] hue jitter left narrow ON PURPOSE: red/green/yellow are "
              "separated by hue alone, widening it would destroy them.")

    extra: dict = {}
    if args.time:
        extra["time"] = args.time
        print(f"[TRAIN] wall-clock cap: {args.time:.2f} h "
              f"(best.pt is still written when it expires)")
    if args.cache:
        extra["cache"] = args.cache

    model = YOLO(base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        workers=args.workers,
        **RECIPES[args.recipe],
        **extra,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n[TRAIN] done. best weights: {best}")
    print("[TRAIN] To deploy:")
    print(f"   copy {best} -> {DEFAULT_MODEL}")
    print("   python TMR2026/tools/export_model.py     # regenerate NCNN")
    print("   python TMR2026/tools/export_imx500.py    # regenerate rpk (on the Pi)")


if __name__ == "__main__":
    main()
