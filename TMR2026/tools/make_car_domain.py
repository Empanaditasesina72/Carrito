#!/usr/bin/env python3
"""Bring the traffic-sign dataset onto the car camera's resolution and light.

CLAUDE.md described traffic_lights/ as "1470 close-up sign images, no track
photos". That is wrong, and measuring it changed this script. The set is
actually scale-model TRACK photography, very close to this vehicle's own setup,
and its geometry is:

    images       320 x 240   (not 640 x 480)
    box heights  14 - 116 px, median 45   (not 200-600 px close-ups)

Scaled to the camera's 640 x 480 that is 28 - 232 px, median 90. The car's own
operating range, from the pinhole model (focal 490 px, 8.5 cm octagon), is

    1.50 m -> 28 px      1.00 m -> 42 px      0.27 m -> 154 px

so the dataset covers the NEAR half of that range densely and thins out exactly
where the car has to see first: past ~1 m. Two consequences drive this file.

1. RESOLUTION. Training at imgsz 640 on 320x240 source is 2x interpolation --
   the model never sees real 640 detail. Inference at imgsz 320 then halves a
   640x480 camera frame, so the 28 px sign at 1.5 m arrives as 14 px, under the
   dataset's own p10. That is the mechanism behind the measured 61% @320 versus
   78% @640. Frames are therefore emitted at the camera's 640x480, so training
   and inference finally share one geometry.

2. CONDITIONS. The dataset is well lit and clean; the car runs at analogue gain
   22 (heavy noise) and 33 ms exposure (motion blur while cruising). Those are
   added here because Ultralytics cannot synthesize them.

Detail loss is kept MILD on purpose. The source is already 320x240, so half the
resolution is gone before this script starts; the aggressive downscaling that
would suit a true close-up set would just destroy these images.

Every degradation is LABEL-PRESERVING: photometric changes, blur, and
whole-frame resize (normalized YOLO coordinates are unchanged by a resize). No
crops, no flips -- flips would poison the directional arrow classes.

Outputs two datasets under TMR2026/datasets/ (both gitignored, regenerable):

  car_domain/   train = clean + degraded copies, valid = clean + degraded
                -> what to TRAIN on
  car_hard/     val   = deterministically degraded copies of the TEST split
                -> what to BENCHMARK on. The test split is never trained on, and
                   the seed is fixed, so the number is comparable across models.

Usage:
    python TMR2026/tools/make_car_domain.py
    python TMR2026/tools/make_car_domain.py --copies 2 --seed 7
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
TMR_ROOT = HERE.parent
REPO_ROOT = TMR_ROOT.parent

SRC = REPO_ROOT / "traffic_lights"
OUT_TRAIN = TMR_ROOT / "datasets" / "car_domain"
OUT_HARD = TMR_ROOT / "datasets" / "car_hard"

CLASS_NAMES = ["green", "left", "red", "right", "stop", "straight", "yellow"]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# --------------------------------------------------------------------------
# Degradations. Each takes/returns a BGR uint8 frame and never moves a box.
# --------------------------------------------------------------------------

def detail_loss(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Mild extra softness on top of the source's own 320x240 ceiling.

    The factor stays in 0.45-0.95 deliberately. Emitting at 640x480 from a
    320x240 original already caps real detail at 0.5, so anything harsher than
    this stacks on top of that and leaves an unreadable sign rather than a
    distant one.
    """
    h, w = img.shape[:2]
    f = rng.uniform(0.45, 0.95)
    small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def motion_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Directional blur: the car is moving during the 33 ms exposure."""
    k = rng.choice((3, 5, 7, 9))
    ang = rng.uniform(0.0, 180.0)
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1.0)
    kern = cv2.warpAffine(kern, m, (k, k))
    s = kern.sum()
    if s <= 1e-6:
        return img
    return cv2.filter2D(img, -1, kern / s)


def low_light(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Gamma darkening + desaturation: the track is lit far below the dataset."""
    gamma = rng.uniform(1.0, 2.6)
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)
    out = cv2.LUT(img, lut)

    sat = rng.uniform(0.45, 1.0)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= sat
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def sensor_noise(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Read + shot noise at analogue gain 22, with the chroma noise that
    dominates a small sensor pushed that hard."""
    sigma = rng.uniform(3.0, 14.0)
    f = img.astype(np.float32)
    f += np.random.normal(0.0, sigma, f.shape).astype(np.float32)
    # chroma noise: correlated across a channel, coarser than luma noise
    if rng.random() < 0.7:
        h, w = img.shape[:2]
        coarse = np.random.normal(0.0, sigma * 0.6, (h // 4 + 1, w // 4 + 1, 3))
        coarse = cv2.resize(coarse.astype(np.float32), (w, h),
                            interpolation=cv2.INTER_LINEAR)
        f += coarse
    return np.clip(f, 0, 255).astype(np.uint8)


def veiling_glare(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Flat contrast loss from stray light on the lens."""
    a = rng.uniform(0.70, 0.95)
    b = rng.uniform(4.0, 26.0)
    return np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


def degrade(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Full car-camera degradation chain, applied in physical order."""
    out = img
    out = detail_loss(out, rng)
    if rng.random() < 0.60:
        out = motion_blur(out, rng)
    if rng.random() < 0.85:
        out = low_light(out, rng)
    if rng.random() < 0.30:
        out = veiling_glare(out, rng)
    out = sensor_noise(out, rng)
    return out


# --------------------------------------------------------------------------

def list_pairs(split_dir: Path):
    """Yield (image_path, label_path) for a Roboflow-layout split."""
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    if not img_dir.is_dir():
        return []
    pairs = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            pairs.append((p, lbl_dir / f"{p.stem}.txt"))
    return pairs


def emit(pairs, out_split: Path, copies: int, seed: int, clean: bool,
         size: tuple[int, int]) -> int:
    """Write `copies` degraded variants (plus optionally the clean original),
    all resampled to the camera's own frame size so training and inference
    share one geometry. Labels are normalized, so a resize never moves a box."""
    (out_split / "images").mkdir(parents=True, exist_ok=True)
    (out_split / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for idx, (img_p, lbl_p) in enumerate(pairs):
        img = cv2.imread(str(img_p))
        if img is None:
            print(f"  ! unreadable, skipped: {img_p.name}")
            continue
        if (img.shape[1], img.shape[0]) != size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

        if clean:
            cv2.imwrite(str(out_split / "images" / f"{img_p.stem}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            if lbl_p.exists():
                shutil.copy2(lbl_p, out_split / "labels" / f"{img_p.stem}.txt")
            n += 1

        for c in range(copies):
            # Seeded per (image, copy) so the whole set is reproducible.
            rng = random.Random(f"{seed}:{img_p.stem}:{c}")
            np.random.seed(abs(hash((seed, img_p.stem, c))) % (2 ** 32))
            deg = degrade(img, rng)

            name = f"{img_p.stem}__car{c}"
            cv2.imwrite(str(out_split / "images" / f"{name}.jpg"), deg,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            if lbl_p.exists():
                shutil.copy2(lbl_p, out_split / "labels" / f"{name}.txt")
            n += 1

        if (idx + 1) % 200 == 0:
            print(f"  {idx + 1}/{len(pairs)} source images...")
    return n


def write_yaml(path: Path, root: Path, train: str, val: str) -> None:
    path.write_text(
        "# Generated by tools/make_car_domain.py -- safe to delete/regenerate.\n"
        f"path: {root.as_posix()}\n"
        f"train: {train}\n"
        f"val: {val}\n"
        "\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=1,
                    help="degraded variants per training image (default 1)")
    ap.add_argument("--hard-copies", type=int, default=2,
                    help="degraded variants per benchmark image (default 2)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--clean", action="store_true", default=True,
                    help="keep the clean originals in the training set")
    ap.add_argument("--size", default="640x480",
                    help="emit at the camera's frame size (config CAMERA_WIDTH"
                         "/HEIGHT), so train and inference geometry agree")
    args = ap.parse_args()

    try:
        sw, sh = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        print(f"ERROR: --size must look like 640x480, got {args.size!r}")
        sys.exit(1)
    size = (sw, sh)

    if not SRC.is_dir():
        print(f"ERROR: source dataset not found: {SRC}")
        sys.exit(1)

    for out in (OUT_TRAIN, OUT_HARD):
        if out.exists():
            shutil.rmtree(out)

    print(f"[CAR] source: {SRC}")
    print(f"[CAR] seed={args.seed}  copies={args.copies}  "
          f"hard_copies={args.hard_copies}  size={sw}x{sh}")

    print("\n[CAR] train split (clean + degraded)...")
    n_tr = emit(list_pairs(SRC / "train"), OUT_TRAIN / "train",
                args.copies, args.seed, args.clean, size)

    print("[CAR] valid split (clean + degraded)...")
    n_va = emit(list_pairs(SRC / "valid"), OUT_TRAIN / "valid",
                args.copies, args.seed, args.clean, size)

    print("[CAR] hard benchmark from the TEST split (degraded only)...")
    n_hd = emit(list_pairs(SRC / "test"), OUT_HARD / "val",
                args.hard_copies, args.seed + 1, False, size)

    write_yaml(OUT_TRAIN / "data.yaml", OUT_TRAIN, "train/images", "valid/images")
    write_yaml(OUT_HARD / "data.yaml", OUT_HARD, "val/images", "val/images")

    print(f"\n[CAR] car_domain : {n_tr} train / {n_va} val  -> {OUT_TRAIN}")
    print(f"[CAR] car_hard   : {n_hd} benchmark images     -> {OUT_HARD}")
    print("[CAR] The benchmark is built from the TEST split, which no training "
          "run has ever seen, and is seeded -- comparable across models.")


if __name__ == "__main__":
    main()
