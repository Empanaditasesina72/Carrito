#!/usr/bin/env python3
"""Composite real sign crops onto real indoor plates at the car's true scale.

The Roboflow set has no image where a sign is 30 px tall, and that is exactly
the size the car must handle. Heavy augmentation of a close-up cannot invent the
missing information: a 400 px octagon shrunk by Ultralytics' `scale` still
carries the sharpness and framing of a close-up. What is missing is a sign that
is genuinely small *inside a cluttered frame*.

So this builds that case directly:

  signs   - crops taken from the Roboflow ground-truth boxes (real appearance,
            real print, real lighting)
  plates  - THE DATASET'S OWN TRACK PHOTOS, upscaled to the camera's 640x480.
            Their existing labels are carried through and merged with the new
            ones, so every sign in the output frame is labeled.
  scale   - from the pinhole model in config.py. With focal 490 px and an 8.5 cm
            octagon, height_px = 0.085 * 490 / d. Pastes are drawn from the band
            in which the car actually decides to brake: 0.40-1.30 m = 32-104 px.
            Beyond 1.3 m the sign is not yet reliably detected; closer than 43 cm
            it has left the lens entirely, so neither end is worth weight.

An earlier version of this script used the legacy 640x480 webcam recordings in
_legacy/runs/detect/ as plates, on the theory that they were shot in the house
where the track is. Inspecting the output killed that idea: the footage is
selfie-framed, so signs ended up pasted across a face and a shirt, nothing like
a floor-level track view, and one frame showing a phone playing a street video
slipped past the reject filter and contributed an unlabeled traffic light. Real
track plates avoid all three problems at once.

Labels are exact by construction: the paste region IS the box. The alpha mask is
an octagon for `stop` and a rounded rectangle otherwise, so the crop's own
background corners are not pasted in as a rectangular seam the model could
learn to key on. Brightness is matched to the local plate region for the same
reason, and pastes never overlap a box that is already there.

Output goes to TMR2026/datasets/synth_signs/ (gitignored, regenerable). It is
TRAINING data only -- the benchmark stays car_hard, built from the untouched
test split, so the evaluation never sees synthetic pixels.

Usage:
    python TMR2026/tools/synth_signs.py --count 2500
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
sys.path.insert(0, str(HERE))

from make_car_domain import degrade, write_yaml, CLASS_NAMES  # noqa: E402

SRC = REPO_ROOT / "traffic_lights"
OUT = TMR_ROOT / "datasets" / "synth_signs"
FRAME = (640, 480)

FOCAL_PX = 490.0
SIGN_H_M = 0.085
# The range in which the car actually DECIDES TO BRAKE, measured on the track
# 2026-07-27, not the range that happens to be thin in the dataset:
#
#   1.30 m ->  32 px   first reliable detection
#   1.04 m ->  40 px   measured, sign read 56-61 %
#   0.43 m ->  97 px   the sign leaves the lens (28 cm off axis, ~+-33 deg FOV)
#
# Closer than 43 cm the sign is simply not in frame, so nothing past that matters.
# This was 1.0-2.5 m (16-41 px), i.e. FAR signs -- almost no overlap with the
# 32-104 px band the braking decision is actually made in. Training weight was
# going to a case the car never uses.
DIST_MIN_M, DIST_MAX_M = 0.40, 1.30

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
STOP_ID = CLASS_NAMES.index("stop")


# --------------------------------------------------------------------------

def harvest_crops(min_px: int = 34):
    """Cut every ground-truth box out of the train split, keyed by class."""
    crops: dict[int, list[np.ndarray]] = {i: [] for i in range(len(CLASS_NAMES))}
    img_dir = SRC / "train" / "images"
    lbl_dir = SRC / "train" / "labels"
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        lbl = lbl_dir / f"{p.stem}.txt"
        if not lbl.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        for line in lbl.read_text().splitlines():
            q = line.split()
            if len(q) < 5:
                continue
            c = int(q[0])
            cx, cy, bw, bh = map(float, q[1:5])
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            # Only crops with real detail survive downscaling to 30 px.
            if x2 - x1 >= min_px and y2 - y1 >= min_px:
                crops[c].append(img[y1:y2, x1:x2].copy())
    return crops


def harvest_plates():
    """Load the real track photos as plates, keeping their existing labels.

    Returns (image, [(cls, cx, cy, w, h) normalized]) pairs at camera size.
    Carrying the labels through is what makes this safe: the frame ends up with
    both its original sign and the pasted ones, all labeled.
    """
    plates = []
    img_dir = SRC / "train" / "images"
    lbl_dir = SRC / "train" / "labels"
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        if (img.shape[1], img.shape[0]) != FRAME:
            img = cv2.resize(img, FRAME, interpolation=cv2.INTER_CUBIC)
        boxes = []
        lbl = lbl_dir / f"{p.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                q = line.split()
                if len(q) >= 5:
                    boxes.append((int(q[0]), *map(float, q[1:5])))
        plates.append((img, boxes))
    print(f"[SYN] plates: {len(plates)} real track photos at "
          f"{FRAME[0]}x{FRAME[1]}, existing labels carried through")
    return plates


def alpha_mask(h: int, w: int, octagon: bool) -> np.ndarray:
    """Soft cut-out so the crop's own rectangular background is not pasted."""
    m = np.zeros((h, w), np.float32)
    if octagon:
        k = 0.293  # regular octagon inscribed in the box
        pts = np.array([
            (w * k, 0), (w * (1 - k), 0), (w - 1, h * k), (w - 1, h * (1 - k)),
            (w * (1 - k), h - 1), (w * k, h - 1), (0, h * (1 - k)), (0, h * k),
        ], np.int32)
        cv2.fillConvexPoly(m, pts, 1.0)
    else:
        r = max(1, int(min(h, w) * 0.12))
        cv2.rectangle(m, (r, 0), (w - r, h), 1.0, -1)
        cv2.rectangle(m, (0, r), (w, h - r), 1.0, -1)
        for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
            cv2.circle(m, (cx, cy), r, 1.0, -1)
    blur = max(1, int(min(h, w) * 0.08)) | 1
    return cv2.GaussianBlur(m, (blur, blur), 0)


def match_brightness(crop: np.ndarray, region: np.ndarray,
                     rng: random.Random) -> np.ndarray:
    """Pull the crop's exposure toward the plate's, so it is not a bright patch
    on a dark wall -- otherwise 'brighter than surroundings' becomes the cue."""
    cm = float(crop.mean()) + 1e-3
    rm = float(region.mean())
    target = rm * rng.uniform(0.85, 1.45)
    g = np.clip(target / cm, 0.35, 2.0)
    return np.clip(crop.astype(np.float32) * g, 0, 255).astype(np.uint8)


def paste(plate: np.ndarray, crop: np.ndarray, cls: int,
          rng: random.Random, taken: list) -> tuple | None:
    """Place one sign at a physically plausible size; return its YOLO box."""
    H, W = plate.shape[:2]

    dist = rng.uniform(DIST_MIN_M, DIST_MAX_M)
    sh = int(round(SIGN_H_M * FOCAL_PX / dist))
    if sh < 14 or sh > int(H * 0.75):
        return None
    ar = crop.shape[1] / crop.shape[0]
    sw = max(8, int(round(sh * ar * rng.uniform(0.92, 1.08))))
    if sw > int(W * 0.6):
        return None

    s = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)

    ang = rng.uniform(-9.0, 9.0)
    m = cv2.getRotationMatrix2D((sw / 2, sh / 2), ang, 1.0)
    s = cv2.warpAffine(s, m, (sw, sh), borderMode=cv2.BORDER_REPLICATE)
    a = cv2.warpAffine(alpha_mask(sh, sw, cls == STOP_ID), m, (sw, sh))

    # These are the FAR pastes (1-2.5 m), so they belong near the horizon: the
    # upper-middle band, never the foreground floor. Anything overlapping a box
    # already in the plate is rejected rather than blended over.
    for _ in range(14):
        x = rng.randint(int(W * 0.05), max(int(W * 0.05) + 1, W - sw - int(W * 0.05)))
        y = rng.randint(int(H * 0.18), max(int(H * 0.18) + 1, int(H * 0.58)))
        if y + sh >= H:
            continue
        if all(x + sw < bx or bx + bw < x or y + sh < by or by + bh < y
               for bx, by, bw, bh in taken):
            break
    else:
        return None

    region = plate[y:y + sh, x:x + sw]
    if region.shape[:2] != (sh, sw):
        return None
    s = match_brightness(s, region, rng)

    a3 = np.dstack([a] * 3)
    plate[y:y + sh, x:x + sw] = (s * a3 + region * (1 - a3)).astype(np.uint8)

    # A thin post under the octagon: the printed sign is 17.5 cm tall overall
    # against an 8.5 cm face, so roughly one sign-height of post is visible.
    # It is deliberately OUTSIDE the label box -- the box is the face only,
    # matching both the Roboflow convention and STOP_SIGN_REAL_HEIGHT_M.
    if cls == STOP_ID and rng.random() < 0.75:
        pw = max(1, int(sw * rng.uniform(0.10, 0.18)))
        px = x + sw // 2 - pw // 2
        py, ph = y + sh, int(sh * rng.uniform(0.6, 1.1))
        ph = min(ph, H - py)
        if ph > 2:
            shade = int(np.clip(region.mean() * rng.uniform(0.35, 0.7), 15, 190))
            plate[py:py + ph, px:px + pw] = shade

    taken.append((x, y, sw, sh))
    return (cls, (x + sw / 2) / W, (y + sh / 2) / H, sw / W, sh / H)


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2500)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-crop-px", type=int, default=26,
                    help="smallest ground-truth box worth cutting out; the "
                         "dataset's p10 is 26 px, so going higher starves the "
                         "rarer classes of crops")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    print("[SYN] harvesting sign crops from ground-truth boxes...")
    crops = harvest_crops(args.min_crop_px)
    for i, n in sorted(crops.items()):
        print(f"       {CLASS_NAMES[i]:9} {len(crops[i]):5d} crops")
    usable = [i for i in crops if crops[i]]
    if not usable:
        print("ERROR: no usable crops")
        sys.exit(1)

    print("[SYN] loading real track plates...")
    plates = harvest_plates()
    if not plates:
        print(f"ERROR: no plates under {SRC / 'train' / 'images'}")
        sys.exit(1)

    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "valid"):
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

    # `stop` is what actually brakes the car, so it gets a third of the budget;
    # the rest is spread evenly so the other six classes do not regress.
    made = 0
    n_val = 0
    n_added = 0
    W, H = FRAME
    for k in range(args.count):
        src_img, src_boxes = plates[rng.randrange(len(plates))]
        plate = src_img.copy()
        cls = STOP_ID if rng.random() < 0.34 else rng.choice(usable)

        # Seed the occupancy list with the plate's real signs so nothing is
        # pasted on top of them, and keep their labels in the output.
        taken = [(int((cx - bw / 2) * W), int((cy - bh / 2) * H),
                  int(bw * W), int(bh * H)) for _, cx, cy, bw, bh in src_boxes]
        boxes = list(src_boxes)

        added = 0
        for _ in range(1 if rng.random() < 0.65 else 2):
            c = cls if added == 0 else rng.choice(usable)
            crop = crops[c][rng.randrange(len(crops[c]))]
            b = paste(plate, crop, c, rng, taken)
            if b:
                boxes.append(b)
                added += 1
        if added == 0 or not boxes:
            continue
        n_added += added

        img = degrade(plate, rng) if rng.random() < 0.85 else plate

        split = "valid" if rng.random() < args.val_frac else "train"
        name = f"syn{k:06d}"
        cv2.imwrite(str(OUT / split / "images" / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        (OUT / split / "labels" / f"{name}.txt").write_text(
            "\n".join(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                      for c, cx, cy, bw, bh in boxes) + "\n",
            encoding="utf-8")
        made += 1
        n_val += (split == "valid")
        if made % 400 == 0:
            print(f"       {made}/{args.count} composites...")

    write_yaml(OUT / "data.yaml", OUT, "train/images", "valid/images")
    print(f"\n[SYN] {made} composites ({made - n_val} train / {n_val} val) -> {OUT}")
    print(f"[SYN] {n_added} signs pasted, drawn from {DIST_MIN_M}-{DIST_MAX_M} m "
          f"= {int(SIGN_H_M * FOCAL_PX / DIST_MAX_M)}-"
          f"{int(SIGN_H_M * FOCAL_PX / DIST_MIN_M)} px tall -- the far range "
          f"the dataset is thin on.")


if __name__ == "__main__":
    main()
