"""Auto-label track captures for domain-adaptation fine-tuning.

The detector was trained on close-up sign photos with no track imagery, so its
usable range on the vehicle is bounded by apparent size and it is brittle to the
printed sign's colour. Fine-tuning on frames from the actual camera, at the
actual distances, against the actual background is the real fix. Labelling those
frames by hand is the slow part; this does the first pass.

How it works, and why it is not circular: the model proposes boxes at a low
confidence, but the CLASS is forced with --class, because the operator knows what
was in front of the camera. That is what makes the pass useful rather than
self-confirming -- the misclassifications we actually observed on track (a STOP
read as `yellow`, `green` or `straight`) get corrected into `stop` instead of
being baked in.

Every frame gets a review JPG with the proposed box drawn on it. Look through
them and delete the bad ones BEFORE merging; a wrong box teaches a wrong lesson.

Usage (from TMR2026/):
    # 1. capture on the Pi, moving the sign through the distances you care about
    python tools/capture_track.py --auto 1.0

    # 2. propose labels (on the PC, where the GPU is)
    python tools/autolabel_signs.py --src tools/captures --class stop

    # 3. review out/review/*.jpg, delete any frame that is wrong

    # 4. merge into the dataset, oversampled so a few hundred track frames
    #    can hold their own against 1029 close-ups
    python tools/autolabel_signs.py --src tools/captures --class stop --merge --repeat 3
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

CLASS_NAMES = ["green", "left", "red", "right", "stop", "straight", "yellow"]
DATASET = REPO / "traffic_lights"
PREFIX = "track_"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(ROOT / "tools" / "captures"),
                    help="folder of captured frames")
    ap.add_argument("--out", default=str(ROOT / "datasets" / "track_signs"))
    ap.add_argument("--class", dest="cls", default="stop", choices=CLASS_NAMES,
                    help="what is actually in these frames")
    ap.add_argument("--weights", default=str(ROOT / "weights" / "tmr_signs.pt"))
    ap.add_argument("--conf", type=float, default=0.10,
                    help="proposal threshold; low on purpose, you review after")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--all-boxes", action="store_true",
                    help="keep every proposal, not just the most confident one")
    ap.add_argument("--merge", action="store_true",
                    help="copy the reviewed set into traffic_lights/train")
    ap.add_argument("--repeat", type=int, default=1,
                    help="how many copies to write when merging (oversampling)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.is_dir():
        print(f"ERROR: {src} does not exist. Capture some frames first.")
        return 1

    frames = sorted([p for p in src.iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not frames:
        print(f"ERROR: no images in {src}")
        return 1

    cls_idx = CLASS_NAMES.index(args.cls)
    img_dir, lbl_dir, rev_dir = out / "images", out / "labels", out / "review"

    if args.merge:
        return do_merge(img_dir, lbl_dir, args.repeat)

    for d in (img_dir, lbl_dir, rev_dir):
        d.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(args.weights)
    print(f"[LABEL] {len(frames)} frames  class={args.cls}  conf>={args.conf}")

    kept = skipped = 0
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        r = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        boxes = list(r.boxes)
        if not boxes:
            skipped += 1
            continue
        if not args.all_boxes:
            boxes = [max(boxes, key=lambda b: float(b.conf))]

        lines, vis = [], img.copy()
        for b in boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(vis, f"{args.cls} (was {r.names[int(b.cls)]} "
                             f"{float(b.conf):.0%})",
                        (int(x1), max(14, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        stem = PREFIX + f.stem
        cv2.imwrite(str(img_dir / (stem + ".jpg")), img)
        (lbl_dir / (stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        cv2.imwrite(str(rev_dir / (stem + ".jpg")), vis)
        kept += 1

    print(f"[LABEL] labelled {kept}, no proposal in {skipped}")
    print(f"[LABEL] REVIEW {rev_dir}\\ and delete any frame whose box is wrong,")
    print(f"        then delete the matching files in images\\ and labels\\.")
    print(f"[LABEL] when the set is clean:")
    print(f"        python tools/autolabel_signs.py --merge --repeat 3")
    return 0


def do_merge(img_dir: Path, lbl_dir: Path, repeat: int) -> int:
    dst_i, dst_l = DATASET / "train" / "images", DATASET / "train" / "labels"
    if not dst_i.is_dir():
        print(f"ERROR: {dst_i} not found.")
        return 1
    imgs = sorted(img_dir.glob("*.jpg"))
    if not imgs:
        print(f"ERROR: nothing in {img_dir}. Run the labelling pass first.")
        return 1

    n = 0
    for p in imgs:
        lbl = lbl_dir / (p.stem + ".txt")
        if not lbl.exists():
            continue
        for k in range(max(1, repeat)):
            stem = p.stem if k == 0 else f"{p.stem}_r{k}"
            shutil.copy2(p, dst_i / (stem + ".jpg"))
            shutil.copy2(lbl, dst_l / (stem + ".txt"))
            n += 1

    for cache in (DATASET / "train" / "labels.cache",
                  DATASET / "valid" / "labels.cache"):
        if cache.exists():
            cache.unlink()

    print(f"[MERGE] wrote {n} files into {dst_i}")
    print(f"[MERGE] label caches cleared so the next run rescans")
    print(f"[MERGE] every added file starts with '{PREFIX}', so to undo:")
    print(f"        del {dst_i}\\{PREFIX}*  and  {dst_l}\\{PREFIX}*")
    print(f"[MERGE] now retrain:")
    print(f"        python TMR2026/tools/train_signs.py --data traffic_lights/data_local.yaml "
          f"--recipe washout --epochs 60")
    return 0


if __name__ == "__main__":
    sys.exit(main())
