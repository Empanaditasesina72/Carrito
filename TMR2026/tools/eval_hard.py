#!/usr/bin/env python3
"""Rank sign detectors on the car-domain benchmark, at the gate that matters.

mAP on the Roboflow close-ups is useless for choosing a model here: every
variant scores ~0.995 and the metric cannot discriminate. What decides whether
the car brakes is narrower:

    does `stop` (or `red`) clear conf >= 0.55 on a frame that looks like what
    the camera actually delivers?

So this reports, per class, on the car_hard benchmark:

  R@gate    recall at the production threshold -- the fraction of real signs
            the FSM would actually act on
  conf      mean confidence of the detections that DID clear the gate; this is
            the safety margin over 0.55
  near      ground-truth signs found but BELOW the gate. These are the ones the
            threshold throws away, and the number that shrinks when the model
            gets better at this domain
  FP        detections above the gate with no matching sign. On `stop`/`red`
            these are phantom brakes, the failure mode that made the car stop
            at green lights before.

Only `stop` and `red` gate the FSM (see main.py:_update_vision), so the summary
line weighs those two.

Usage:
    python TMR2026/tools/eval_hard.py --weights TMR2026/weights/tmr_signs.pt
    python TMR2026/tools/eval_hard.py --weights runs/x/weights/best.pt --imgsz 640
    python TMR2026/tools/eval_hard.py --compare runs/a/weights/best.pt runs/b/weights/best.pt
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TMR_ROOT = HERE.parent
REPO_ROOT = TMR_ROOT.parent

HARD = TMR_ROOT / "datasets" / "car_hard" / "val"
CLASS_NAMES = ["green", "left", "red", "right", "stop", "straight", "yellow"]
GATING = ("stop", "red")          # the only classes that brake the car
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = ua + ub - inter
    return inter / u if u > 0 else 0.0


def load_gt(lbl: Path, w: int, h: int):
    out = []
    if not lbl.exists():
        return out
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(p[0])
        cx, cy, bw, bh = map(float, p[1:5])
        out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def evaluate(weights: Path, imgsz: int, gate: float, iou_thr: float,
             probe: float, limit: int | None):
    from ultralytics import YOLO

    imgs = [p for p in sorted((HARD / "images").iterdir())
            if p.suffix.lower() in IMG_EXTS]
    if limit:
        imgs = imgs[:limit]
    if not imgs:
        print(f"ERROR: no benchmark images in {HARD / 'images'}")
        print("Run:  python TMR2026/tools/make_car_domain.py")
        sys.exit(1)

    model = YOLO(str(weights))

    hit   = defaultdict(int)   # GT matched by a detection above the gate
    near  = defaultdict(int)   # GT matched only below the gate
    miss  = defaultdict(int)   # GT not matched at all
    fp    = defaultdict(int)
    csum  = defaultdict(float)
    ccnt  = defaultdict(int)

    for img_p in imgs:
        res = model.predict(str(img_p), imgsz=imgsz, conf=probe,
                            verbose=False)[0]
        h, w = res.orig_shape
        gt = load_gt(HARD / "labels" / f"{img_p.stem}.txt", w, h)

        dets = []
        for b in res.boxes:
            dets.append((int(b.cls.item()), float(b.conf.item()),
                         [float(v) for v in b.xyxy[0].tolist()]))
        dets.sort(key=lambda d: -d[1])

        used = set()
        for gi, (gc, *gb) in enumerate(gt):
            best_i, best_c = -1, 0.0
            for di, (dc, dconf, db) in enumerate(dets):
                if di in used or dc != gc:
                    continue
                if iou(gb, db) >= iou_thr and dconf > best_c:
                    best_i, best_c = di, dconf
            name = CLASS_NAMES[gc]
            if best_i < 0:
                miss[name] += 1
            else:
                used.add(best_i)
                if best_c >= gate:
                    hit[name] += 1
                    csum[name] += best_c
                    ccnt[name] += 1
                else:
                    near[name] += 1

        for di, (dc, dconf, _) in enumerate(dets):
            if di not in used and dconf >= gate:
                fp[CLASS_NAMES[dc]] += 1

    return hit, near, miss, fp, csum, ccnt, len(imgs)


def report(tag: str, stats, gate: float) -> tuple[float, float, int]:
    hit, near, miss, fp, csum, ccnt, n_img = stats

    print(f"\n=== {tag}   ({n_img} benchmark frames, gate={gate}) ===")
    print(f"{'class':10} {'R@gate':>8} {'conf':>7} {'near':>6} "
          f"{'miss':>6} {'FP':>5}   {'GT':>5}")
    print("-" * 60)

    g_hit = g_tot = g_fp = 0
    for c in CLASS_NAMES:
        tot = hit[c] + near[c] + miss[c]
        if tot == 0:
            continue
        r = hit[c] / tot
        mc = (csum[c] / ccnt[c]) if ccnt[c] else 0.0
        mark = " *" if c in GATING else "  "
        print(f"{c:10}{mark}{r * 100:6.1f}% {mc:7.3f} {near[c]:6d} "
              f"{miss[c]:6d} {fp[c]:5d}   {tot:5d}")
        if c in GATING:
            g_hit += hit[c]
            g_tot += tot
            g_fp += fp[c]

    gr = (g_hit / g_tot) if g_tot else 0.0
    gc = sum(csum[c] for c in GATING)
    gn = sum(ccnt[c] for c in GATING)
    gconf = (gc / gn) if gn else 0.0
    print("-" * 60)
    print(f"{'GATING':10} *{gr * 100:6.1f}% {gconf:7.3f} "
          f"{sum(near[c] for c in GATING):6d} "
          f"{sum(miss[c] for c in GATING):6d} {g_fp:5d}   {g_tot:5d}")
    print("  (* = stop/red, the only classes that brake the car)")
    return gr, gconf, g_fp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(TMR_ROOT / "weights" / "tmr_signs.pt"))
    ap.add_argument("--compare", nargs="*", default=None,
                    help="rank several .pt files against each other")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--imgszs", default=None,
                    help="comma list, e.g. 320,640 -- evaluate each")
    ap.add_argument("--gate", type=float, default=0.55,
                    help="production confidence gate (config SignDetector conf)")
    ap.add_argument("--iou", type=float, default=0.40)
    ap.add_argument("--probe", type=float, default=0.15,
                    help="inference threshold; must be below --gate to see "
                         "the near-misses")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sizes = ([int(s) for s in args.imgszs.split(",")]
             if args.imgszs else [args.imgsz])
    targets = args.compare if args.compare else [args.weights]

    rank = []
    for w in targets:
        wp = Path(w)
        if not wp.exists():
            print(f"! missing, skipped: {w}")
            continue
        for s in sizes:
            stats = evaluate(wp, s, args.gate, args.iou, args.probe, args.limit)
            tag = f"{wp.parent.parent.name}/{wp.name} @{s}"
            gr, gconf, gfp = report(tag, stats, args.gate)
            rank.append((gr, gconf, gfp, tag))

    if len(rank) > 1:
        print("\n\n########  RANKING (by stop/red recall at the gate)  ########")
        print(f"{'R@gate':>8} {'conf':>7} {'FP':>5}   model")
        print("-" * 62)
        for gr, gconf, gfp, tag in sorted(rank, key=lambda r: (-r[0], -r[1])):
            print(f"{gr * 100:7.1f}% {gconf:7.3f} {gfp:5d}   {tag}")


if __name__ == "__main__":
    main()
