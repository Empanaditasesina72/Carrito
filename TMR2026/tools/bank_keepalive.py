"""Periodic current burst to stop a power bank from switching itself off.

Measured on this vehicle: the bank cuts power after about three minutes with no
undervoltage flags at any point, so it is not failing to supply current, it is
deciding nothing is connected. The Pi draws ~2.4 W from a 200 W bank, about 1 %
of its rating, and a flat load at that level reads as "device finished charging".

Raising the average draw was already tried -- pinning the governor to performance
took it to 3.6 W and the bank still cut out. What is left is the shape of the
load rather than its level: many banks reset their idle timer on a change in
current, not on an absolute threshold. This spends a fraction of a second every
interval running all cores flat out, which roughly triples instantaneous draw,
then goes back to sleep.

This is a workaround for a hardware behaviour and it may simply not work on this
bank. Run it with tools/power_monitor.py logging and give it longer than the
three minutes it previously survived; if it still dies, the bank cannot run the
Pi and no software will change that.

Usage (on the Pi):
    nohup python tools/bank_keepalive.py > /dev/null 2>&1 &
    python tools/bank_keepalive.py --interval 15 --burst 0.4
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time


def _spin(deadline: float) -> None:
    x = 0.0
    while time.monotonic() < deadline:
        for _ in range(20000):
            x += 1.0000001
    return None


def burst(seconds: float, workers: int) -> None:
    deadline = time.monotonic() + seconds
    procs = [multiprocessing.Process(target=_spin, args=(deadline,))
             for _ in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=20.0,
                    help="seconds between bursts")
    ap.add_argument("--burst", type=float, default=0.3,
                    help="seconds each burst lasts")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    print(f"[KEEPALIVE] {args.burst:.2f} s burst on {args.workers} cores "
          f"every {args.interval:.0f} s. Ctrl+C to stop.")
    print("[KEEPALIVE] Workaround for the bank's idle cutoff, not a fix. Watch "
          "power_monitor.csv and give it more than 3 minutes before believing it.")
    n = 0
    try:
        while True:
            burst(args.burst, args.workers)
            n += 1
            if n % 10 == 0:
                print(f"[KEEPALIVE] {n} bursts")
            time.sleep(max(0.0, args.interval - args.burst))
    except KeyboardInterrupt:
        print(f"\n[KEEPALIVE] stopped after {n} bursts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
