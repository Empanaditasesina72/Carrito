"""Log the Pi's power state to disk so a shutdown leaves evidence.

The vehicle dies after a few minutes on the external battery and the cause
determines the fix. Two very different failures look identical from the outside:

  * the supply cannot hold 5 V under load -> undervoltage, `throttled` sets its
    low bits before the board dies, and the fix is a stronger supply;
  * the power bank decides the load is too small and switches itself off -> a
    clean cut with no warning at all, and the fix is the bank's low-current mode
    or a higher standing draw.

Sampling to a file every second separates them: flush after every write, so the
last line on disk is the state immediately before power was lost. A file that
ends with rising undervoltage flags means the first case; a file that just stops
mid-stride with everything nominal means the second.

Run it detached so it outlives the SSH session:
    nohup python tools/power_monitor.py > /dev/null 2>&1 &

Then read the tail after a failure:
    python tools/power_monitor.py --report

Output: validation_results/power_monitor.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "validation_results" / "power_monitor.csv"

THROTTLE_BITS = {
    0:  "undervoltage NOW",
    1:  "arm freq capped NOW",
    2:  "currently throttled",
    3:  "soft temp limit NOW",
    16: "undervoltage HAPPENED",
    17: "arm freq capped HAPPENED",
    18: "throttling HAPPENED",
    19: "soft temp limit HAPPENED",
}


def _vcgen(arg: str):
    try:
        return subprocess.check_output(["vcgencmd"] + arg.split(),
                                       timeout=3).decode().strip()
    except Exception:
        return ""


def _throttled() -> int:
    out = _vcgen("get_throttled")
    try:
        return int(out.split("=")[1], 16)
    except Exception:
        return -1


def decode(flags: int) -> str:
    if flags <= 0:
        return "ok" if flags == 0 else "unknown"
    return "; ".join(t for b, t in THROTTLE_BITS.items() if flags & (1 << b))


def _power_w() -> float | None:
    out = _vcgen("pmic_read_adc")
    if not out:
        return None
    cur, vol = {}, {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2 or "=" not in parts[1]:
            continue
        name = parts[0]
        try:
            # vcgencmd appends the unit to the number: "current(0)=0.0985A"
            val = float(parts[1].split("=")[1].rstrip("AVav"))
        except ValueError:
            continue
        if name.endswith("_A"):
            cur[name[:-2]] = val
        elif name.endswith("_V"):
            vol[name[:-2]] = val
    rails = sum(cur[k] * vol[k] for k in cur if k in vol)
    return rails / 0.85 if rails else None


def report(path: Path) -> int:
    if not path.exists():
        print(f"No log at {path}")
        return 1
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        print("Log is empty.")
        return 1
    print("=" * 66)
    print(f"  samples          : {len(rows)}")
    print(f"  first / last      : {rows[0]['iso']}  ->  {rows[-1]['iso']}")
    span = float(rows[-1]["uptime_s"]) - float(rows[0]["uptime_s"])
    print(f"  covered           : {span/60:.1f} min")
    bad = [r for r in rows if r["throttled"] not in ("0x0", "0")]
    print(f"  samples with flags: {len(bad)}")
    if bad:
        print(f"  FIRST flag at     : {bad[0]['iso']}  -> {bad[0]['decoded']}")
        print()
        print("  => UNDERVOLTAGE. The supply could not hold 5 V. A power bank's")
        print("     low-current mode will not fix this; it needs a supply that")
        print("     sustains 5 V at the current the Pi draws.")
    else:
        print()
        print("  => No undervoltage in any sample. If the board still died, the")
        print("     supply cut cleanly rather than sagging, which points at the")
        print("     bank switching itself off, not at insufficient capacity.")
    ws = [float(r["watt"]) for r in rows if r["watt"]]
    if ws:
        print()
        print(f"  power  mean/min/max: {sum(ws)/len(ws):.2f} / {min(ws):.2f} / "
              f"{max(ws):.2f} W")
    ts = [float(r["temp_c"]) for r in rows if r["temp_c"]]
    if ts:
        print(f"  temp   mean/max    : {sum(ts)/len(ts):.1f} / {max(ts):.1f} C")
    print("=" * 66)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--report", action="store_true", help="summarise an existing log")
    args = ap.parse_args()
    path = Path(args.out)

    if args.report:
        return report(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    f = path.open("a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if fresh:
        w.writerow(["iso", "uptime_s", "throttled", "decoded", "volt_core",
                    "temp_c", "watt"])
        f.flush()

    print(f"[PWR] logging to {path} every {args.interval:.1f} s. Ctrl+C to stop.")
    try:
        while True:
            flags = _throttled()
            volt = _vcgen("measure_volts").replace("volt=", "").replace("V", "")
            temp = _vcgen("measure_temp").replace("temp=", "").replace("'C", "")
            watt = _power_w()
            with open("/proc/uptime") as u:
                up = float(u.read().split()[0])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), f"{up:.1f}",
                        hex(flags) if flags >= 0 else "", decode(flags),
                        volt, temp, f"{watt:.2f}" if watt else ""])
            # Flush every sample: the whole point is that the last line on disk
            # survives an abrupt power loss.
            f.flush()
            os.fsync(f.fileno())
            if flags > 0:
                print(f"[PWR] {decode(flags)}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[PWR] stopped.")
    finally:
        f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
