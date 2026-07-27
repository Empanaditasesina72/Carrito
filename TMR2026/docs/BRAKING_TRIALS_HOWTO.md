# P2 — Braking trials, step by step

The last measurement the camera-ready needs. Ten runs, roughly 25 minutes.

Everything else for the paper is already measured; this is what turns a simulated
braking figure into a physical one.

---

## What the experiment actually measures

The car drives straight at a fixed low speed. The camera detects the STOP sign,
the pinhole model converts its apparent height to a distance, and when that
distance falls to **320 mm** the real `AutonomousFSM` triggers braking — the same
code path that runs in production. The car coasts to a stop, and **you** measure
where it actually ended up.

| Quantity | Value |
|---|---|
| FSM braking trigger | 320 mm |
| **Setpoint reported in the paper** | **270 mm** |
| Acceptance band | **±30 mm** → 240–300 mm |
| Simulator result to compare against | 292.5 mm |

Steering runs **closed-loop by default**: the lane pipeline feeds the PID and the
car keeps itself centred while it approaches, the same way it does in the
simulator. Pass `--straight` to force the wheels centred instead, which isolates
the braking controller but lets the car drift.

---

## Before you start

**1. Power.** Pi on the official 27 W supply. Motor and servo on the battery —
their V+ must NOT come from the Pi's 5 V rail.

**2. Free the camera.** The service grabs it at boot:
```bash
sudo systemctl stop carrito_tmr
```

**3. Mark a start line** on the track with tape. Every trial begins there, or the
runs are not comparable.

**4. Sign position.** Leave it where detection was confirmed. The car needs to
see it for the whole approach; it does not need to see it from the start line.

**5. WHEELS OFF THE GROUND FIRST.** Run one trial holding the car up. Confirm the
wheels spin, then stop when the sign is seen. Only then put it on the track.

**6. STEERING TRIM — do this before any driving session.** The first real session
(2026-07-26) ended with the car diagonal across the lane, and the numbers say why a
crooked car cannot be fixed by the controller alone: a trim bias of `b` degrees
leaves a standing offset of `b / Kp` pixels (6° → 75 px ≈ 11 cm, at the lane edge),
and the integrator at Ki=0.002 needs ~40 s to cancel what a 3 s run gives it. The
bias must be removed at the servo, not fought by the PID:

```bash
# 1) Push test, motor off: roll the car 2 m by hand. If it curves on its own,
#    the problem is mechanical (tie rod), not software.
# 2) Powered trim: short bursts, adjust until it tracks straight, then persist
#    the winning value in config.py:SERVO_TRIM_DEG.
python tools/trim_steering.py --drive --seconds 1.5 --cruise 25 --kick 60
```

**7. Heading feedforward (optional, after trim).** The pipeline now measures the
lane's lean (`heading` column in diag_track) and the FSM can steer on it —
`config.py:STEER_HEADING_GAIN`, shipped disabled. Enable ONLY after confirming the
sign on the car: nose rotated left ⇒ `head` positive, nose right ⇒ negative. With
the sign wrong it is positive feedback. Start at 1.5.

---

## The command

```bash
cd ~/Carrito/TMR2026
python tools/bench_braking_physical.py --manual --trials 10 --cruise 25 --max-drive 3.0
```

| Flag | Why |
|---|---|
| `--manual` | no ToF; you measure with a tape |
| `--cruise 25` | 25 % PWM, deliberately slow |
| `--max-drive 3.0` | **the only automatic stop.** Without the ToF there is no emergency cutoff: if the sign is missed, this brakes the car after 3 s so it cannot run off a 3.46 m track |

Ctrl+C brakes and exits at any moment.

---

## What each trial looks like

```
[Trial 3/10] Place car at the start line, press Enter (Ctrl+C to stop)...
  -> the car stopped (braked, 2.4 s). Now measure it.
  [3] Measured gap car-to-sign in mm (Enter = trial failed): 285
  -> stopped 285.0 mm  (within tol: True, braked)
```

1. Put the car on the start line, press **Enter**.
2. The car drives and brakes on its own. Keep a hand near it.
3. **Do not move the car.** Measure it where it stopped.
4. Type the number in **millimetres** and press Enter.

Press Enter with nothing typed to mark a trial as failed — do that if the car
was bumped, left the lane, or you are unsure of the reading.

---

## Where to measure — this must be identical every time

**From the front face of the camera lens, horizontally, to the face of the STOP
octagon.**

```
   [camera lens]|<------------ measure this ------------>|[STOP face]
        car                                                  sign
```

The camera is the reference because the distance the controller acts on comes
from the pinhole model, which measures lens to sign. Measuring from the bumper
instead adds a constant offset and shifts every result.

Keep the tape horizontal and parallel to the track. Read to the nearest 5 mm;
that is well inside the ±30 mm band.

---

## Reading the results

The run writes `validation_results/braking_physical.csv` and prints a summary.

`stop_reason` tells you whether each trial counts:

| Value | Meaning |
|---|---|
| `braked` | ✅ stopped because it saw the sign — **this is a valid trial** |
| `max_drive` | ⚠️ never detected the sign, stopped on the 3 s timer — **exclude it** |

If several trials come back `max_drive`, stop and tell me: the detection is
failing and more runs will not help.

Expected: mean near 270–300 mm with a small spread, and most trials inside
240–300 mm.

---

## If something goes wrong

| Symptom | What it means |
|---|---|
| Car does not move | Battery off, or motor V+ not connected |
| Every trial is `max_drive` | Sign not being detected — stop and report |
| Car stops far too early (> 400 mm) | Detecting something else as a sign, or the sign is closer than assumed |
| Car hits the sign | Raise `--max-drive` is NOT the fix — lower `--cruise` to 20 |
| Car drifts off the lane | Only expected with `--straight`. By default the lane PID corrects; if it still drifts, check `tools/diag_track.py` reports a stable lock |

---

## When you are done

Send me the CSV, or just read the ten numbers out. I compute mean ± std, the
within-tolerance rate and the simulation-vs-physical comparison, then write them
into the camera-ready package as the P2 physical row.
