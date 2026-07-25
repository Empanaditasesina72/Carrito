# Camera-ready — every edit in paper order (Paper 2069)

Work top to bottom with the manuscript open. Each block says **where** it goes,
**what** to remove and **what** to write. Every number here is measured and
traceable to a file in this repository — nothing is estimated.

Deadline for this pass: **Monday 27 July 2026**. Conference deadline 31 July.

Legend: 🔴 must-fix (a reviewer will check) · 🟠 strongly expected · ⚪ optional

---

## 1. 🔴 TITLE

Add the scope so the title stops promising Sim2Real transfer:

> ...: Monolithic Edge AI Architecture with Software-in-the-Loop Validation and
> On-Device Latency Measurement

---

## 2. 🔴 ABSTRACT

Three edits:

1. Replace every "Sim2Real validation" with **"software-in-the-loop (SIL)
   validation"**.
2. Delete "flawless", "confirms the viability", and any claim of complete
   physical transfer.
3. Add the on-device result — this is the strongest sentence in the paper:

> On the physical vehicle, the per-cycle control latency measured over 999
> consecutive cycles was 3.83 ms on average (p99 6.16 ms), with every cycle
> inside the 20 ms deadline of the 50 Hz loop. A per-stage decomposition shows
> traffic-sign inference costs 0.019 ms of CPU time because it executes inside
> the image sensor.

---

## 3. 🔴 INTRODUCTION

- Soften the claims about distributed architectures to what is actually
  measured (we compare on-sensor inference against the CPU cost, not against a
  distributed system).
- Add one sentence of scope:

> The evaluation reported here is a software-in-the-loop study — the control
> code is byte-identical in simulation and on the vehicle — complemented by an
> on-device latency measurement on the target hardware. Full physical
> trajectory transfer is ongoing work and is stated as a limitation.

---

## 4. 🟠 RELATED WORK

Add a short paragraph plus a comparison table covering: embedded vision sensors
(IMX500-class on-sensor inference), Edge-AI robotics, and HIL/SIL validation
practice. Close by positioning this work as the **integration** of on-device
perception, threaded control, an FSM and PID under a real-time budget — not as
novel individual components (Reviewer 2's point 1).

---

## 5. 🔴 METHODOLOGY

### 5.1 Terminology
Replace **"digital twin"** with **"virtual prototype"** or **"SIL
environment"** everywhere. No physical calibration of the simulator is
demonstrated, so "twin" overclaims.

### 5.2 Formal latency definition (Reviewer 1, point 2)

> We define the per-cycle control latency **L** as the wall-clock duration of
> one control iteration:
>
> **L = t_read + t_lane + t_sign + t_control**
>
> where *t_read* fetches the most recent camera frame, *t_lane* is the lane
> pipeline (inverse-perspective transform, HSV white segmentation and sliding
> windows), *t_sign* is the sign-gating decision, and *t_control* is the PID
> update and servo-command formation. On the IMX500 path the detector executes
> **inside the sensor, concurrently**, so inference time does not enter L; L is
> the CPU work the Raspberry Pi 5 performs each 20 ms cycle. A cycle **misses
> its deadline** when L > 20 ms.

### 5.3 PID controller (all three reviewers)

Sampled at **50 Hz (T_s = 20 ms)**, gains **K_p = 0.08, K_i = 0.002,
K_d = 0.025**, output saturation **±32°**, integral clamp **±25** (anti-windup),
**derivative on the measurement** to avoid derivative kick.

```latex
u(t) = K_p\,e(t) + K_i\!\int_0^t e(\tau)\,d\tau - K_d\,\frac{dy(t)}{dt},
\qquad e(t) = r - y(t),\; r = 0
```

```latex
\begin{aligned}
I_k &= \mathrm{clamp}\!\big(I_{k-1} + e_k T_s,\; -25,\; 25\big)\\
D_k &= -K_d\,(y_k - y_{k-1})/T_s\\
u_k &= \mathrm{clamp}\!\big(K_p e_k + K_i I_k + D_k,\; -32,\; +32\big)\ [^{\circ}]
\end{aligned}
```

Servo command: `angle = 90° + u_k`.

> Gains were tuned by raising K_p until the steering oscillated and halving it,
> then adding K_d for damping and a small K_i to remove steady-state bias. The
> integrator is clamped to ±25 for anti-windup and the derivative is taken on
> the measurement to suppress noise-induced kick.

### 5.4 🔴 Vehicle and camera geometry (Reviewer 1 asked for Ackermann + servo)

Add this table. Every row is reproducible via `python tools/track_geometry.py`.

| Parameter | Value |
|---|---|
| Wheelbase | 31.0 cm |
| Wheel track | 17.2 cm |
| Wheel diameter | 3.0 cm |
| Body (chassis) | 28.0 × 19.0 cm |
| Servo authority | 58°–122° (±32° about 90°) |
| Max wheel angle | **measure at full lock — see the open item below** |
| Minimum turning radius | 44.3 cm at 35°, 49.6 cm at 32° |
| Camera | IMX500, 640 × 480 @ 30 fps |
| Focal length | 490 px |
| Field of view | 66.3° horizontal × 52.2° vertical |
| Camera mounting | 22 cm above ground, 10° downward |
| Ground visible | from 30 cm ahead to the horizon; frame centre at 1.25 m |

> ⚠️ **Two open items before publishing this table.**
> 1. The servo spans ±32° but the code declares a 35° max wheel angle. State
>    the angle **measured with a protractor at full servo lock**; the turning
>    radius follows from it.
> 2. The measured wheelbase (31.0 cm) exceeds the measured chassis length
>    (28.0 cm), which is impossible axle-to-axle. Re-measure the overall
>    length including wheels — publishing a wheelbase longer than the vehicle
>    is an easy target.

### 5.5 🟠 Thresholds and failure handling

- The **700 mm** braking-onset and **120 mm** emergency thresholds follow from
  the sign being reliably resolved at those ranges (see the pinhole table in
  §7.3) and from the ToF minimum range.
- **Stale frames**: the perception threads publish latest-value only; the
  control loop never blocks on a frame.
- Thread synchronisation: locks and bounded queues, non-blocking reads.
- The wait state uses a monotonic clock, never a blocking sleep, so the loop
  keeps serving the FSM.

### 5.6 🟠 FSM naming
Make every state name in the text identical to Figure 3
(`CRUCERO → PRECAUCION → FRENADO → ESPERA → REANUDAR`), or rename both to
English consistently. Do not mix.

---

## 6. 🔴 EXPERIMENTAL SETUP

> Latency is measured per control cycle with a monotonic high-resolution timer
> (`time.perf_counter`) over N ≥ 999 consecutive cycles at the 50 Hz loop rate;
> the warm-up cycle is discarded. We report the full distribution — mean,
> median, standard deviation as jitter, p95, p99, maximum — and the
> deadline-miss rate against the 20 ms budget, rather than the mean alone.

State the platform: Raspberry Pi 5, Sony IMX500, Python 3.13, OpenCV 4.13,
Ultralytics 8.4.33, Unity 6000.4. Report CPU temperature during the run
(47.2 → 48.3 °C, stable, no throttling).

**Track** (Figure: photo): 346 cm straight, lane 56.5 cm wide between the outer
solid lines (27.5 cm from the left solid to the dashed centre, 29.0 cm from the
dashed centre to the right solid), 4 cm white lines on a matte dark surface,
STOP sign with an 8.5 cm octagon on a 9 cm post (17.5 cm total) at 150 cm, and
a 29 cm perpendicular parking slot on the left.

---

## 7. 🔴 RESULTS

### 7.1 Replace Table 2 (it currently duplicates Table 1)

| ID | Purpose | Initial condition | Reps | Metric | Acceptance | Result |
|---|---|---|---|---|---|---|
| **P1** | Control-loop latency | 50 Hz loop | 2423 SIL + 999 on-device | mean, p95, p99, deadline-miss | < 20 ms | 3.83 ms mean on-device, p99 6.16, **100 % < 20 ms** |
| **P2** | PID braking at STOP | cruising, STOP ahead | 1 SIL | stop distance, steady-state error, overshoot | 270 ± 30 mm, no overshoot | 292.5 mm, error 22.5 mm, no overshoot |
| **P3** | FSM transitions | full run (drive→stop→resume→park) | 1 SIL | states visited, non-blocking wait | STOP 5/5 + parking 3/3 | 5/5 + 3/3, wait non-blocking |

### 7.2 🔴 Latency — the distribution, not a mean

| Configuration | n | mean | median | jitter | p95 | p99 | max | < 20 ms |
|---|---|---|---|---|---|---|---|---|
| SIL (Unity control loop) | 2423 | 9.23 | 9.18 | 1.29 | 11.34 | 13.64 | 20.83 | 99.96 % |
| **Pi 5 + IMX500 (on-device)** | 999 | **3.83** | 3.72 | **0.43** | 4.49 | **6.16** | 7.35 | **100.00 %** |
| Pi 5 compute-only (synthetic) | 1199 | 3.97 | 3.85 | 0.66 | 5.20 | 6.09 | 7.98 | 100.00 % |

All values in ms. Measured 2026-07-24 on the physical vehicle.

**Per-stage decomposition (on-device) — the empirical core of the paper:**

| Stage | mean (ms) | share |
|---|---|---|
| `t_read` | 0.222 | 5.8 % |
| `t_lane` | **3.583** | **93.6 %** |
| **`t_sign` (on-sensor NPU)** | **0.019** | **0.5 %** |
| `t_control` | 0.010 | 0.3 % |

> Over 999 consecutive control cycles on the physical vehicle the latency
> averaged 3.83 ms (median 3.72 ms) with 0.43 ms jitter; p95 and p99 were
> 4.49 ms and 6.16 ms, the maximum was 7.35 ms, and **100 % of cycles met the
> 20 ms deadline with zero misses**, at a stable CPU temperature. The per-stage
> decomposition shows sign detection consumes **0.019 ms of CPU** — inference
> runs inside the IMX500 and the CPU only parses output tensors — while the
> classical lane pipeline accounts for 93.6 % of the budget. The on-device
> figure is *lower* than the SIL figure because the latter includes JPEG
> encoding and TCP transport from the simulator, which the physical system does
> not incur.

### 7.3 🔴 Braking, against a tolerance band defined a priori

Setpoint **270 mm**, tolerance **±30 mm** (`STOP_TOLERANCE_MM`, fixed before
testing). Result: rest at **292.5 mm**, steady-state error **22.5 mm (8.3 %)**,
**no overshoot** (minimum approach distance equals the final distance).

Do **not** call this ideal convergence — report the residual error against the
band, as above.

Geometric soundness of the experiment (from `tools/track_geometry.py`): the
8.5 cm octagon subtends 59.5 px at the 700 mm braking onset and 154 px at the
270 mm setpoint, and remains fully inside the frame down to 178 mm, so the
detector never loses the sign during braking.

### 7.4 🔴 CPU-vs-NPU ablation (all three reviewers asked for this)

Measured 2026-07-25 on the vehicle with `tools/bench_ablation.py`, 599 control
cycles per configuration at 50 Hz.

| Metric | IMX500 NPU (on-sensor) | CPU detector thread |
|---|---|---|
| Loop latency, mean | **3.75 ms** | 5.45 ms (+45 %) |
| Jitter (std) | **0.39 ms** | 0.61 ms (+56 %) |
| p95 / p99 | **4.22 / 6.02 ms** | 6.51 / 8.16 ms |
| Maximum | **6.17 ms** | 9.30 ms |
| Deadline misses (20 ms) | 0 | 0 |
| In-loop sign-gating cost | 0.013 ms | 0.018 ms |
| **CPU utilisation** | **19.4 %** | **51.0 %** (2.6×) |
| **Temperature rise** | **+1.1 °C** | **+5.5 °C** (5×) |
| Per-inference cost | 0.019 ms of CPU | **81.3 ms** |
| Achievable detection rate | camera rate, 30 fps | **12.3 detections/s** |

> **Both configurations met the 20 ms deadline**, because the architecture already
> runs the detector in its own thread and the control loop only reads the latest
> result — the in-loop gating cost is 0.013 ms against 0.018 ms, effectively
> identical. The difference is not whether the loop meets its deadline but what
> it costs to do so. Moving inference into the sensor cuts CPU utilisation from
> 51.0 % to 19.4 %, reduces the thermal rise over the same run by a factor of
> five, lowers loop jitter by 36 %, and raises the achievable detection rate from
> 12.3 to 30 per second, so every camera frame is inspected instead of roughly
> two in five. On a passively cooled platform that also runs the lane pipeline,
> the state machine, the distance sensors and the lighting, that reclaimed
> headroom is what makes the design viable, and it is what leaves room for a
> learned steering model on the CPU.

> **Caveat, state it explicitly.** The 81.3 ms figure is the PyTorch checkpoint.
> The NCNN export, which the vehicle prefers when available, could not be timed
> because the `ncnn` runtime is not installed on this Pi; the repository reports
> it as 3–4× faster, which would place it at roughly 20–27 ms — still at or above
> the entire control-cycle budget. Report 81.3 ms as measured and the NCNN figure
> as an estimate, or install `ncnn` and measure it.

### 7.5 🟠 Perception metrics

| Split | mAP@50 | mAP@50-95 | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Validation | 0.995 | 0.647 | — | 1.00 (all classes) | — |
| Held-out test @ conf 0.55 | — | — | 0.993 | 0.986 | **0.990** |

> Because only the *stop* and *red* classes gate braking, the safety-critical
> false-trigger rate is bounded by the precision of those two classes.

---

### 7.6 🟠 Detector operating range (measured, and it belongs in Limitations)

The detector was trained on a close-up sign dataset, so its usable range on the
track is bounded by apparent size rather than by confidence tuning. Measured by
scaling a held-out test image into a 640×480 frame at the deployed imgsz of 320:

| Apparent octagon height | % of frame height | Detected |
|---|---|---|
| 150 px | 31.2 % | stop, 88 % |
| 100 px | 20.8 % | stop, 81 % |
| 60 px | 12.5 % | stop, 80 % |
| **35 px** | **7.3 %** | **stop, 71 % — threshold** |
| 28 px | 5.8 % | no |

Training images place the sign at 27–45 % of frame height; on the track at 1.5 m
an 8.5 cm octagon subtends 5.8 %, below the threshold. Combined with the pinhole
model this gives a first-detection distance of **1.19 m** for the deployed sign,
against a braking onset of 0.70 m — so the sign is acquired 49 cm before braking
must begin.

> State in Limitations: the sign detector reaches F1 = 0.990 on held-out data but
> that figure is obtained at the apparent scale of its training set. On the
> vehicle the usable range is bounded below 1.19 m for an 8.5 cm sign, because the
> training data contains only close-ups and no track imagery. Enlarging the sign
> or fine-tuning on track-scale imagery extends the range; the reported
> detection-range figure should accompany the F1 score rather than be omitted.

## 8. 🔴 LIMITATIONS (new subsection — do not skip)

> The evaluation is software-in-the-loop plus an on-device latency measurement.
> We do not claim validated physical trajectory transfer: closed-loop lane
> following and the parking manoeuvre were exercised in simulation, and the
> braking distribution reported here is simulated. The parking manoeuvre is
> open-loop and time-parameterised, so it requires per-track calibration. The
> simulator is a virtual prototype, not a calibrated twin: its dynamics are
> kinematic and unmodelled effects (tyre slip, battery sag, actuator lag) are
> absent. We report no worst-case execution-time bound, so the system is
> soft real-time and we avoid the term "deterministic". Physical closed-loop
> trials on the instrumented track are the immediate next step.

---

## 9. 🔴 CONCLUSION

- Translate the untranslated Spanish paragraph (Reviewer 2).
- Replace "flawless execution" and "confirms the viability" with the bounded
  statements: "100 % of cycles met the 20 ms deadline on-device", "within the
  ±30 mm tolerance band".
- Restate the contribution as the integration, and point to Limitations.

---

## 10. 🟠 REPRODUCIBILITY

> The control software, simulator, configuration, logs and plotting scripts are
> archived at [DOI]. Reported versions: Python 3.13, OpenCV 4.13, Ultralytics
> 8.4.33, Unity 6000.4. Track calibration is distributed as `track_calib.json`
> and the vehicle geometry is reproducible via `tools/track_geometry.py`.

---

## 11. FIGURES — attach each file separately, ≥ 300 dpi

All five are verified at 300 dpi in `TMR2026/paper_results/`:

| Figure | File | Status |
|---|---|---|
| Latency vs time (SIL) | `fig1_latency.png` | ✅ 2700×1350 |
| PID braking | `fig2_braking.png` | ✅ 2700×1350 |
| FSM timeline | `fig3_fsm.png` | ✅ 3000×1350 |
| Latency histogram + CDF (SIL) | `P1_latency_distribution.png` | ✅ 3300×1260 |
| **Latency histogram + CDF (on-device)** | `bench_latency_pi_distribution.png` | ✅ 3300×1260 |
| **Figure 2 — hardware photo** | *pending* | 📷 Sony camera |
| Track photo | *pending* | 📷 Sony camera |

Photo requirements: tripod, f/5.6–f/8, ISO 100–400, diffuse light, no direct
flash, plain background, maximum resolution, exported as PNG or TIFF.
**Never transfer them through WhatsApp** — recompression drops them below
300 dpi. Use USB or the SD card.

---

## 12. Submission package

- `WITCOM2026_2069.zip` containing the manuscript source (.docx or .tex), the
  final PDF, and every figure as a separate file.
- Signed Springer Consent-to-Publish.
- Email to **upiitawitcom@gmail.com**.
- Registration by 15 August 2026.
