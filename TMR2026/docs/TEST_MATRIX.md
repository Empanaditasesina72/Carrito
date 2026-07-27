# Test matrix — Table 2 replacement (WITCOM 2026 / CCIS, paper 2069)

All three reviewers rejected Table 2 as a duplicate of Table 1 and asked for a
proper matrix: purpose, conditions, repetitions, metrics, acceptance criteria,
results. This is that table, built only from measurements that exist in this
repository, each row traceable to the file that produced it.

**Every number below is sourced.** Where a figure is still pending it says so
rather than carrying a simulated value into a physical row — the single mistake
most likely to be caught on a second review.

---

## The table

| ID | Purpose | Environment | Initial condition | Reps | Metric | Acceptance | Result |
|----|---------|-------------|-------------------|------|--------|------------|--------|
| **P1a** | Control-loop latency, on-device | Pi 5 + IMX500 | 50 Hz loop, warm-up discarded | 999 cycles | L = t_read + t_lane + t_sign + t_control | < 20 ms (50 Hz deadline) | **3.83 ms mean**, median 3.72, jitter 0.43, p95 4.49, p99 6.16, max 7.35 — **100 % under deadline, 0 misses** |
| **P1b** | Control-loop latency, SIL | Unity + PC | identical control code | 2 423 cycles | same | < 20 ms | 9.23 ms mean, p99 13.64, **99.96 %** under deadline |
| **P2** | Braking to the STOP setpoint | **physical** | cruise 30 % PWM, right lane | **10** | lens-to-octagon distance | 270 ± 30 mm, no overshoot | ⏳ **PENDING** — 2 pilot runs gave 362 and 412 mm (see note) |
| **P2s** | Braking, SIL reference | Unity | cruise, STOP at track end | 1 | same | 270 ± 30 mm | 292.5 mm, steady-state error **22.5 mm (8.3 %)**, no overshoot |
| **P3** | FSM state transitions | SIL + physical | full run | 5 + 2 | transition sequence, ESPERA duration | 5/5 sequence, 5.0 s hold, non-blocking | 5/5 in SIL; physically CRUCERO→PRECAUCION→FRENADO→ESPERA→REANUDAR with a **5.00 s** hold |
| **P4** | Ablation: NPU vs CPU inference | Pi 5 | same model, imgsz 320 | 599 cycles / cfg | loop latency, CPU load, deadline miss | NPU must not exceed the budget | NPU loop **3.75 ms**, CPU busy **19.4 %**, 0 misses · CPU inference **81.3 ms** = **4.1×** the whole cycle, CPU busy 55.6 % |
| **P5** | Sign detector accuracy | held-out benchmark | 294 degraded frames of the unseen test split | 294 | recall at the production gate, mean confidence, false positives | recall ≥ 95 % at conf 0.55 | **100 % recall**, mean conf **0.824**, **0 false positives** (stop/red, the only gating classes) |
| **P6** | Lane following, closed loop | physical | car centred in the right lane | 14 frames + driven runs | lane confidence, lateral error, weave | ≥ 90 % confidence, ‖error‖ < 5 cm | **100 % confidence, both lines 14/14**; centred reads **+0.1 px**; end of a driven run **−11.5 px (−1.7 cm)**, weave **0.7 px (0.1 cm)** |
| **P7** | Exposure adaptation | physical | exposure forced to blackout (V = 0.0) | 1 | time to return to the working band | recover < 10 s | **~4 s** (exp 1000→33000 µs, then gain 1.09→16.0, to V = 62) |

---

## Sources

| Row | Produced by | Artifact |
|---|---|---|
| P1a | `tools/bench_latency.py` → `tools/latency_stats.py` | `paper_results/bench_latency_pi_stats.txt`, `bench_latency_pi.csv` |
| P1b | `main_simulator.py --validate` | SIL logs |
| P2 | `tools/bench_braking_physical.py --manual --trials 10` | `validation_results/braking_physical.csv` |
| P2s | Unity SIL run | `paper_results/fig2_braking.png` |
| P3 | `control/fsm.py` transitions, logged by `tools/demo_full.py` | run logs |
| P4 | `tools/bench_ablation.py` | `paper_results/ablation_cpu_vs_npu.txt` |
| P5 | `tools/eval_hard.py --imgsz 320` | `TMR2026/datasets/car_hard` (degraded copies of the untouched test split) |
| P6 | `tools/diag_track.py`, `tools/drive_straight.py` | run logs |
| P7 | `vision/camera_stream.py:_adapt_exposure` | forced-blackout test log |

---

## Notes that belong in the text, not only in the table

### P2 — why the pilot runs land long, and why it is a finding

The two pilot runs stopped at 362 and 412 mm against a 270 mm setpoint. That gap
is **geometric, not a tuning error**, and it is worth a paragraph because it is
exactly the kind of simulation-versus-physical discrepancy the reviewers asked to
see documented.

The sign stands ~28 cm off the camera's optical axis and the lens covers ~±33°,
so the sign leaves the frame at

    d = 0.28 / tan(33°) ≈ 43 cm

The vehicle is therefore **blind to the sign before it can ever reach the 320 mm
bounding-box threshold** — that threshold can never fire on the physical car. Every
physical stop is instead decided by the loss-of-detection rule, at whatever
distance the sign happened to leave the field of view. The simulator does not
model the lens field of view at all, so in SIL the 320 mm threshold *does* fire
and the vehicle stops at 292.5 mm.

`SIGN_LOST_COAST_S` (0.6 s) now dead-reckons across the blind stretch. The
physical fix is to move the sign closer to the lane edge, so it stays in frame
longer.

### P4 — one number to re-measure before submission

The ablation reports CPU inference at **81.3 ms** (2026-07-25). A measurement on
2026-07-27, same model and imgsz, gave **64.4 ms mean / 68.8 ms p99**. The
difference is most likely the CPU governor (later set to `performance`) or the
retrained weights. **Re-run `tools/bench_ablation.py` before submitting** and use
one consistent figure; do not quote both.

Note also that the NPU path is currently disabled (`USE_IMX500_NPU = False`)
because the `.rpk` was exported with the channel order swapped. The P4 numbers
remain valid as measured, but the shipped configuration is the CPU path — the text
must say so.

### P5 — what the benchmark is, and what it is not

`car_hard` is 294 frames built from the Roboflow **test** split, which no training
run has ever seen, degraded to match the camera's measured statistics (brightness,
contrast, saturation, sharpness and noise all measured on real frames). It is a
fair held-out benchmark for the detector.

It is **not** a measure of performance on this track: it contains no photograph
taken by this vehicle's camera. On the real camera the same sign reads 56–81 %
confidence depending on light and distance. State the benchmark result as a
detector metric, and the on-track confidence range separately.

### Terminology

Per the rebuttal: **SIL / virtual prototype**, never "digital twin" or "Sim2Real
validation"; "within the ±30 mm band", never "ideal" or "flawless".

---

## LaTeX

```latex
\begin{table}[t]
\centering
\caption{Evaluation test matrix. SIL rows use the identical control code executed
against the virtual prototype; on-device rows were measured on the Raspberry~Pi~5.}
\label{tab:matrix}
\footnotesize
\begin{tabular}{@{}llcllp{3.6cm}@{}}
\toprule
ID & Purpose & Reps & Metric & Acceptance & Result \\
\midrule
P1a & Loop latency (device) & 999   & $L$ per cycle & $<20$\,ms & 3.83\,ms mean, p99 6.16, jitter 0.43, 100\,\% under deadline \\
P1b & Loop latency (SIL)    & 2423  & $L$ per cycle & $<20$\,ms & 9.23\,ms mean, p99 13.64, 99.96\,\% \\
P2  & Braking (physical)    & 10    & stop distance & $270\pm30$\,mm & \emph{pending} \\
P2s & Braking (SIL)         & 1     & stop distance & $270\pm30$\,mm & 292.5\,mm, err.\ 22.5\,mm (8.3\,\%), no overshoot \\
P3  & FSM transitions       & 5+2   & sequence, hold & 5/5, 5.0\,s & 5/5 SIL; 5.00\,s hold on device \\
P4  & NPU vs CPU inference  & 599   & latency, CPU & no deadline miss & NPU 3.75\,ms / 19.4\,\% CPU; CPU inference 81.3\,ms ($4.1\times$ budget) \\
P5  & Detector accuracy     & 294   & recall @0.55, FP & recall $\geq95$\,\% & 100\,\% recall, conf.\ 0.824, 0 FP \\
P6  & Lane following        & 14    & confidence, error & $<5$\,cm & 100\,\%, $-1.7$\,cm at run end, weave 0.1\,cm \\
P7  & Exposure adaptation   & 1     & recovery time & $<10$\,s & $\sim$4\,s from full blackout \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Filling P2 once the trials are run

```
python tools/bench_braking_physical.py --manual --trials 10 --cruise 30 --kick 70
```

Rows are written to `validation_results/braking_physical.csv` one at a time with
`fsync`, so an interrupted session keeps everything measured so far. Count only
rows whose `stop_reason` is `braked`; `max_drive` means the sign was never
detected and the trial does not belong in the statistics.

Report: mean ± std, the number of trials inside 240–300 mm, and the
SIL-versus-physical comparison against the 292.5 mm reference.
