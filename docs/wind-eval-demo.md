# Wind-Field Generation for Balloon RL — Benchmark v2

A **reference-based** benchmark for wind generators: every metric is a raw, literature-
standard distance to held-out ERA5 (spectral residuals, Wasserstein-1, effective
resolution) — no 0–1 scores, no composites. The spec is `Metrics & Baselines (2).pdf`;
every design decision and calibration finding is logged in
[`benchmark-v2-changes.md`](benchmark-v2-changes.md). The previous reference-free suite
is fully retired (git history has it).

## How to read the board

- **Reference** = held-out ERA5: days 8–14 of Jan/Apr/Jul/Oct 2023, NE Pacific, model
  levels 49–66 — **zero overlap with the training dates** (days ~15–28).
- **`self-split floor`** = one disjoint half of the held-out set scored against the
  other: the same-distribution sampling-noise level. A generator is "good on a metric"
  when it approaches the floor, not when it hits zero.
- **Baselines are a distilled set — each earns its place:** *white noise* is the trivial
  anchor for the spectral metrics (but at-floor on marginal W1 *by construction* — its
  documented blind spot); *phase shuffle* is its complement (right stats, zero structure —
  fires the distribution metrics and brackets the classic diffusion failure mode);
  *BLE-VAE* is the prior state of the art to beat. Dropped rows (idiff toy,
  time-shuffled) keep their calibration role inside the test suite
  (`tests/test_windeval/test_metrics_v2.py`, 14 checks). Temporal rows (kinematic toy,
  M2/M3) return via `--temporal` when temporal training resumes.

## Headline results (full table: `src/eval/windeval/benchmark_v2_report.md`)

| Metric (lower = better) | floor | **idiff trained** | ble_vae | phase-shuffle | white noise |
|---|---|---|---|---|---|
| SR_E (log-PSD residual) | 0.25 | **2.09** | 2.72 | 5.76 | 8.96 |
| SR_vort | 0.25 | **2.10** | 2.15 | 5.85 | 8.99 |
| L_eff — trusted down to (km) | 56† | **673** | 842 | 56† | 3367 |
| W1 shear u ((m/s)/km) | 0.20 | **1.70** | N/A | 39.6 | 18.4 |
| W1 u marginal (m/s) | 1.07 | **4.26** | 13.7 | 8.78 | 0.53* |
| tail err 0.1% (m/s) | 2.92 | **15.6** | 13.2 | 5.97 | 2.59* |

† resolved over the whole compared range. \* the moment-matched noise anchor has correct
marginals *by construction* — the distribution metrics' designed blind spot; SR_E is what
catches it (8.96). That's why the suite has multiple families.

**What the new suite says about the trained model** (sharper than the old suite's
"spectrum solved"):

- It tracks the ERA5 energy spectrum through the synoptic range but hits a noise floor at
  fine scales — **trustworthy only above ~673 km wavelength** (see the PSD triptych).
- **Tails are the main gap**: extreme-quantile error ~5× the floor (11.5–15.6 m/s) — the
  model under-produces extreme winds, consistent with the known under-dispersion.
- Shear and marginal W1 sit well above the floor but far below every anchor: the vertical
  structure is broadly right, amplitude/detail still short.

## Tiling penalty (the procedural check)

Run the suite on a single-tile (64²) and a multi-tile (192²) generation; penalty =
multi − single. For the **trained** model every penalty is ≈ 0 or negative (blending
windows *helps* — more samples, no seam damage). During development the analytic toy
demonstrated the failure this check exists for (SR_div +1.86 — tiling injecting seam
divergence). (This one number replaces the old seam/revisit/budget/extent axis.)

## Temporal (gated behind `--temporal` until M2/M3 land)

The kinematic toy (mean-wind advection — the no-learned-dynamics floor) scored
SR_time 5.46 vs floor 0.28 (frozen field, no temporal power) and tracer final-spread
ratio 0.20 (under-transports). The M3 (autoregressive) and M2 (spacetime) rows fill in
when their training checkpoints land — same `SR_time` / trajectory-dispersion metrics,
no new machinery. Calibration note: dispersion deliberately does *not* punish
time-shuffling (the jet is quasi-stationary — winds aren't passive tracers); it punishes
wrong transport, and SR_time punishes incoherence.

## Figures

`docs/figures/benchmark_v2/`: PSD triptych (E/divergence/vorticity vs ERA5), marginal
log-density of u (tail inspection), temporal PSD + tracer-dispersion curves.

## Reproduce

```bash
# from balloon-env-dev/, full pixi env:
PY=".pixi/envs/default/bin/python"
PYTHONPATH=. $PY -m src.eval.windeval.benchmark --ckpt runs/idiff_m1/step_84000.pt
PYTHONPATH=. $PY tests/test_windeval/test_metrics_v2.py     # metric calibration (14 checks)
```

Training gate on the cluster (same metrics, standalone, no jax):
`python src/eval/windeval/generators/infinite_diffusion/gate.py <ckpt> --ref <era5_train.zarr>`.
