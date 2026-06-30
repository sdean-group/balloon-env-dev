# M1 Leaderboard — TRAINED InfiniteDiffusion on both axes

Phase 3 swap: the trained window denoiser dropped into the *unmodified* machinery. Axis-1 = field quality (reference-free); Axis-2 = the procedure claims only the unbounded class makes. Scores ∈ [0,1], higher = better; **N/A** = capability not declared (not a failure).

Trained checkpoint: `runs/idiff_m1/step_84000.pt` · device `mps` · internal EDM steps 18.

## Headline — two axes

| Axis | era5_real (peer) | ble_vae | infinite_diffusion_trained | infinite_diffusion_toy | phase_shuffle | white_noise |
|---|---|---|---|---|---|---|
| Axis-1: spatial COMPOSITE | **0.926** | **0.500** | **0.660** | **0.477** | **0.565** | **0.000** |
| Axis-2: PROC COMPOSITE | N/A | N/A | **0.873** | **0.986** | N/A | N/A |

## Axis-1 detail (field quality — the model's job)

| Metric | era5_real (peer) | ble_vae | infinite_diffusion_trained | infinite_diffusion_toy | phase_shuffle | white_noise |
|---|---|---|---|---|---|---|
| spectrum slope (≈ −3) | -3.92 | 1.67 | -3.53 | -2.52 | -3.92 | 0.05 |
| increment kurtosis (> 3) | 5.56 | 13.90 | 3.96 | 3.35 | 3.39 | 3.00 |
| vort/div ratio (> 1) | 0.77 | 59.49 | 0.93 | 3.76 | 0.97 | 0.96 |
| score: spectrum | 1.000 | 0.000 | 1.000 | 0.838 | 1.000 | 0.000 |
| score: intermittency | 0.852 | 1.000 | 0.320 | 0.116 | 0.130 | 0.000 |

### Amplitude (scale — NOT in COMPOSITE; scored vs the ERA5 peer RMS)

The four metrics above are scale-invariant, so a too-calm field with the right structure slips past them. `amplitude rms` (RMS wind speed) catches that; `score: amplitude` = min(r, 1/r) vs the peer (1 = parity). The trained model targets its *training* climatology (lat 25–55°N, 4 seasonal weeks), genuinely calmer than this narrower era5_real peer (lat 33–42°N), so part of the deficit is climatology, not generation error:

> Decomposition for the trained model: peer rms 21.2, its training-climatology rms 14.8 (a perfect model would still score only **0.70** vs the peer — the unavoidable climatology offset), trained rms 10.2 → **0.69** vs its own climatology = the real generation defect (anomaly-variance under-dispersion).

| Metric | era5_real (peer) | ble_vae | infinite_diffusion_trained | infinite_diffusion_toy | phase_shuffle | white_noise |
|---|---|---|---|---|---|---|
| amplitude rms (m/s) | 21.16 | 11.08 | 10.15 | 14.77 | 21.16 | 21.17 |
| score: amplitude (vs peer) | 1.000 | 0.524 | 0.480 | 0.698 | 1.000 | 1.000 |

## Axis-2 detail — the procedure (machinery preserved under the swap?)

| Procedure metric | era5_real (peer) | ble_vae | infinite_diffusion_trained | infinite_diffusion_toy | phase_shuffle | white_noise |
|---|---|---|---|---|---|---|
| seam discontinuity (score; 1=seamless) | N/A | N/A | 0.872 | 0.969 | N/A | N/A |
|   └ seam |div| excess (1=ideal) | N/A | N/A | 1.13 | 1.03 | N/A | N/A |
| revisit determinism (score; 1=exact) | N/A | N/A | 1.000 | 1.000 | N/A | N/A |
|   └ revisit max|Δ| | N/A | N/A | 0.0e+00 | 0.0e+00 | N/A | N/A |
| budget / O(1) (score; far≈near) | N/A | N/A | 0.937 | 0.983 | N/A | N/A |
|   └ budget far/near ratio | N/A | N/A | 1.06 | 1.02 | N/A | N/A |
| extent drift (score; 1=flat) | N/A | N/A | 0.685 | 0.992 | N/A | N/A |
|   └ drift slope / octave | N/A | N/A | +0.063 | -0.002 | N/A | N/A |

## Verdict (automated checks)
- ✅ AXIS-1: trained (0.66) beats the toy (0.48) — the learned denoiser produces better wind structure than the analytic stand-in
- ✅   spectrum solved: score 1.00 (slope -3.53)
- ✅   trained sits between noise (0.00) and the ERA5 ceiling (0.93); residual gap is mostly intermittency (score 0.32)
- ✅ AXIS-2 PRESERVED: swapping a *learned* denoiser keeps the machinery claims (PROC COMPOSITE 0.87) — seamless / seed-consistent / O(1)
- ✅   seam 0.87, revisit 1.00, budget 0.94, extent 0.69
- ✅ bounded/peer generators remain Axis-2 N/A (the procedure axis is only for unbounded generators)