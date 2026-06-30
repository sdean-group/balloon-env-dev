# M0 Leaderboard — InfiniteDiffusion on both axes

An *infinite* generator running through the benchmark. Axis-1 = field quality (reference-free, grid-independent); Axis-2 = the procedure claims only the unbounded class makes. Scores ∈ [0,1], higher = better; **N/A** = capability not declared (not a failure).

## Headline — two axes

| Axis | era5_real (peer) | ble_vae | infinite_diffusion_toy | phase_shuffle | white_noise |
|---|---|---|---|---|---|
| Axis-1: spatial COMPOSITE | **0.926** | **0.500** | **0.477** | **0.565** | **0.000** |
| Axis-2: PROC COMPOSITE | N/A | N/A | **0.974** | N/A | N/A |

## Axis-2 detail — the procedure (only `unbounded`/`tiled`/`random_access` gens)

| Procedure metric | era5_real (peer) | ble_vae | infinite_diffusion_toy | phase_shuffle | white_noise |
|---|---|---|---|---|---|
| seam discontinuity (score; 1=seamless) | N/A | N/A | 0.969 | N/A | N/A |
|   └ seam |div| excess (1=ideal) | N/A | N/A | 1.03 | N/A | N/A |
| revisit determinism (score; 1=exact) | N/A | N/A | 1.000 | N/A | N/A |
|   └ revisit max|Δ| | N/A | N/A | 0.0e+00 | N/A | N/A |
| budget / O(1) (score; far≈near) | N/A | N/A | 0.937 | N/A | N/A |
|   └ latency p50 (ms) | N/A | N/A | 2.2 | N/A | N/A |
|   └ budget far/near ratio | N/A | N/A | 1.06 | N/A | N/A |
| extent drift (score; 1=flat) | N/A | N/A | 0.992 | N/A | N/A |
|   └ drift slope / octave | N/A | N/A | -0.002 | N/A | N/A |

## Axis-1 detail

| Metric | era5_real (peer) | ble_vae | infinite_diffusion_toy | phase_shuffle | white_noise |
|---|---|---|---|---|---|
| spectrum slope (≈ −3) | -3.92 | 1.67 | -2.52 | -3.92 | 0.05 |
| increment kurtosis (> 3) | 5.56 | 13.90 | 3.35 | 3.39 | 3.00 |
| vort/div ratio (> 1) | 0.77 | 59.49 | 3.76 | 0.97 | 0.96 |

## Verdict (automated checks)
- ✅ MACHINERY: infinite gen passes Axis-2 (PROC COMPOSITE 0.97) — seamless / seed-consistent / O(1) / no extent drift
- ✅   seam 0.97, revisit 1.00, budget 0.94, extent 0.99
- ✅ bounded/peer generators are Axis-2 N/A (not scored as losses) — InfiniteDiffusion is the first to exercise the procedure axis
- ✅ Axis-1 sanity: ERA5 ceiling (0.93) > white noise (0.00)
- ✅ EXPECTED: toy is mid-pack on Axis-1 (0.48) — field quality is the *denoiser's* job (Phase 2); the toy only proves the machinery