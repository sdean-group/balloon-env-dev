# Wind Generator Benchmark Results

## Scope

This document compares only:

1. Held-out ERA5.
2. Independent direct diffusion tiles with no cross-tile consistency mechanism.
3. InfiniteDiffusion with T=1, T=2, and T=3.
4. BLE-VAE.

All conclusions below concern the original frozen conditional diffusion checkpoint. No
multiyear retrained checkpoint is included.

## Shared Spatial Protocol

- ERA5 reference: days 8-14 of January, April, July, and October 2023 at 00 and
  12 UTC.
- Conditional generators: 56 matched conditions and two seeds, giving 112 samples per
  method.
- BLE-VAE: 112 independent latent samples. The benchmarked decoder is unconditional
  with respect to physical location, date, season, and hour, so it has no
  condition-matched score.
- Common evaluation region: 16x16 cells centered near San Francisco at 50 km spacing.
- Vertical range: 60-130 hPa using the same interpolation for ERA5 and every conditional
  method.
- Scored time: frame 0 of every four-frame generated block.
- ERA5 floor: discrepancy between independent held-out ERA5 subsets under the same
  metric. It is not a perfect-score row.

## Spatial Results

Lower is better for every reported metric.

| Metric | ERA5 self-split floor | Independent tiles | InfiniteDiffusion T=1 | InfiniteDiffusion T=2 | InfiniteDiffusion T=3 | BLE-VAE |
|---|---:|---:|---:|---:|---:|---:|
| SR_E | **0.2294** | 1.4713 | **1.3186** | 1.6400 | 1.7589 | 2.2428 |
| SR_div | **0.2615** | 1.7407 | **1.5738** | 1.8793 | 2.0306 | 4.5233 |
| SR_vort | **0.1790** | 0.9928 | **0.8048** | 1.1082 | 1.2037 | 1.6154 |
| L_eff (km) | **100.0000** | 800.0000 | **266.6667** | 800.0000 | 800.0000 | 800.0000 |
| W1 u (m/s) | **2.3560** | **3.7453** | 4.3950 | 4.4838 | 4.5663 | 13.0459 |
| W1 v (m/s) | **1.2391** | 4.5923 | 4.7755 | 4.8474 | 4.9197 | **3.9746** |
| Tail error 1% (m/s) | **3.2152** | 9.9525 | 9.9422 | 10.0725 | 10.2144 | **9.3268** |
| Tail error 0.1% (m/s) | **3.3188** | 12.5066 | 12.4805 | 12.6779 | 12.8194 | **10.6333** |
| Conditional W1 (m/s) | **3.3060** | **4.3383** | 4.7238 | 4.7975 | 4.8668 | N/A |

Bold values identify the ERA5 floor and the best generated method for each metric.

## What Each Spatial Metric Measures

| Metric | Interpretation |
|---|---|
| SR_E | Error in the spatial kinetic-energy spectrum |
| SR_div | Error in the horizontal-divergence spectrum |
| SR_vort | Error in the horizontal-vorticity spectrum |
| L_eff | Smallest wavelength retained before generated spectral energy falls below ERA5 |
| W1 u, W1 v | Wasserstein distance between generated and ERA5 wind-component distributions |
| Tail errors | Absolute error in extreme wind-speed quantiles |
| Conditional W1 | Distribution error when location and time conditions are matched |

## Direct Tiles Versus InfiniteDiffusion

The values below are InfiniteDiffusion minus the independent-tile result. Negative is an
improvement; positive is a regression.

| Metric | T=1 minus direct | T=2 minus direct | T=3 minus direct |
|---|---:|---:|---:|
| SR_E | **-0.1527** | +0.1687 | +0.2876 |
| SR_div | **-0.1669** | +0.1386 | +0.2899 |
| SR_vort | **-0.1880** | +0.1154 | +0.2109 |
| W1 u | +0.6497 | +0.7385 | +0.8210 |
| W1 v | +0.1832 | +0.2551 | +0.3274 |
| Tail error 1% | -0.0103 | +0.1200 | +0.2619 |
| Tail error 0.1% | -0.0261 | +0.1713 | +0.3128 |
| Conditional W1 | +0.3855 | +0.4592 | +0.5285 |

T=1 improves spatial spectral structure relative to independent tiles, particularly
energy, divergence, vorticity, and effective resolution. It worsens the bulk
distribution and condition-matched W1 scores. T=2 and T=3 are worse than independent
tiles on nearly every listed metric.

## InfiniteDiffusion Depth Comparison

The values below are changes relative to T=1. Positive is worse.

| Metric | T=2 minus T=1 | T=3 minus T=1 |
|---|---:|---:|
| SR_E | +0.3214 | +0.4403 |
| SR_div | +0.3055 | +0.4568 |
| SR_vort | +0.3034 | +0.3989 |
| W1 u | +0.0888 | +0.1713 |
| W1 v | +0.0719 | +0.1442 |
| Tail error 1% | +0.1303 | +0.2722 |
| Tail error 0.1% | +0.1974 | +0.3389 |
| Conditional W1 | +0.0737 | +0.1430 |

Under this checkpoint and protocol, increasing T consistently degrades measured spatial
quality. T=1 is the strongest InfiniteDiffusion setting.

## InfiniteDiffusion Versus BLE-VAE

| Comparison | Result |
|---|---|
| Spatial energy spectrum | Every InfiniteDiffusion depth beats BLE-VAE |
| Divergence spectrum | Every InfiniteDiffusion depth substantially beats BLE-VAE |
| Vorticity spectrum | Every InfiniteDiffusion depth beats BLE-VAE |
| Effective resolution | T=1 beats BLE-VAE; T=2 and T=3 tie its reported value |
| u distribution | Every InfiniteDiffusion depth substantially beats BLE-VAE |
| v distribution | BLE-VAE beats every InfiniteDiffusion depth |
| Extreme-wind tails | BLE-VAE beats every InfiniteDiffusion depth |
| Conditional distribution | Not comparable because BLE-VAE is unconditional |

InfiniteDiffusion is better overall for spatial structure and the zonal-wind
distribution. BLE-VAE is better on the meridional marginal and extreme quantiles.
Therefore, the accurate statement is not that InfiniteDiffusion wins every metric; it
wins the principal spatial-physics metrics while BLE-VAE retains specific distributional
advantages.

## Current Conclusions

1. Every generated method remains substantially separated from the ERA5 floor.
2. Independent tiles outperform InfiniteDiffusion on marginal and conditional
   distribution metrics.
3. InfiniteDiffusion T=1 improves spatial spectral structure enough to outperform both
   independent tiles and BLE-VAE on SR_E, SR_div, SR_vort, and effective resolution.
4. Increasing T from 1 to 2 or 3 does not improve spatial realism under this protocol.
5. The gap between direct tiles and ERA5 shows that synchronization is not the only
   problem; the frozen base denoiser is itself a major quality limitation.
6. Because architecture changes move metrics in different directions, conclusions must
   report spectra, distributions, and extremes separately.

## Temporal Benchmark

The temporal benchmark is still running and no temporal values are available yet. The
completed report will compare:

- ERA5 self-split floor.
- Independent four-hour diffusion tiles concatenated without consistency.
- InfiniteDiffusion T=1.
- InfiniteDiffusion T=2.
- InfiniteDiffusion T=3.
- BLE-VAE as N/A because its decoder slices have no validated hourly spacing.

It will report:

1. Temporal spectral residual, SR_time.
2. Passive-agent dispersion log-MSD RMSE.
3. Final trajectory spread ratio.
4. Mean adjacent hourly wind change.
5. Temporal seam-jump ratio.

Expected output:

```text
/share/dean/$USER/balloon-research/outputs/all_methods_temporal.md
```

The temporal table should be inserted here only after the benchmark completes. Empty or
fabricated values must not be used.

## Required Follow-Up

After temporal evaluation, the remaining required comparison is a checkpoint ablation:

| Required run | Purpose |
|---|---|
| Original checkpoint: direct tiles versus T=1 | Existing base-versus-tiling result |
| Multiyear checkpoint: direct tiles versus T=1 | Determine whether improved training data closes the ERA5 gap |

The same spatial and temporal metrics must be used for both checkpoints. This separates
improvements caused by base-model training from improvements caused by
InfiniteDiffusion.
