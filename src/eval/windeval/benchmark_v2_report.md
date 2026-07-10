# Benchmark v2 — reference-based metric suite

Raw values only (units per metric; direction in parentheses). Reference = held-out ERA5, days 8–14 of Jan/Apr/Jul/Oct 2023 (zero overlap with training dates); `self-split floor` = one disjoint half of the held-out set vs the other — read every row against it. N/A = metric not applicable (missing capability/levels), not a failure. Design + calibration: `docs/benchmark-v2-changes.md`.

## Physical consistency — spatial

| Metric (see METRIC_INFO) | self-split floor | phase shuffle | white noise | ble_vae | idiff trained |
|---|---|---|---|---|---|
| SR_E (lower) | 0.25 | 5.76 | 8.96 | 2.72 | 2.09 |
| SR_div (lower) | 0.25 | 5.47 | 8.98 | 5.21 | 2.14 |
| SR_vort (lower) | 0.25 | 5.85 | 8.99 | 2.15 | 2.10 |
| L_eff (km) (lower) | 56.12† | 56.12† | 3367.43 | 841.86 | 673.49 |
| W1 shear u ((m/s)/km) (lower) | 0.20 | 39.63 | 18.44 | N/A | 1.70 |
| W1 shear v ((m/s)/km) (lower) | 0.10 | 18.11 | 15.33 | N/A | 1.62 |

† = never dropped below the 0.5 energy-ratio threshold: resolved over the whole compared range (the value shown is the finest compared wavelength).

## Data distribution

| Metric (see METRIC_INFO) | self-split floor | phase shuffle | white noise | ble_vae | idiff trained |
|---|---|---|---|---|---|
| W1 u (m/s) (lower) | 1.07 | 8.78 | 0.53 | 13.65 | 4.26 |
| W1 v (m/s) (lower) | 0.65 | 2.59 | 0.38 | 3.48 | 3.00 |
| tail err 1% (m/s) (lower) | 2.01 | 5.65 | 1.53 | 10.74 | 11.52 |
| tail err 0.1% (m/s) (lower) | 2.92 | 5.97 | 2.59 | 13.22 | 15.59 |
| W1 cond (m/s) (lower) | N/A | N/A | N/A | N/A | 3.39 |

`W1 cond` samples N=8 seed-crops under the single (unconditional) condition — see changes doc; real per-condition averaging starts when the conditioning layer lands.

## Physical consistency — temporal

| Metric (see METRIC_INFO) | self-split floor | phase shuffle | white noise | ble_vae | idiff trained |
|---|---|---|---|---|---|
| SR_time (lower) | 0.28 | 3.16 | 2.40 | N/A | N/A |
| disp log-MSD RMSE (lower) | 0.18 | 1.91 | 0.39 | N/A | N/A |
| final spread ratio (≈1) | 1.04 | 0.40 | 0.38 | N/A | N/A |

Caveats: `ble_vae` is the SF box at 0.45° with 10 arbitrary levels — its distribution rows partly measure climate mismatch, and level-indexed comparisons pair the first 10 reference levels. `white noise` is at-floor on marginal W1 by construction (moment-matched) — read it on the spectral rows; `phase shuffle` covers the distribution rows.

## Tiling penalty (multi-tile − single-tile; 0 = seamless)

| Metric | idiff trained |
|---|---|
| L_eff (km) | +0.000 |
| SR_E | -0.320 |
| SR_div | -0.275 |
| SR_vort | -0.353 |
| W1 shear u ((m/s)/km) | -0.206 |
| W1 shear v ((m/s)/km) | +0.027 |
| W1 u (m/s) | -0.883 |
| W1 v (m/s) | -0.329 |
| tail err 0.1% (m/s) | -1.642 |
| tail err 1% (m/s) | -1.067 |

## Figures

![psd_triptych.png](../../../docs/figures/benchmark_v2/psd_triptych.png)
![marginal_u.png](../../../docs/figures/benchmark_v2/marginal_u.png)
![temporal.png](../../../docs/figures/benchmark_v2/temporal.png)
