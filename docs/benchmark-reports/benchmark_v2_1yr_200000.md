# Benchmark v2 — reference-based metric suite

Raw values only (units per metric; direction in parentheses). Reference = held-out ERA5, days 8–14 of Jan/Apr/Jul/Oct 2023 (zero overlap with training dates); `self-split floor` = one disjoint half of the held-out set vs the other — read every row against it. N/A = metric not applicable (missing capability/levels), not a failure. Design + calibration: `docs/benchmark-v2-changes.md`.

## Physical consistency — spatial

| Metric (see METRIC_INFO) | self-split floor | phase shuffle | white noise | simplex noise | helmholtz gp | ble_vae | idiff trained | idiff m2cond |
|---|---|---|---|---|---|---|---|---|
| SR_E (lower) | 0.25 | 5.76 | 8.96 | 0.75 | 3.19 | 2.73 | 2.09 | 1.64 |
| SR_div (lower) | 0.25 | 5.47 | 8.98 | 0.87 | 3.24 | 5.23 | 2.14 | 1.63 |
| SR_vort (lower) | 0.25 | 5.85 | 8.99 | 0.72 | 3.14 | 2.16 | 2.10 | 1.67 |
| L_eff (km) (lower) | 56.12† | 56.12† | 3367.43 | 56.12† | 93.54 | 841.86 | 673.49 | 420.93 |
| W1 shear u ((m/s)/km) (lower) | 0.20 | 39.63 | 18.44 | 0.31 | 0.41 | 3.62 | 1.70 | 1.08 |
| W1 shear v ((m/s)/km) (lower) | 0.10 | 18.11 | 15.33 | 0.97 | 0.64 | 0.61 | 1.62 | 1.11 |

† = never dropped below the 0.5 energy-ratio threshold: resolved over the whole compared range (the value shown is the finest compared wavelength).

## Data distribution

| Metric (see METRIC_INFO) | self-split floor | phase shuffle | white noise | simplex noise | helmholtz gp | ble_vae | idiff trained | idiff m2cond |
|---|---|---|---|---|---|---|---|---|
| W1 u (m/s) (lower) | 1.07 | 8.78 | 0.53 | 1.12 | 0.94 | 17.92 | 4.26 | 4.01 |
| W1 v (m/s) (lower) | 0.65 | 2.59 | 0.38 | 0.40 | 0.33 | 4.89 | 3.00 | 3.36 |
| tail err 1% (m/s) (lower) | 2.01 | 5.65 | 1.53 | 1.73 | 1.31 | 15.36 | 11.52 | 10.56 |
| tail err 0.1% (m/s) (lower) | 2.92 | 5.97 | 2.59 | 2.34 | 1.42 | 18.94 | 15.59 | 13.36 |
| W1 speed (m/s) (lower) | 1.11 | 1.21 | 1.14 | 1.69 | 1.21 | 4.56 | 3.85 | 3.89 |
| jet speed err 99% (m/s) (lower) | 4.75 | 3.93 | 4.53 | 3.41 | 2.82 | 9.51 | 12.85 | 11.22 |
| jet speed err 99.9% (m/s) (lower) | 5.63 | 3.19 | 6.76 | 4.20 | 3.45 | 11.99 | 17.56 | 13.77 |
| W1 cond (m/s) (lower) | 2.38 | 7.40 | 3.13 | 3.38 | 3.22 | N/A | 3.87 | 3.58 |

`W1 cond` protocol: a condition = (fixed 64² center window, month, hour-of-day); reference pool = that hour on each held-out day 8–14, model pool = seeds at those same timestamps, averaged over the 8 conditions. The floor row uses the condition-matched split (days 8–10 vs 11–14). **Unconditional rows (noise baselines, `idiff trained`) are now scored on the same axis as the CLIMATOLOGICAL baseline** — their condition-independent pool vs each condition's reference, window- and sample-size-matched (14 frames/condition). That makes every row comparable and turns the old N/A ("not applicable") into the truthful reading, "cannot condition": the gap between a row and the floor is what conditioning has to buy. `W1 speed` and `jet speed err` are the |V| analogues of the u/v rows — moment-matching u and v does not automatically match wind SPEED or its jet-core tail.

## Structure & vertical realism

| Metric (see METRIC_INFO) | self-split floor | phase shuffle | white noise | simplex noise | helmholtz gp | ble_vae | idiff trained | idiff m2cond |
|---|---|---|---|---|---|---|---|---|
| opp-wind frac (≈ref) | 0.20 | 1.00 | 0.78 | 0.23 | 0.39 | 0.58 | 0.01 | 0.05 |
| opp-wind err (lower) | 0.00 | 0.80 | 0.58 | 0.02 | 0.19 | 0.37 | 0.19 | 0.15 |
| jet area ratio (≈1*) | 1.42 | 0.77 | 0.04 | 0.65 | 0.08 | 0.68 | 0.36 | 0.64 |
| jet elong ratio (≈1) | 1.17 | 1.10 | 0.87 | 0.85 | 0.70 | 1.76 | 0.93 | 0.95 |

Every other row in this suite is a population statistic — 1-point marginals and the 2-point covariance (the PSD rows) — which fitted noise reproduces by construction. These four look at what a matched spectrum does NOT fix. `opp-wind frac` = fraction of vertical columns holding a pair of levels whose winds oppose by >90° with both ≥5 m/s; **ERA5 reference = 0.201** on the same window (floor = the self-split row's `opp-wind err`). Vertical decoupling is the physical basis of balloon station-keeping and is the stated reason NRL rejected simplex noise for RL-HAB's realized field (arXiv 2502.05014); `shear.py` cannot see it, since it pools ADJACENT-level differences per component and never forms an angle between wind vectors. Jet rows threshold each slice at ITS OWN per-level 95th |V| percentile (so coverage is 5% for every row and only SHAPE is compared) and report mean component area / elongation as pred/ref ratios. All structural metrics use a common centered 64² box — a 64²-vs-121² mismatch inverts the decoupling verdict. N/A here means the row lacks the reference's 18-level stack (ble_vae) or has no components above the 8-px size floor (no coherent features at all). **Reliability:** `opp-wind` is the sharp one (floor 0.00; phase-shuffle 0.80 brackets the far end). `jet elong ratio` is a tight estimator (±0.05 splitting one weather period against itself) against a ~±0.17 synoptic floor. **`jet area ratio` is PROVISIONAL — do not score it:** its 1.42 floor is real synoptic variability (days 11–14 hold ~2× as many, smaller components as days 8–10), not estimator noise, and a median statistic does not fix it; making the geometry condition-matched the way W1_cond is would.

## Physical consistency — temporal

| Metric (see METRIC_INFO) | self-split floor | phase shuffle | white noise | simplex noise | helmholtz gp | ble_vae | idiff trained | idiff m2cond |
|---|---|---|---|---|---|---|---|---|
| SR_time (lower) | 0.28 | 3.16 | 2.40 | 1.19 | 5.56 | N/A | N/A | N/A |
| disp log-MSD RMSE (lower) | 0.18 | 1.91 | 0.39 | 0.70 | 0.62 | N/A | N/A | N/A |
| final spread ratio (≈1) | 1.04 | 0.40 | 0.38 | 0.53 | 0.46 | N/A | N/A | N/A |

Caveats: `ble_vae` is the SF box at 0.45° with 10 arbitrary levels — its distribution rows partly measure climate mismatch; its levels (a PRESSURE coordinate, hPa) are paired to the reference by nearest pressure — every pair within 4.5 hPa. `white noise` is at-floor on marginal W1 by construction (moment-matched) — read it on the spectral rows; `phase shuffle` covers the distribution rows. `simplex noise` / `helmholtz gp` are structured-noise baselines with every knob fit on half A (per-level moments, level/time correlations, div/vort split, horizontal scale by SR_E search — baselines.py): each row is the best its noise family can do, so a trained model must beat them everywhere to claim it learned weather rather than smooth statistics. Their marginals are Gaussian-ish by construction (near-floor W1, thin tails).

## Tiling penalty (multi-tile − single-tile; 0 = seamless)

| Metric | idiff trained |
|---|---|
| L_eff (km) | +0.000 |
| SR_E | -0.320 |
| SR_div | -0.275 |
| SR_vort | -0.353 |
| W1 shear u ((m/s)/km) | -0.206 |
| W1 shear v ((m/s)/km) | +0.027 |
| W1 speed (m/s) | -0.811 |
| W1 u (m/s) | -0.883 |
| W1 v (m/s) | -0.329 |
| jet area ratio | -0.018 |
| jet elong ratio | +0.117 |
| jet speed err 99% (m/s) | -1.970 |
| jet speed err 99.9% (m/s) | -2.246 |
| opp-wind err | +0.048 |
| opp-wind frac | -0.048 |
| tail err 0.1% (m/s) | -1.642 |
| tail err 1% (m/s) | -1.067 |

## Figures

![psd_triptych.png](../../../docs/figures/benchmark_v2/psd_triptych.png)
![marginal_u.png](../../../docs/figures/benchmark_v2/marginal_u.png)
![temporal.png](../../../docs/figures/benchmark_v2/temporal.png)
