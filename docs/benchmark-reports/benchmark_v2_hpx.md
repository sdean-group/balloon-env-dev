# Benchmark v2 with the HEALPix two-stage model

Same metric code, same held-out NE Pacific reference (days 8-14 of Jan/Apr/Jul/Oct 2023), same 64x64 window and (month, day, hour) conditions as the poster. New rows are the global two-stage model regridded onto the window. Lower is better unless stated.

| metric | self-split floor | simplex noise | ble_vae | summer m2cond (poster) | coarse upsampled | summer m2coarse2 | ours: Stage 1 -> Stage 2 | ours: Stage 2 | ERA5 coarse | ERA5 through our grid |
|---|---|---|---|---|---|---|---|---|---|
| W1(u | c) (m/s) | 2.08 | 4.69 | N/A | 3.05 | 0.25 | 0.04 | 2.40 | 0.04 | 0.12 |
| W1(v | c) (m/s) | 2.67 | 2.08 | N/A | 3.83 | 0.20 | 0.04 | 3.56 | 0.04 | 0.29 |
| W1(|V|) (m/s) | 1.11 | 1.69 | 4.56 | 2.89 | 0.19 | 0.11 | 0.76 | 0.05 | 0.08 |
| tail error 0.1% (m/s) | 2.92 | 2.34 | 18.94 | 13.87 | 5.91 | 5.05 | 6.60 | 4.93 | 4.79 |
| SR_E | 0.25 | 0.75 | 2.73 | 1.53 | 1.74 | 1.00 | 1.69 | 1.71 | 1.41 |
| SR_div | 0.25 | 0.87 | 5.23 | 1.54 | 1.70 | 1.04 | 1.75 | 1.77 | 1.47 |
| SR_vort | 0.25 | 0.72 | 2.16 | 1.53 | 1.82 | 0.98 | 1.68 | 1.70 | 1.40 |
| L_eff (km) | 56.12 | 56.12 | 842 | 374 | 481 | 198 | 210 | 198 | 125 |
| W1 shear u | 0.20 | 0.31 | 3.62 | 1.03 | 0.87 | 0.38 | 0.36 | 0.39 | 0.38 |
| W1 shear v | 0.10 | 0.97 | 0.61 | 1.06 | 0.71 | 0.11 | 0.40 | 0.10 | 0.12 |
| SR_time | 0.28 | 1.19 | N/A | N/A | N/A | N/A | 3.08 | 0.40 | 0.35 |
| final spread ratio (1 = ERA5) | 1.04 | 0.53 | N/A | N/A | N/A | N/A | 0.67 | 0.57 | 0.54 |
| opposing-wind fraction (ERA5 0.20) | 0.20 | 0.23 | 0.58 | 0.02 | 0.18 | 0.19 | 0.18 | 0.20 | 0.20 |
| jet speed error 99.9% (m/s) | 5.63 | 4.20 | 11.99 | 13.16 | 1.39 | 0.61 | 3.97 | 0.45 | 0.08 |
| trajectory dispersion RMSE | 0.18 | 0.70 | N/A | N/A | N/A | N/A | 0.17 | 0.05 | 0.05 |
