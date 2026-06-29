# BLE-VAE Generator Scorecard (vs ERA5 peer + anchors)

Objective scores ∈ [0,1] (higher = more wind-like). BLE-VAE = mean of 6 samples. Grid 21×21 @50km, 10 pressure levels, 9 time slices.

| Score | era5_real (peer) | ble_vae (n=6) | phase_shuffle | white_noise |
|---|---|---|---|---|
| **spatial composite** | 0.926 | 0.500 | 0.565 | 0.000 |
| **vertical coherence** | 0.805 | 0.793 | 0.008 | 0.000 |
| **temporal persistence** | 0.924 | 0.941 | 0.008 | 0.000 |

**Raw diagnostics**

| Metric | era5_real (peer) | ble_vae (n=6) | phase_shuffle | white_noise |
|---|---|---|---|---|
| spectrum slope (≈−3) | -3.92 | 1.30 | -3.92 | 0.05 |
| increment kurtosis (>3) | 5.56 | 14.57 | 3.39 | 3.00 |
| Helmholtz rot. frac | 0.45 | 0.78 | 0.49 | 0.49 |
| vertical coherence (raw) | 0.81 | 0.79 | 0.01 | -0.00 |
| temporal persistence (raw) | 0.92 | 0.94 | 0.01 | -0.00 |

## Reading

**Robust wins (dimensionless, grid-independent):**
- **Vertical coherence 0.79 vs ERA5 0.81** and **temporal persistence 0.94 vs 0.92** — the VAE genuinely learned coherent 4D structure; it nearly matches real ERA5 on the control-relevant axes.
- **Helmholtz 0.78 vs ERA5 0.45** — the divergence-free-by-construction design is detected (rotational by design).

**Spatial anomalies (real, but confounded — read with care):**
- **Spectrum slope +1.30** (vs ERA5 -3.92): positive/blue, dragging the composite to 0.50. **Increment kurtosis 14.6** (higher than ERA5 5.6): spiky, not smooth.
- Both trace to the decoder differentiating a 7×7 Ψ upsampled to 23×23 — piecewise-linear resize → high-frequency artifacts at segment boundaries (blue spectrum + heavy tails).
- **CONFOUND:** BLE's 21×21 grid makes PSD-slope estimation unreliable, and spectrum comparison across different grids isn't clean. Coherence metrics are dimensionless and unaffected; spectrum/kurtosis need a common-grid resample before strong claims.

## Fair spectral comparison (common 16×16 @ 50 km grid)

Both regridded to the SAME coarse grid inside both domains (ERA5 anti-aliased 28→50 km; BLE near-native). Now the spectrum is measured over the same scale range, so slope/kurtosis are comparable.

| Metric | ERA5 native | ERA5 common | BLE native | BLE common |
|---|---|---|---|---|
| spectrum slope | -3.92 | 2.31 | 1.30 | 1.87 |
| increment kurtosis | 5.56 | 4.87 | 14.57 | 3.80 |

**Verdict:**
- ⚠️ **Spectrum SLOPE is unreliable at this domain size.** ERA5's *own* slope swings -3.92→2.31 under regridding — a ~700 km box with ≤~16 points/side has too few Fourier modes for a stable slope. **Slope should be dropped at this domain; it needs a much larger box.** The resampling did its job: it exposed that the limit here is the *domain*, not just the cross-grid mismatch.
- On the FAIR grid BLE kurtosis 3.80 < ERA5 4.87: BLE is mildly **over-smooth** (under-intermittent). Its native kurtosis 14.6 was a grid-scale resize ARTIFACT the fair regrid removes — so over-smooth, not spiky, is the correct read.
- **Net:** trust the grid-independent verdict — strong vertical/temporal coherence, divergence-free, mildly over-smooth. Spectral *slope* is not usable at SF-box scale.