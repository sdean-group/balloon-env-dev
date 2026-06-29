# Calibration Benchmark — Axis-1 OBJECTIVE quality scoreboard

Objective scores vs physical ideals (no distance-to-ERA5). ERA5 is a **peer row**, not the answer. Scores ∈ [0,1], higher = more wind-like.

**Empirical finding (this resolution):** the robust reference-free discriminator of wind-like structure is **intermittency** (non-Gaussian velocity increments) — real wind is leptokurtic, phase-randomization Gaussianizes toward 3. The spectrum slope is *necessary but fooled* by phase-shuffle. **Helmholtz/vort-div are weak here** (rotational character is mostly spectrum-encoded, so phase-shuffle keeps it) and the FFT decomposition is biased on a bounded box — so they are caveated diagnostics, excluded from the composite.

**Raw physical values**

| Metric (ideal) | era5_real (peer) | phase_shuffle | white_noise |
|---|---|---|---|
| spectrum slope (≈ −3 (steep)) | -3.95 | -3.95 | 0.0548 |
| Helmholtz rot. frac (→ 1 (rotational)) | 0.538 | 0.577 | 0.497 |
| vort/div ratio (> 1) | 0.964 | 1.18 | 0.99 |
| increment kurtosis (> 3 (intermittent)) | 5.72 | 3.38 | 3.03 |

**Objective scores ∈ [0,1]**

| Score | era5_real (peer) | phase_shuffle | white_noise |
|---|---|---|---|
| score: spectrum | 1.000 | 1.000 | 0.000 |
| score: Helmholtz | 0.077 | 0.155 | 0.000 |
| score: intermittency | 0.906 | 0.127 | 0.009 |
| COMPOSITE | **0.953** | **0.564** | **0.004** |

**Relative diagnostic** (speed dist. vs ERA5 peer-distribution — weak separator, kept as a sanity check)

| Metric | era5_real (peer) | phase_shuffle | white_noise |
|---|---|---|---|
| speed Wasserstein (m/s) | 0 | 0.325 | 0.363 |

## Verdict (automated checks)
- ✅ ranking: real (0.953) > phase-shuffle (0.564) > noise (0.004)
- ✅ spectrum FOOLED by phase-shuffle (real 1.00 ≈ shuffle 1.00) — necessary but insufficient
- ✅ intermittency is THE discriminator: real kurtosis 5.72 >> shuffle 3.38 ≈ noise 3.03 (~3)
- ✅ FINDING: Helmholtz WEAK here (real 0.54 vs shuffle 0.58, Δ=0.04) — rotational character is spectrum-encoded + bounded-box FFT bias; diagnostic only
- ✅ white noise fails the composite (score 0.004) — lower bound confirmed