# Calibration Benchmark — Axis-1 OBJECTIVE quality scoreboard

Objective scores vs physical ideals (no distance-to-ERA5). ERA5 is a **peer row**, not the answer. Scores ∈ [0,1], higher = more wind-like.

**Empirical finding (this resolution):** the robust reference-free discriminator of wind-like structure is **intermittency** (non-Gaussian velocity increments) — real wind is leptokurtic, phase-randomization Gaussianizes toward 3. The spectrum slope is *necessary but fooled* by phase-shuffle. **Helmholtz/vort-div are weak here** (rotational character is mostly spectrum-encoded, so phase-shuffle keeps it) and the FFT decomposition is biased on a bounded box — so they are caveated diagnostics, excluded from the composite.

**Raw physical values**

| Metric (ideal) | era5_real (peer) | phase_shuffle | white_noise |
|---|---|---|---|
| spectrum slope (≈ −3 (steep)) | -3.92 | -3.92 | 0.0507 |
| Helmholtz rot. frac (→ 1 (rotational)) | 0.45 | 0.491 | 0.488 |
| vort/div ratio (> 1) | 0.766 | 0.972 | 0.964 |
| increment kurtosis (> 3 (intermittent)) | 5.56 | 3.39 | 3 |

**Objective scores ∈ [0,1]**

| Score | era5_real (peer) | phase_shuffle | white_noise |
|---|---|---|---|
| score: spectrum | 1.000 | 1.000 | 0.000 |
| score: Helmholtz | 0.000 | 0.000 | 0.000 |
| score: intermittency | 0.852 | 0.130 | 0.000 |
| COMPOSITE | **0.926** | **0.565** | **0.000** |

**Relative diagnostic** (speed dist. vs ERA5 peer-distribution — weak separator, kept as a sanity check)

| Metric | era5_real (peer) | phase_shuffle | white_noise |
|---|---|---|---|
| speed Wasserstein (m/s) | 0 | 0.299 | 0.292 |

## Verdict (automated checks)
- ✅ ranking: real (0.926) > phase-shuffle (0.565) > noise (0.000)
- ✅ spectrum FOOLED by phase-shuffle (real 1.00 ≈ shuffle 1.00) — necessary but insufficient
- ✅ intermittency is THE discriminator: real kurtosis 5.56 >> shuffle 3.39 ≈ noise 3.00 (~3)
- ✅ FINDING: Helmholtz NOT usable here (real 0.45 ≈ shuffle 0.49, Δ=0.04). Decomposition math is validated on analytic vortex/source (tests/test_helmholtz) but real structure fills this small/coarse box — needs a larger domain or a bounded-domain Poisson solver. Excluded from composite.
- ✅ white noise fails the composite (score 0.000) — lower bound confirmed