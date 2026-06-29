# Stage-2 Calibration Benchmark — spatial + vertical + temporal

Objective scores ∈ [0,1] (higher = more wind-like); ERA5 is a peer row. 24 hourly SF timesteps, model-level band 49-66.

**Headline scores**

| Score | era5_real (peer) | phase_shuffle | white_noise |
|---|---|---|---|
| spatial (spectrum+intermittency) | **0.926** | **0.565** | **0.000** |
| vertical coherence | **0.805** | **0.008** | **0.000** |
| temporal persistence | **0.924** | **0.008** | **0.000** |

**Vertical (control axis)**

| Metric | era5_real (peer) | phase_shuffle | white_noise |
|---|---|---|---|
| vertical coherence | 0.805 | 0.00801 | -0.000672 |
| shear mean (m/s/km) | 8.62 | 71.8 | 21 |

**Temporal / dynamics**

| Metric | era5_real (peer) | phase_shuffle | white_noise |
|---|---|---|---|
| temporal persistence | 0.924 | 0.00819 | -0.000319 |
| tendency (m/s/h) | 1.82 | 24.6 | 6.85 |
| drift slope/step (≈0 ok) | 9.68e-05 | -0.00248 | 5.01e-06 |

**Spatial detail**

| Metric | era5_real (peer) | phase_shuffle | white_noise |
|---|---|---|---|
| spectrum slope | -3.92 | -3.92 | 0.0507 |
| increment kurtosis | 5.56 | 3.39 | 3 |
| Helmholtz (diagnostic) | 0.45 | 0.491 | 0.488 |

## Verdict (automated checks)
- ✅ VERTICAL coherence CATCHES both anchors: real 0.81 vs shuffle 0.01 / noise 0.00 (per-level randomization decorrelates adjacent levels)
- ✅ TEMPORAL persistence CATCHES both anchors: real 0.92 vs shuffle 0.01 / noise 0.00
- ✅ tendency: real evolves slowly (1.82 m/s/h) vs incoherent anchors (24.6 / 6.9)
- ✅ real ERA5 does NOT drift (slope +0.0001/step ≈ 0) — machinery validated; drift bites on generator rollouts
- ✅ spatial spectrum STILL fooled by phase-shuffle (composite 0.57) — but vertical+temporal catch it: the full suite is robust