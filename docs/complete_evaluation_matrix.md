# Wind Generator Evaluation Matrix

## Objective

Determine whether wind-field quality is limited by the frozen base denoiser, by the
InfiniteDiffusion overlap algorithm, or by both. Every architecture comparison must hold
the checkpoint, conditions, seeds, grid, pressure levels, diffusion schedule, and ERA5
reference fixed.

## Methods

| Method | Role |
|---|---|
| ERA5 self-split floor | Expected discrepancy between two held-out ERA5 subsets |
| Direct base model | Measures the frozen denoiser without infinite-field synchronization |
| InfiniteDiffusion T=1 | Current primary infinite-field baseline |
| InfiniteDiffusion T=2 | Depth ablation |
| InfiniteDiffusion T=3 | Depth ablation |
| CFGD | Per-step shared-chart consensus |
| SyncTweedies adaptation | Per-step consensus of predicted clean fields |
| Overlap-guided adaptation | Separate diffusion paths with overlap-loss guidance |
| Consensus-equilibrium adaptation | Fixed-round denoiser and consensus corrections |
| BLE-VAE | Existing unconditional wind-generator baseline |

## Fixed Spatial Protocol

- Reference: held-out ARCO-ERA5 from days 8-14 of January, April, July, and
  October 2023 at 00 and 12 UTC.
- Generated set: 112 samples per conditional method: 56 conditions and two seeds.
- Comparison grid: common 16x16 crop at 50 km spacing.
- Vertical coordinates: 60-130 hPa, with the same interpolation for every method.
- Scored frame: frame 0 of each four-frame generated block.
- BLE-VAE: 112 independent latent samples; conditional W1 is not applicable.

## Table 1: Primary Spatial Quality

This is the principal architecture table. Lower is better for every metric.

| Metric | ERA5 floor | Direct | ID T=1 | ID T=2 | ID T=3 | CFGD | SyncTweedies | Overlap-guided | Consensus equilibrium | BLE-VAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SR_E | | | | | | | | | | |
| SR_div | | | | | | | | | | |
| SR_vort | | | | | | | | | | |
| L_eff (km) | | | | | | | | | | |
| W1 u (m/s) | | | | | | | | | | |
| W1 v (m/s) | | | | | | | | | | |
| Tail error 1% (m/s) | | | | | | | | | | |
| Tail error 0.1% (m/s) | | | | | | | | | | |
| Conditional W1 (m/s) | | | | | | | | | N/A |

Generated report:

```text
/share/dean/$USER/balloon-research/outputs/all_methods_plus_sync_shared.md
```

## Fixed Temporal Protocol

- Four 24-hour episodes beginning at 00 UTC on January 8, April 8, July 8, and
  October 8, 2023.
- ERA5 day 8 is the reference; day 9 is the independent seasonal floor.
- Same seed, 16x16 grid, pressure coordinates, and physical conditions for all
  conditional generators.
- Direct base concatenates six independently sampled four-hour blocks.
- Every infinite-field method generates the complete 24-hour query as one field.
- BLE-VAE is not scored because its decoder slices have no established physical hourly
  spacing.

## Table 2: Primary Temporal Quality

| Metric | ERA5 floor | Direct | ID T=1 | ID T=2 | ID T=3 | CFGD | SyncTweedies | Overlap-guided | Consensus equilibrium | BLE-VAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SR_time, lower | | | | | | | | | | N/A |
| Dispersion log-MSD RMSE, lower | | | | | | | | | | N/A |
| Final spread ratio, near 1 | | | | | | | | | | N/A |
| Mean adjacent change (m/s), match ERA5 | | | | | | | | | | N/A |
| Time-seam jump ratio, near 1 | | | | | | | | | | N/A |

Generated report:

```text
/share/dean/$USER/balloon-research/outputs/all_methods_plus_sync_temporal.md
```

## Table 3: Computational Cost

Report quality together with cost. Runtime must use the same GPU type or be normalized
by measured denoiser throughput.

| Method | Seconds/sample | Model-window evaluations | Charts or windows generated | Peak GPU memory | Relative work |
|---|---:|---:|---:|---:|---:|
| Direct | | | | | |
| ID T=1 | | | | | |
| ID T=2 | | | | | |
| ID T=3 | | | | | |
| CFGD | | | | | |
| SyncTweedies | | | | | 1x synchronized-chart work |
| Overlap-guided | | | | | 2x model evaluations plus backward pass |
| Consensus equilibrium | | | | | 2x synchronized-chart work |

Use median and interquartile range across samples. Do not compare raw wall time from
different GPU models without normalization.

## Table 4: InfiniteDiffusion Depth Ablation

This isolates the effect of recursive depth.

| Depth | Split steps | Spatial quality summary | Temporal quality summary | Model evaluations | Runtime |
|---:|---|---:|---:|---:|---:|
| 1 | none | | | | |
| 2 | 9 | | | | |
| 3 | 6, 12 | | | | |

Required conclusions:

1. Whether increasing T improves overlap continuity.
2. Whether increasing T degrades spectra, distributions, extremes, or temporal change.
3. Whether any quality change justifies the growth in model evaluations.

## Table 5: Base Model Versus Tiling

This table answers whether the principal limitation comes from the denoiser or the
infinite-field architecture.

| Comparison | SR_E change | SR_div change | SR_vort change | W1 change | Tail change | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Direct -> ID T=1 | | | | | | Cost of InfiniteDiffusion tiling |
| Direct -> CFGD | | | | | | Cost or benefit of chart consensus |
| Direct -> best synchronized method | | | | | | Best observed architecture effect |
| ERA5 floor -> Direct | | | | | | Base-denoiser quality gap |

If Direct is already far from ERA5 and architecture deltas are small, retraining the
base model is the priority. If Direct is good but infinite-field methods are poor,
synchronization is the priority.

## Table 6: Synchronization Rule Ablation

Compare methods with the same canonical chart geometry.

| Method | Synchronized quantity | Exact merge? | Extra gradient? | Consensus rounds | Spatial rank | Temporal rank | Cost rank |
|---|---|---|---|---:|---:|---:|---:|
| CFGD | EDM direction/shared chart state | Yes | No | 1 | | | |
| SyncTweedies | Predicted clean field | Yes | No | 1 | | | |
| Overlap-guided | No exact merge; overlap loss | No | Yes | 1 | | | |
| Consensus equilibrium | Clean proposal plus disagreement state | Yes | No | 2 | | | |

This determines whether quality depends on synchronizing noisy states, clean estimates,
soft gradients, or corrected multi-round consensus.

## Table 7: Robustness and Uncertainty

Aggregate means alone are insufficient for a final paper claim.

| Breakdown | Required output |
|---|---|
| Season | Metric values for January, April, July, and October |
| Pressure level | Metric values or distributions by level |
| Condition | Paired method differences at each matched timestamp |
| Seed | Variation across the two existing seeds |
| Confidence interval | Paired bootstrap 95% interval for each method minus ID T=1 |

The primary significance test is a paired bootstrap over matched conditions. A method is
an improvement only when its interval excludes zero in the favorable direction on the
preselected primary metrics.

## Table 8: Procedural Guarantees

| Method | Deterministic repeat | Order-independent query | Crop consistency | Seamless overlap | Bounded random-access work | Test |
|---|---|---|---|---|---|---|
| ID T=1 | | | | | | |
| ID T=2 | | | | | | |
| ID T=3 | | | | | | |
| CFGD | | | | | | |
| SyncTweedies | | | | | | |
| Overlap-guided | | | | | | |
| Consensus equilibrium | | | | | | |

Each entry must be backed by a mechanical test, not visual inspection. At minimum:

1. Repeating an identical query returns exactly identical values.
2. Querying A then B equals querying B then A.
3. A crop generated alone equals the same crop taken from a larger query.
4. Work for a fixed-size query does not depend on absolute coordinates or prior queries.

## Table 9: Retrained-Model Follow-Up

Run this only after a better multiyear base checkpoint exists.

| Checkpoint | Direct | ID T=1 | Best alternative | ERA5 floor |
|---|---:|---:|---:|---:|
| Original one-year checkpoint | | | | |
| Multiyear checkpoint | | | | |

Repeat the primary spatial and temporal metrics. This separates gains from better
training data from gains caused by the infinite-field architecture.

## Decision Rules

1. Select primary metrics before reading the new results: SR_E, SR_div, SR_vort,
   conditional W1, SR_time, and dispersion log-MSD RMSE.
2. Reject an architecture that improves seams only by over-smoothing temporal or spatial
   spectra.
3. Prefer the simplest method within uncertainty of the best method.
4. Report architecture quality and computational cost together.
5. Treat the current synchronized-method settings as pilots until guidance strength,
   consensus rounds, and relaxation have small held-out ablations.
