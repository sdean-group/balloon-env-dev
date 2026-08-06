# Synchronized Diffusion Benchmarks

## Methods

All three methods use the frozen M2 conditional checkpoint and the same canonical chart
geometry as CFGD.

| Method | Synchronization rule | Default extra work |
|---|---|---|
| SyncTweedies adaptation | Average predicted clean fields every solver evaluation | 1x CFGD factor evaluations |
| Overlap-guided adaptation | Add an RMS-normalized gradient of clean-field overlap disagreement | 2x |
| Consensus-equilibrium adaptation | Two denoiser/consensus/dual-correction rounds | 2x |

The overlap-guided and consensus settings are untuned pilot values. Conclusions must be
reported as applying to these configurations, not to every possible hyperparameter.

## Shared protocol

Spatial generation uses all 112 existing seasonal conditions and seeds. Temporal
generation uses the same four 24-hour seasonal episodes as the existing temporal
benchmark. The final reports reuse the exact ERA5, direct base, InfiniteDiffusion
T=1/T=2/T=3, CFGD, and BLE-VAE artifacts.

Outputs:

```text
/share/dean/$USER/balloon-research/outputs/all_methods_plus_sync_shared.md
/share/dean/$USER/balloon-research/outputs/all_methods_plus_sync_temporal.md
```

## Submission

If the original temporal benchmark is still queued, pass both its ID and the ERA5
reference job ID. The extended table will then wait for the old and new artifacts:

```bash
TEMPORAL_REFERENCE_JOB=558916 \
TEMPORAL_BASELINES_JOB=558948 \
PYTHON="$HOME/envs/idiff-eval-titan/bin/python" \
CHECKPOINT="$HOME/wind-idiff-checkpoint-eval/idiff_m2cond_latest.pt" \
bash src/eval/windeval/generators/synchronized_diffusion/configs/submit_extended_benchmarks.sh
```

If both existing jobs have completed successfully, omit both job variables.
