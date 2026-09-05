# Canonical Factor-Graph Diffusion

This package implements an experimental infinite-field sampler around the frozen
conditional space-time wind denoiser. It does not change or train model weights.

## Architecture

Each canonical chart contains a fixed overlapping-window factor graph. At every EDM
noise level, all window factors read from one shared chart state, predict local denoised
states, and contribute weighted ODE directions. The fused direction advances the shared
state through one global Heun update. Neighboring deterministic charts are combined by a
partition of unity and cached by integer `(time, y, x)` chart index.

The implementation combines three published ideas:

- **MultiDiffusion:** per-step weighted least-squares reconciliation of overlapping
  diffusion paths.
- **DiffCollage:** local diffusion models organized as parallel factors coupled through
  shared overlap variables.
- **InfiniteDiffusion:** coordinate-indexed noise, lazy evaluation, overlap accumulation,
  caching, seed consistency, and random access.

The canonical chart atlas, per-step EDM-direction consensus, and extension to conditional
four-dimensional wind are the experimental contribution in this package.

## Important Scope

This defines a new locally finite random field. It is not an approximation to a particular
InfiniteDiffusion output and it does not solve one globally connected infinite factor
graph. Long-range dependence is bounded by chart support.

## CLI

Run `evaluate.py` directly so it loads only the diffusion dependencies:

```bash
python src/eval/windeval/generators/canonical_factor_graph/evaluate.py \
  --checkpoint /path/to/idiff_m2cond_latest.pt \
  --device mps \
  --num-steps 1 \
  --core-size 64 \
  --halo-size 16 \
  --core-time 4 \
  --halo-time 0 \
  --query-size 16 \
  --query-frames 1 \
  --query-t0 0 \
  --query-y0 24 \
  --query-x0 24 \
  --window-batch-size 1 \
  --output-dir outputs/cfgd_mps_smoke
```

That smoke test evaluates four mutually overlapping spatial factors in one chart. It is
small enough to establish checkpoint/MPS compatibility but does not exercise cross-chart
atlas blending. The default geometry exercises both layers and is the research baseline.

The default full geometry uses a `64x64x2` core, `32x32x1` halo, and the checkpoint's
`64x64x4` windows. Increase `--window-batch-size` only after measuring MPS memory use.

For a matched-compute comparison against the existing Infinite Diffusion implementation,
run `benchmark_against_infinite.py`. It evaluates the same physical spacetime region across
multiple seeds, alternates method order, preserves every raw field, and reports paired
confidence intervals. The default five-seed MPS run uses 420 model forwards per method and
seed; it tests composition and consistency, not realism against ERA5.

`compare_multiseed_to_era5_npz.py` scores those saved samples against the small local
January 15 ARCO-ERA5 reference without loading xarray or downloading the global archive.
The exact overlap is three hours by `16x16` cells at all 18 model levels. This date overlaps
the checkpoint's training range, so it is a realism sanity check rather than held-out evidence.

## Held-Out Condition Protocol

`generate_condition_set.py` mirrors Infinite Diffusion's seasonal condition generator but
uses a bounded `16x16` query whose complete chart support stays inside the checkpoint's
training coordinate domain. The default geometry has three temporal charts and four
overlapping spatial factors per chart, for 420 model forwards with an 18-step sampler.

`benchmark_condition_sets.py` finds the exact spatial and condition intersection across
CFGD and Infinite Diffusion runs, crops ERA5 and every run to it, and applies the existing
held-out metric suite. Distribution and tail metrics are directly comparable. Spatial
spectral metrics on the `16x16` crop have limited wavelength range and must be labeled as
supporting diagnostics rather than the primary conclusion.

For a full `4x64x64` block aligned with the existing January ERA5 reference, use query
indices `--query-t0 2 --query-y0 32 --query-x0 32`. The resulting `wind.npz` can be
compared with the local reference using `compare_to_era5.py`; this comparison measures
quality, not similarity to an InfiniteDiffusion sample.
