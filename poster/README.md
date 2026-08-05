# Research poster

`poster.tex` builds a one-page, 30 in x 40 in portrait poster titled **Unbounded Wind
Fields for Stratospheric Balloon Simulation**.

```bash
make            # build poster.pdf
make preview    # render a low-resolution PNG
make open       # open the PDF on macOS
```

The poster uses the completed conditional base diffusion model and the spatial
InfiniteDiffusion wrapper. Coarse synoptic conditioning, arbitrary-length temporal
tiling, and balloon-agent training are intentionally listed as future work.

## Story

1. Balloon simulation motivates realistic, effectively unbounded wind worlds.
2. A bounded ERA5-conditioned denoiser generates four hourly wind volumes on 18 levels.
3. InfiniteDiffusion turns local windows into a reproducible, spatially unbounded field.
4. A reference-based benchmark shows that extent is solved, while the base denoiser
   remains over-smoothed and weak on vertical wind reversal.

The method column keeps two complementary equations: the InfiniteDiffusion overlap-and-
normalize update and the EDM denoising objective. The dependency-pyramid image is
reproduced from Figure 2 of the InfiniteDiffusion paper as a temporary explanatory visual.
A three-part guarantee strip summarizes spatial extent, fixed-window random access, and
seed consistency.

The results table reports the four-year, 250k-step checkpoint — `runs/idiff_m2cond_4yr/
step_250000.pt` — selected because the benchmark board identifies it as the best overall
checkpoint before later training plateaued or regressed. Checkpoints are not in git (`runs/`
is ignored; ~671 MB each); ask Shaurya for a copy or pull them from the Kahan cluster. The PSD and January eye-test figures show the earlier, explicitly
labelled one-year churned checkpoints; the captions keep that distinction visible.

The table reports component-wise conditional W1 across vertical levels and fixed
(location, month, hour) conditions. For the four-year 250k checkpoint:

- conditional W1(u): floor 2.08, simplex 4.69, ours 3.00;
- conditional W1(v): floor 2.67, simplex 2.08, ours 4.04.

The metrics show a conditioning advantage over simplex for u, but not for v. The former
combined conditional W1, marginal W1 speed, and opposing-wind error are omitted from the
poster table to keep the comparison focused; they remain available in the benchmark.

## Figures

- `intro_era5.png`, `intro_simplex.png`, `intro_ble.png`, `intro_diff.png`: qualitative
  approach comparison at a common pressure level.
- `eye_test_2x2.png`: difficult January case; ERA5 above and 300k diffusion below, at
  +0 h and +3 h.
- `zoom_montage.png`: nested random-access queries from one seed.
- `psd_energy.png`: kinetic-energy spectra for reference, baselines, and churned model
  checkpoints.

All quantitative values come from `src/eval/windeval/benchmark_v2_report.md` and
`docs/conditional-base-changes.md`. If a checkpoint or sampler setting changes, update
the table label and figure captions as well as the values.
