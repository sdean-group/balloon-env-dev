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

## Story and column structure

1. The first column motivates global, coherent wind generation and then introduces the
   bounded 4D base model: its input, output, conditioning, architecture, training data,
   and a qualitative ERA5 comparison.
2. The second column is entirely about InfiniteDiffusion. Plain-language pseudocode
   replaces the dense update equation; the dependency pyramid, cached reuse, corrected
   64x64 zoom sequence, and naive-tiling comparison form one continuous explanation.
3. The third column defines the ERA5 reference/floor and each baseline, groups the full
   metric suite into four physical questions, reports a diagnostic subset, and explains
   the spectral residual visually.

Future work is distributed to the relevant column: coarse-field conditioning beside the
base model, temporal tiling beside InfiniteDiffusion, and temporal/agent evaluation beside
the metric suite. `DIAGRAM_PROMPTS.md` contains production briefs for artwork slots D01-D07.

The results table reports the four-year, 250k-step checkpoint — `runs/idiff_m2cond_4yr/
step_250000.pt` — selected because the benchmark board identifies it as the best overall
checkpoint before later training plateaued or regressed. Checkpoints are not in git (`runs/`
is ignored; ~671 MB each); ask Shaurya for a copy or pull them from the Kahan cluster.

The compact table deliberately selects metrics whose rankings expose different failure
modes: energy/divergence spectra, combined condition-matched W1, 0.1% tails, opposing
winds across height, and jet elongation. The surrounding taxonomy lists the broader suite.

## Artwork slots

- D01: globe-scale contrast between ERA5, bounded BLE-VAE, and lazy global queries.
- D02: base-model input, conditioning, architecture, and 18-level x four-hour output.
- D03: latest-checkpoint ERA5-versus-base-model qualitative comparison.
- D04: dependency pyramid with explicit cached reuse for a second query.
- D05: corrected zoom sequence centered on the 64x64 denoiser window.
- D06: bounded tile versus naive averaging versus InfiniteDiffusion.
- D07: simplified PSD plot with a visual definition of spectral residual.

The older PNGs remain in `figures/` as source material, but the revised layout does not
embed them because several are obsolete or too cluttered for the new story. Follow the
briefs in `DIAGRAM_PROMPTS.md` and replace one slot at a time without changing its bounds.

All quantitative values come from `src/eval/windeval/benchmark_v2_report.md` and
`docs/conditional-base-changes.md`. If a checkpoint or sampler setting changes, update
the table label and figure captions as well as the values.
