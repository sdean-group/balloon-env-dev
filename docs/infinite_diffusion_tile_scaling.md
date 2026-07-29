# InfiniteDiffusion tile-scaling experiment

## Question

How do spatial coherence, ERA5 realism, inference time, model evaluations, and GPU memory
change when the same physical output is represented by 4, 16, or 64 tile cores?

## Controlled geometry

The output is always the same 64x64 ERA5-aligned field. Model windows overlap by 50%.

| Core regions | Core grid | Model window | Stride | Final windows including halos |
|---:|---:|---:|---:|---:|
| 4 | 2x2 | 64x64 | 32 | 9 |
| 16 | 4x4 | 32x32 | 16 | 25 |
| 64 | 8x8 | 16x16 | 8 | 81 |

The distinction between core regions and final model windows is important. A 2x2 core
partition needs a 3x3 set of overlapping model windows; the additional windows are halo
work needed to blend the fixed output.

All rows use identical:

- location and timestamp conditions;
- two random seeds for each condition;
- coordinate-keyed noise;
- 18-step EDM sampler;
- InfiniteDiffusion depth and split schedule;
- 50% overlap;
- 64x64 scored output;
- native 0.25-degree ERA5 grid and levels 49-66.

The full sweep contains 112 samples per row: four seasons, seven held-out days, two UTC
hours, and two seeds.

## Measurements

### ERA5 quality

- SR_E, SR_div, SR_vort;
- effective resolution;
- W1 u and W1 v;
- 1% and 0.1% tail error;
- conditional W1.

### Coherence

- **Boundary jump ratio:** vector jump across core boundaries divided by the mean jump
  across all adjacent cells. One means a tile line is statistically ordinary.
- **Boundary squared-jump ratio:** the same comparison on squared jumps, making rare
  severe seams more visible.
- **Boundary direction gap:** global neighboring-vector cosine minus boundary cosine.
  Zero means direction changes at tile boundaries are ordinary.
- **Vector correlation length:** first distance where the mean two-point vector
  correlation falls below 0.5, compared with condition-matched ERA5.

Each boundary diagnostic is also computed on ERA5 at the same artificial grid lines.
This controls for real weather fronts that happen to cross a proposed tile boundary.

### Inference

- synchronized cold wall time per sample;
- p10, median, and p90 wall time;
- model-window calls;
- denoiser forward evaluations;
- peak allocated GPU memory;
- relative speed against the 64-core configuration.

## Two-stage interpretation

### Stage A: same-checkpoint pilot

Use the existing 64x64-trained checkpoint for all three rows. The network is fully
convolutional, so 32x32 and 16x16 inference is mechanically valid. This is the fastest
test of whether smaller windows are promising.

It is not a causal tile-size comparison: the smaller windows are outside the training
crop distribution. A loss of quality could come from tiling or from that distribution
shift.

### Stage B: crop-matched models

Repeat with models trained using crop sizes 64, 32, and 16. The supplied crop32 and
crop16 configurations keep the architecture, data, optimizer, batch size, update count,
conditioning, and seed fixed. Pass the resulting checkpoints as `CHECKPOINT_16` and
`CHECKPOINT_64` when submitting the sweep.

Keeping batch size and steps fixed preserves the training recipe but does not equalize
the number of wind pixels seen. An additional pixel-budget control would require 4x as
many crop32 examples and 16x as many crop16 examples. That is a separate experiment and
must not be silently mixed into the primary table.

## Submit the same-checkpoint pilot

```bash
cd ~/balloon-env-dev-code

PYTHON="$HOME/envs/idiff-eval-titan/bin/python" \
CHECKPOINT="$HOME/wind-idiff-checkpoint-eval/idiff_m2cond_latest.pt" \
EXPERIMENT="same_checkpoint" \
bash src/eval/windeval/generators/infinite_diffusion/tiling_scaling/configs/submit.sh
```

The generation jobs are serialized, so the sweep occupies at most one GPU. The final
scoring job requests CPU and memory on a Dean node but does not request a GPU.

## Outputs

```text
/share/dean/$USER/balloon-research/outputs/idiff_tiling_scaling/same_checkpoint/report/report.md
/share/dean/$USER/balloon-research/outputs/idiff_tiling_scaling/same_checkpoint/report/results.json
/share/dean/$USER/balloon-research/outputs/idiff_tiling_scaling/same_checkpoint/report/summary.csv
/share/dean/$USER/balloon-research/outputs/idiff_tiling_scaling/same_checkpoint/report/runtime_vs_tiles.png
/share/dean/$USER/balloon-research/outputs/idiff_tiling_scaling/same_checkpoint/report/quality_vs_tiles.png
/share/dean/$USER/balloon-research/outputs/idiff_tiling_scaling/same_checkpoint/report/coherence_vs_tiles.png
/share/dean/$USER/balloon-research/outputs/idiff_tiling_scaling/same_checkpoint/report/quality_runtime_tradeoff.png
/share/dean/$USER/balloon-research/outputs/idiff_tiling_scaling/same_checkpoint/report/matched_sample_montage.png
```

Use a distinct `EXPERIMENT`, such as `crop_matched`, for the matched-checkpoint
comparison. Existing output directories reject configuration changes instead of mixing
samples from different checkpoints or protocols.
