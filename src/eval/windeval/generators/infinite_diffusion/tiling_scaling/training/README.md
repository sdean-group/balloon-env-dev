# Crop-matched tile-scaling training

## Purpose

The completed same-checkpoint experiment changed two variables at once:

1. the number of overlapping tiles used to generate a fixed 64x64 field; and
2. the denoiser input size, even though the denoiser had only seen 64x64 crops in training.

This workflow removes the second confound by training one base denoiser for each inference
window size.

| Benchmark row | Core regions | Denoiser window | Training checkpoint |
|---:|---:|---:|---|
| 4 cores | 2x2 | 64x64 | existing original 64x64 checkpoint |
| 16 cores | 4x4 | 32x32 | newly trained crop-32 checkpoint |
| 64 cores | 8x8 | 16x16 | newly trained crop-16 checkpoint |

All rows still generate and score the same 64x64 physical domain. The labels count
non-overlapping core regions; 50% overlap means the actual window counts are 9, 25, and 81.

## Primary training control

The crop-32 and crop-16 configurations reproduce the original checkpoint recipe:

- the original 2023 ERA5 training store, with days 8-14 excluded;
- model levels 49-66, u and v;
- four consecutive hourly frames;
- latitude, longitude, and six cyclic calendar features;
- the same U-Net, temporal layers, and attention downsample factors;
- EDM loss and noise distribution;
- batch size 16;
- Adam with learning rate 2e-4 and 1,000-step warmup;
- EMA decay 0.9999;
- 100,000 optimizer updates;
- seed 0;
- checkpoints every 2,000 updates.

Only `crop` and `out_dir` differ. The preflight job compares the controlled recipe directly
against the original step-100,000 checkpoint and refuses to train if anything else differs.

This is a **fixed-update control**. Crop-32 sees one quarter as many wind pixels per update
as crop-64, and crop-16 sees one sixteenth as many. That is intentional for the primary
comparison because it preserves the original training recipe. A pixel-budget control is a
separate experiment and must be reported separately.

## Required data

The 90 MB held-out conditional reference is not training data. This experiment needs the
original `era5_2023.zarr` used to train `idiff_m2cond_latest.pt`.

Locate it on Unicorn:

```bash
find /share/dean "$HOME" -maxdepth 7 -type d \
  -name 'era5_2023.zarr' 2>/dev/null
```

Confirm its dimensions and timestamps:

```bash
PY="$HOME/envs/idiff-eval-titan/bin/python"
DATASET="/actual/path/to/era5_2023.zarr"

"$PY" -c '
import xarray as xr
ds=xr.open_zarr("'"$DATASET"'", consolidated=False)
print(dict(ds.sizes))
print(ds.time.values[0], ds.time.values[-1])
print(list(ds.data_vars))
'
```

The automated preflight additionally verifies 18 levels, a grid at least 64x64, year 2023,
u/v variables, and the absence of held-out days 8-14.

## Submit the complete experiment

Run from the Unicorn repository after this implementation is present there:

```bash
cd ~/balloon-env-dev-code

DATASET="/actual/path/to/era5_2023.zarr" \
PYTHON="$HOME/envs/idiff-eval-titan/bin/python" \
BASE_CHECKPOINT="$HOME/wind-idiff-checkpoint-eval/idiff_m2cond_latest.pt" \
bash \
  src/eval/windeval/generators/infinite_diffusion/tiling_scaling/training/configs/submit_crop_matched.sh
```

## Start from a tar archive on a Mac

Do not use a path ending in `.download`; wait until the browser has finalized the file.
Then, on the Mac:

```bash
cd "/Users/rohanshankar/Downloads/Balloon Research/wind-idiff-checkpoint-eval"

bash \
  src/eval/windeval/generators/infinite_diffusion/tiling_scaling/training/upload_archive_from_mac.sh \
  "$HOME/Downloads/era5_2023.zarr.tar"
```

The uploader:

- fully reads the tar index to detect an incomplete archive;
- rejects absolute paths, parent traversal, and archive links;
- requires evidence of a Zarr store;
- transfers resumably with `rsync --partial`;
- writes a SHA-256 sidecar that the extraction job verifies on the compute node.

After the code is present in `~/balloon-env-dev-code`, submit the entire extraction,
training, generation, and scoring chain from Unicorn:

```bash
cd ~/balloon-env-dev-code

PYTHON="$HOME/envs/idiff-eval-titan/bin/python" \
ARCHIVE="/share/dean/$USER/balloon-research/incoming/era5_2023.zarr.tar" \
bash \
  src/eval/windeval/generators/infinite_diffusion/tiling_scaling/training/configs/submit_from_archive.sh
```

This command returns immediately after adding all jobs to Slurm. Extraction runs on a
Dean compute node rather than the shared login node. It installs the Zarr atomically, then
the existing preflight validates its variables, dimensions, year, and held-out-day
exclusions before either training job can start.

The submission creates this dependency chain:

```text
preflight
  -> train crop-32
  -> train crop-16
  -> generate 4-core samples
  -> generate 16-core samples
  -> generate 64-core samples
  -> score and plot
```

Only one GPU job runs at a time. Training jobs receive a warning before the 24-hour Slurm
limit, save `latest.pt`, requeue themselves, and resume at the saved optimizer step.

## Monitor

```bash
squeue --me

watch -n 20 '
squeue --me
echo
tail -n 8 ~/balloon-env-dev-code/idiff-crop-*.out 2>/dev/null
'
```

Inspect exact states, including completed and requeued jobs:

```bash
sacct --starttime today \
  --format=JobID,JobName%20,State,ExitCode,Elapsed,NodeList%24 |
grep -E 'crop-preflight|idiff-crop|tiles-|idiff-tile-bench'
```

Training progress is the latest line containing `step`:

```bash
grep -h '\[train\] step' idiff-crop-*.out | tail
```

## Expected cost

Actual time should be estimated from the first 1,000 updates:

```text
remaining hours = (100000 - current step) / reported iterations_per_second / 3600
```

Crop-32 will be the longer of the two new training runs. A reasonable planning range on one
RTX 6000 Ada is roughly one to two days total, but the live iteration rate is authoritative.
The existing crop-64 model is reused, so it is not retrained.

Each checkpoint is about 641 MB. With `latest.pt` plus 50 numbered checkpoints per model,
the strict original cadence can consume roughly 65 GB for the two new models. Generated
condition sets and reports add less than 1 GB. Keep the numbered checkpoints until the
benchmark is complete; later, retain `latest.pt`, the final numbered checkpoint, logs, and
the preflight manifest.

## Outputs

```text
/share/dean/$USER/balloon-research/runs/idiff_crop_matched_2023/
  preflight.json
  crop32/latest.pt
  crop32/complete.json
  crop16/latest.pt
  crop16/complete.json

/share/dean/$USER/balloon-research/outputs/idiff_tiling_scaling/
  crop_matched_fixed_updates/
    tiles_4/
    tiles_16/
    tiles_64/
    report/
      report.md
      results.json
      summary.csv
      runtime_vs_tiles.png
      quality_vs_tiles.png
      coherence_vs_tiles.png
      quality_runtime_tradeoff.png
      matched_sample_montage.png
```

The generator records each checkpoint's training crop and recipe. Crop-matched mode rejects
the wrong checkpoint before generating samples, and the scorer rejects mixed recipes.

## Interpretation

Compare the crop-matched report with the existing same-checkpoint report:

- If 16/64-core quality improves substantially, much of the previous degradation was caused
  by inference at unseen crop sizes.
- If quality remains poor while seams stay normal, small-window training itself loses
  large-scale atmospheric structure.
- If boundary diagnostics worsen, overlap consensus is the limiting factor.
- If all crop-matched models remain far from ERA5, the base denoising objective, data, or
  conditioning is the dominant limitation rather than tiling.

Do not compare pointwise generated fields to the realized ERA5 weather event. Compare their
distributions, spectra, tails, coherence, and boundary statistics over all matched conditions.
