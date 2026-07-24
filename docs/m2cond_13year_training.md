# Thirteen-Year Conditional Base-Model Training

## Experimental contract

This is a controlled data-scaling run of `idiff_m2cond_latest.pt`, not a new model
architecture. It retains the original:

- EDM denoising objective and preconditioning;
- conditional factorized space-time U-Net;
- 4 consecutive hourly frames;
- 18 model levels (49-66), interleaved `u,v`;
- 64x64 crops on the 0.25-degree NE-Pacific grid;
- latitude/longitude channels and annual, semiannual, and diurnal time features;
- batch size 16, Adam at `2e-4`, EMA `0.9999`, and 1,000-step warmup;
- 100,000 optimizer steps, seed 0, and checkpoint interval 2,000.

Only the data scale changes. Per-level `u,v` normalization is recomputed from the
training years, as required for a statistically correct larger-data run. Validation
does not update the model.

## Time split

| Role | Years | Treatment |
|---|---|---|
| Training | 2009-2021 | All 13 complete years, hourly |
| Validation | 2022 | Fixed validation batches every 2,000 steps |
| Benchmark | 2023 | Never downloaded into either training store |

The source region and variables match the original model: 25-55 N, 225-255 E,
ERA5 model levels 49-66, hourly `u` and `v`. "Full ERA5" here means all hourly data
for this fixed regional/model-level contract, not the global ERA5 archive.

Expected persistent storage is approximately 240 GB for training, 19 GB for
validation, up to 2 GB of temporary monthly GRIB data, and roughly 35 GB if every
2,000-step checkpoint is retained.

## One-time environment check

The download uses the Copernicus `reanalysis-era5-complete` model-level product.
`~/.cdsapirc` must exist and the account must have accepted that dataset's license.

```bash
PY="$HOME/envs/idiff-eval-titan/bin/python"
"$PY" -m pip install "zarr<3" xarray pandas dask cdsapi cfgrib eccodes pyyaml
"$PY" -c 'import cdsapi, cfgrib, eccodes, xarray, zarr; print("ERA5 stack ready")'
test -f "$HOME/.cdsapirc" && echo "CDS credentials found"
```

## Prepare the stores

Run both CPU-only jobs. They use Dean compute nodes but request no GPU. Downloads are
monthly and resumable: re-submitting skips every month already appended. The training
job computes normalization statistics only after all 13 years are present. Each job
then validates the exact hourly timeline, grid, levels, and statistics before writing
its `.complete.json` gate.

```bash
cd "$HOME/balloon-env-dev-code"

REPO="$PWD" \
PYTHON="$HOME/envs/idiff-eval-titan/bin/python" \
DATA_ROOT="/share/dean/$USER/balloon-research/era5" \
sbatch src/eval/windeval/generators/infinite_diffusion/configs/prepare_era5_13year_dean.sbatch

REPO="$PWD" \
PYTHON="$HOME/envs/idiff-eval-titan/bin/python" \
DATA_ROOT="/share/dean/$USER/balloon-research/era5" \
sbatch src/eval/windeval/generators/infinite_diffusion/configs/prepare_era5_2022_validation_dean.sbatch
```

If either job reaches the 24-hour limit, submit that same command again. Do not delete
the Zarr store.

Verify both gates before training:

```bash
DATA_ROOT="/share/dean/$USER/balloon-research/era5"
cat "$DATA_ROOT/era5_2009_2021.complete.json"
cat "$DATA_ROOT/era5_2022.complete.json"
du -sh "$DATA_ROOT/era5_2009_2021.zarr" "$DATA_ROOT/era5_2022.zarr"
```

## Train

```bash
cd "$HOME/balloon-env-dev-code"

REPO="$PWD" \
PYTHON="$HOME/envs/idiff-eval-titan/bin/python" \
DATA_ROOT="/share/dean/$USER/balloon-research/era5" \
OUT_DIR="/share/dean/$USER/balloon-research/runs/idiff_m2cond_era5_13year" \
sbatch src/eval/windeval/generators/infinite_diffusion/configs/train_era5_13year_dean.sbatch
```

The training job requests one RTX 6000 Ada GPU on `dean-compute-02`. Five minutes
before each 24-hour limit, Slurm sends `SIGUSR1`; the trainer atomically writes
`latest.pt` and the batch script requeues the same job. Ordinary preemption is also
requeue-enabled. A genuine Python or data error exits nonzero and is not hidden.

Monitor:

```bash
squeue --me
tail -F idiff-13yr-JOB_ID.out
```

The final and validation-selected checkpoints are:

```text
/share/dean/$USER/balloon-research/runs/idiff_m2cond_era5_13year/latest.pt
/share/dean/$USER/balloon-research/runs/idiff_m2cond_era5_13year/best.pt
```

Use `latest.pt` for the strict step-100,000 comparison with the original checkpoint.
Report `best.pt` separately rather than silently replacing the controlled endpoint.
