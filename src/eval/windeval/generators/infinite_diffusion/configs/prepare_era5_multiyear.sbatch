#!/bin/bash
#SBATCH --job-name=era5-prepare
#SBATCH --partition=default_partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=era5-prepare-%j.out
#SBATCH --error=era5-prepare-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
PY="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to persistent project/shared storage}"
YEARS="${YEARS:?set YEARS, for example '2012 2013'}"
DATASET_NAME="${DATASET_NAME:?set DATASET_NAME, for example era5_2012_2021}"
STATS_NAME="${STATS_NAME:-}"

cd "$REPO"
mkdir -p "$DATA_ROOT" "$DATA_ROOT/grib_tmp"

stats_args=()
if [[ -n "$STATS_NAME" ]]; then
  stats_args=(--stats-out "$DATA_ROOT/$STATS_NAME")
fi

# YEARS deliberately expands into separate integer arguments.
# shellcheck disable=SC2086
"$PY" src/eval/windeval/generators/infinite_diffusion/prepare_era5_multiyear.py \
  --years $YEARS \
  --out "$DATA_ROOT/$DATASET_NAME.zarr" \
  --work-dir "$DATA_ROOT/grib_tmp/$DATASET_NAME" \
  "${stats_args[@]}"
