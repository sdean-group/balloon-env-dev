#!/bin/bash
#SBATCH --job-name=era5-13yr-data
#SBATCH --partition=gpu
#SBATCH --nodelist=dean-compute-[01-02]
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=era5-13yr-data-%j.out
#SBATCH --error=era5-13yr-data-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
PY="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
DATA_ROOT="${DATA_ROOT:-/share/dean/$USER/balloon-research/era5}"
STORE="$DATA_ROOT/era5_2009_2021.zarr"
STATS="$DATA_ROOT/era5_2009_2021_norm_stats.npz"
MANIFEST="$DATA_ROOT/era5_2009_2021.complete.json"

cd "$REPO"
mkdir -p "$DATA_ROOT"
test -f "$HOME/.cdsapirc"
"$PY" -c 'import cdsapi, cfgrib, xarray, zarr'

"$PY" src/eval/windeval/generators/infinite_diffusion/prepare_era5_multiyear.py \
  --years {2009..2021} \
  --out "$STORE" \
  --work-dir "$DATA_ROOT/grib_tmp/era5_2009_2021" \
  --stats-out "$STATS"

"$PY" src/eval/windeval/generators/infinite_diffusion/validate_era5_multiyear.py \
  --path "$STORE" \
  --first-year 2009 \
  --last-year 2021 \
  --stats "$STATS" \
  --manifest "$MANIFEST"
