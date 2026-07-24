#!/bin/bash
#SBATCH --job-name=era5-2022-val
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dean-compute-[01-02]
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=era5-2022-val-%j.out
#SBATCH --error=era5-2022-val-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
PY="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
DATA_ROOT="${DATA_ROOT:-/share/dean/$USER/balloon-research/era5}"
STORE="$DATA_ROOT/era5_2022.zarr"
MANIFEST="$DATA_ROOT/era5_2022.complete.json"

cd "$REPO"
mkdir -p "$DATA_ROOT"
test -f "$HOME/.cdsapirc"
"$PY" -c 'import cdsapi, cfgrib, xarray, zarr'

"$PY" src/eval/windeval/generators/infinite_diffusion/prepare_era5_multiyear.py \
  --years 2022 \
  --out "$STORE" \
  --work-dir "$DATA_ROOT/grib_tmp/era5_2022"

"$PY" src/eval/windeval/generators/infinite_diffusion/validate_era5_multiyear.py \
  --path "$STORE" \
  --first-year 2022 \
  --last-year 2022 \
  --manifest "$MANIFEST"
