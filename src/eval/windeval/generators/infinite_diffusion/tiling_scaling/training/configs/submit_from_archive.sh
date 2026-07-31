#!/bin/bash
# Submit extraction, crop-matched training, generation, and scoring as one dependency chain.
set -euo pipefail

REPO="${REPO:-$PWD}"
PYTHON="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
DATA_ROOT="${DATA_ROOT:-/share/dean/$USER/balloon-research}"
ARCHIVE="${ARCHIVE:-$DATA_ROOT/incoming/era5_2023.zarr.tar}"
DATASET="${DATASET:-$DATA_ROOT/era5/era5_2023.zarr}"
NODELIST="${NODELIST:-dean-compute-02}"
EXTRACT_SCRIPT="src/eval/windeval/generators/infinite_diffusion/tiling_scaling/training/configs/extract_era5_archive.sbatch"
TRAIN_SCRIPT="src/eval/windeval/generators/infinite_diffusion/tiling_scaling/training/configs/submit_crop_matched.sh"

cd "$REPO"
test -x "$PYTHON"
test -f "$ARCHIVE"
test ! -e "$DATASET" || {
  echo "Dataset destination already exists: $DATASET" >&2
  echo "Use submit_crop_matched.sh directly if that dataset has already passed validation." >&2
  exit 2
}

extract=$(
  ARCHIVE="$ARCHIVE" DESTINATION="$DATASET" PYTHON="$PYTHON" \
  sbatch --parsable --nodes=1 --nodelist="$NODELIST" "$EXTRACT_SCRIPT"
)
extract="${extract%%;*}"
echo "ERA5 extraction:  $extract"

REPO="$REPO" PYTHON="$PYTHON" DATA_ROOT="$DATA_ROOT" DATASET="$DATASET" \
NODELIST="$NODELIST" UPSTREAM_DEPENDENCY="$extract" \
bash "$TRAIN_SCRIPT"
