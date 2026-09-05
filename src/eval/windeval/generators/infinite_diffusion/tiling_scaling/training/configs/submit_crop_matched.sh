#!/bin/bash
# Train crop-matched 32/16 models sequentially, then run the controlled benchmark.
set -euo pipefail

REPO="${REPO:-$PWD}"
PYTHON="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
DATA_ROOT="${DATA_ROOT:-/share/dean/$USER/balloon-research}"
DATASET="${DATASET:-$DATA_ROOT/era5/era5_2023.zarr}"
REFERENCE="${REFERENCE:-$DATA_ROOT/era5/era5_heldout_conditional.zarr}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-$HOME/wind-idiff-checkpoint-eval/idiff_m2cond_latest.pt}"
TRAIN_ROOT="${TRAIN_ROOT:-$DATA_ROOT/runs/idiff_crop_matched_2023}"
NODELIST="${NODELIST:-dean-compute-02}"
UPSTREAM_DEPENDENCY="${UPSTREAM_DEPENDENCY:-}"
SBATCH_ROOT="src/eval/windeval/generators/infinite_diffusion/tiling_scaling/training/configs"
SWEEP_SCRIPT="src/eval/windeval/generators/infinite_diffusion/tiling_scaling/configs/submit.sh"

cd "$REPO"
test -x "$PYTHON"
if [[ -z "$UPSTREAM_DEPENDENCY" ]]; then
  test -d "$DATASET" || {
    echo "Missing original 2023 training dataset: $DATASET" >&2
    echo "Set DATASET=/actual/path/to/era5_2023.zarr and rerun." >&2
    exit 2
  }
fi
test -d "$REFERENCE"
test -f "$BASE_CHECKPOINT"
mkdir -p "$TRAIN_ROOT"

preflight_dependency=()
if [[ -n "$UPSTREAM_DEPENDENCY" ]]; then
  preflight_dependency=(--dependency="afterok:$UPSTREAM_DEPENDENCY")
fi

preflight=$(
  DATASET="$DATASET" CHECKPOINT_64="$BASE_CHECKPOINT" \
  PREFLIGHT_REPORT="$TRAIN_ROOT/preflight.json" REPO="$REPO" PYTHON="$PYTHON" \
  sbatch --parsable --nodes=1 --nodelist="$NODELIST" \
    "${preflight_dependency[@]}" \
    "$SBATCH_ROOT/preflight.sbatch"
)
preflight="${preflight%%;*}"

crop32=$(
  CROP=32 DATASET="$DATASET" OUT_DIR="$TRAIN_ROOT/crop32" \
  REPO="$REPO" PYTHON="$PYTHON" \
  sbatch --parsable --nodes=1 --nodelist="$NODELIST" \
    --dependency="afterok:$preflight" --job-name=idiff-crop32 \
    "$SBATCH_ROOT/train_crop.sbatch"
)
crop32="${crop32%%;*}"

crop16=$(
  CROP=16 DATASET="$DATASET" OUT_DIR="$TRAIN_ROOT/crop16" \
  REPO="$REPO" PYTHON="$PYTHON" \
  sbatch --parsable --nodes=1 --nodelist="$NODELIST" \
    --dependency="afterok:$crop32" --job-name=idiff-crop16 \
    "$SBATCH_ROOT/train_crop.sbatch"
)
crop16="${crop16%%;*}"

echo "Preflight:        $preflight"
echo "Train crop 32:    $crop32 (after $preflight)"
echo "Train crop 16:    $crop16 (after $crop32)"
echo "Submitting crop-matched generation and scoring after $crop16..."

REPO="$REPO" PYTHON="$PYTHON" DATA_ROOT="$DATA_ROOT" REFERENCE="$REFERENCE" \
CHECKPOINT_4="$BASE_CHECKPOINT" \
CHECKPOINT_16="$TRAIN_ROOT/crop32/latest.pt" \
CHECKPOINT_64="$TRAIN_ROOT/crop16/latest.pt" \
EXPERIMENT="crop_matched_fixed_updates" \
UPSTREAM_DEPENDENCY="$crop16" \
REQUIRE_CROP_MATCH=1 \
NODELIST="$NODELIST" \
bash "$SWEEP_SCRIPT"
