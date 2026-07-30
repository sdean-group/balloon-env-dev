#!/bin/bash
# Submit a sequential, resumable 4/16/64-core tile sweep and its CPU-only scorer.
set -euo pipefail

REPO="${REPO:-$PWD}"
PYTHON="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
DATA_ROOT="${DATA_ROOT:-/share/dean/$USER/balloon-research}"
CHECKPOINT="${CHECKPOINT:-$HOME/wind-idiff-checkpoint-eval/idiff_m2cond_latest.pt}"
CHECKPOINT_4="${CHECKPOINT_4:-$CHECKPOINT}"
CHECKPOINT_16="${CHECKPOINT_16:-$CHECKPOINT}"
CHECKPOINT_64="${CHECKPOINT_64:-$CHECKPOINT}"
REFERENCE="${REFERENCE:-$DATA_ROOT/era5/era5_heldout_conditional.zarr}"
EXPERIMENT="${EXPERIMENT:-same_checkpoint}"
UPSTREAM_DEPENDENCY="${UPSTREAM_DEPENDENCY:-}"
NODELIST="${NODELIST:-dean-compute-02}"
REQUIRE_CROP_MATCH="${REQUIRE_CROP_MATCH:-0}"
ROOT="$DATA_ROOT/outputs/idiff_tiling_scaling/$EXPERIMENT"
RUN_4="$ROOT/tiles_4"
RUN_16="$ROOT/tiles_16"
RUN_64="$ROOT/tiles_64"
REPORT="$ROOT/report"
SBATCH_DIR="src/eval/windeval/generators/infinite_diffusion/tiling_scaling/configs"

cd "$REPO"
test -x "$PYTHON"
test -f "$CHECKPOINT_4"
test -f "$CHECKPOINT_16"
test -f "$CHECKPOINT_64"
test -d "$REFERENCE"
mkdir -p "$ROOT"

submit_generation() {
  local count="$1"
  local checkpoint="$2"
  local output="$3"
  local dependency="${4:-}"
  local dependency_args=()
  if [[ -n "$dependency" ]]; then
    dependency_args=(--dependency="afterok:$dependency")
  fi
  CORE_TILES="$count" CHECKPOINT="$checkpoint" OUTPUT_DIR="$output" \
  PYTHON="$PYTHON" REPO="$REPO" REQUIRE_CROP_MATCH="$REQUIRE_CROP_MATCH" \
  sbatch --parsable --nodes=1 --nodelist="$NODELIST" \
    --job-name="tiles-$count" "${dependency_args[@]}" \
    "$SBATCH_DIR/generate_profile.sbatch"
}

# Serialize the GPU work. This prevents the sweep from occupying multiple group GPUs.
job4=$(submit_generation 4 "$CHECKPOINT_4" "$RUN_4" "$UPSTREAM_DEPENDENCY")
job4="${job4%%;*}"
job16=$(submit_generation 16 "$CHECKPOINT_16" "$RUN_16" "$job4")
job16="${job16%%;*}"
job64=$(submit_generation 64 "$CHECKPOINT_64" "$RUN_64" "$job16")
job64="${job64%%;*}"

bench=$(
  REFERENCE="$REFERENCE" RUN_4="$RUN_4" RUN_16="$RUN_16" RUN_64="$RUN_64" \
  OUTPUT_DIR="$REPORT" PYTHON="$PYTHON" REPO="$REPO" \
  sbatch --parsable --nodes=1 --nodelist="$NODELIST" \
    --dependency="afterok:$job64" \
    "$SBATCH_DIR/benchmark.sbatch"
)
bench="${bench%%;*}"

printf '4-core generation:  %s\n' "$job4"
printf '16-core generation: %s (after %s)\n' "$job16" "$job4"
printf '64-core generation: %s (after %s)\n' "$job64" "$job16"
printf 'Scoring and plots:  %s (after %s)\n' "$bench" "$job64"
printf 'Final report: %s/report.md\n' "$REPORT"
