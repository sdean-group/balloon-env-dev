#!/bin/bash
# Submit missing direct-base/CFGD generation and the dependent shared benchmark.
set -euo pipefail

REPO="${REPO:-$PWD}"
PYTHON="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
CHECKPOINT="${CHECKPOINT:-$HOME/wind-idiff-checkpoint-eval/idiff_m2cond_latest.pt}"
DATA_ROOT="${DATA_ROOT:-/share/dean/$USER/balloon-research}"

REFERENCE="${REFERENCE:-$DATA_ROOT/era5/era5_heldout_conditional.zarr}"
BASE_RUN="${BASE_RUN:-$DATA_ROOT/outputs/m2cond_conditions_direct_base}"
T1_RUN="${T1_RUN:-$HOME/wind-idiff-checkpoint-eval/outputs/m2cond_conditions_t1}"
T2_RUN="${T2_RUN:-$HOME/wind-idiff-checkpoint-eval/outputs/m2cond_conditions_t2_split9}"
T3_RUN="${T3_RUN:-$DATA_ROOT/outputs/m2cond_conditions_t3}"
CFGD_RUN="${CFGD_RUN:-$DATA_ROOT/outputs/cfgd_conditions_full64}"
BLE_DECODER="${BLE_DECODER:-$DATA_ROOT/ble/offlineskies22_decoder.msgpack}"
BLE_CACHE="${BLE_CACHE:-$DATA_ROOT/outputs/t2_t3_vs_ble_shared_ble_cache.npz}"
OUTPUT="${OUTPUT:-$DATA_ROOT/outputs/all_methods_shared.md}"

cd "$REPO"
test -x "$PYTHON"
test -f "$CHECKPOINT"
test -d "$REFERENCE"
test -d "$T1_RUN"
test -d "$T2_RUN"
test -d "$T3_RUN"
test -f "$BLE_DECODER"
mkdir -p "$DATA_ROOT/outputs"

direct_job=$(
  CHECKPOINT="$CHECKPOINT" \
  OUTPUT_DIR="$BASE_RUN" \
  NUM_SEEDS=2 \
  PYTHON="$PYTHON" \
  sbatch --parsable \
    src/eval/windeval/generators/infinite_diffusion/configs/generate_direct_base_condition_set.sbatch
)
direct_job="${direct_job%%;*}"

cfgd_job=$(
  CHECKPOINT="$CHECKPOINT" \
  OUTPUT_DIR="$CFGD_RUN" \
  QUERY_SIZE=64 \
  QUERY_T0=2 \
  QUERY_Y0=32 \
  QUERY_X0=32 \
  LAT_ORIGIN=25 \
  LON_ORIGIN=225 \
  NUM_SEEDS=2 \
  PYTHON="$PYTHON" \
  sbatch --parsable \
    src/eval/windeval/generators/canonical_factor_graph/configs/generate_condition_set.sbatch
)
cfgd_job="${cfgd_job%%;*}"

benchmark_job=$(
  REFERENCE="$REFERENCE" \
  BASE_RUN="$BASE_RUN" \
  T1_RUN="$T1_RUN" \
  T2_RUN="$T2_RUN" \
  T3_RUN="$T3_RUN" \
  CFGD_RUN="$CFGD_RUN" \
  BLE_DECODER="$BLE_DECODER" \
  BLE_CACHE="$BLE_CACHE" \
  OUTPUT="$OUTPUT" \
  PYTHON="$PYTHON" \
  sbatch --parsable \
    --dependency="afterok:$direct_job:$cfgd_job" \
    src/eval/windeval/generators/infinite_diffusion/configs/benchmark_all_shared.sbatch
)
benchmark_job="${benchmark_job%%;*}"

printf 'direct-base job: %s\n' "$direct_job"
printf 'CFGD job:        %s\n' "$cfgd_job"
printf 'benchmark job:   %s (waits for both)\n' "$benchmark_job"
printf 'report:          %s\n' "$OUTPUT"
