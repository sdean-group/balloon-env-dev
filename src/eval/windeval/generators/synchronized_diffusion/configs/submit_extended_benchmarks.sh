#!/bin/bash
# Submit three synchronized-diffusion alternatives and extended shared benchmarks.
set -euo pipefail

REPO="${REPO:-$PWD}"
PYTHON="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
CHECKPOINT="${CHECKPOINT:-$HOME/wind-idiff-checkpoint-eval/idiff_m2cond_latest.pt}"
DATA_ROOT="${DATA_ROOT:-/share/dean/$USER/balloon-research}"
OLD_RUN_ROOT="${OLD_RUN_ROOT:-$HOME/wind-idiff-checkpoint-eval/outputs}"

SPATIAL_REFERENCE="${SPATIAL_REFERENCE:-$DATA_ROOT/era5/era5_heldout_conditional.zarr}"
TEMPORAL_REFERENCE="${TEMPORAL_REFERENCE:-$DATA_ROOT/era5/era5_temporal_24h.zarr}"
BLE_DECODER="${BLE_DECODER:-$DATA_ROOT/ble/offlineskies22_decoder.msgpack}"
BLE_CACHE="${BLE_CACHE:-$DATA_ROOT/outputs/t2_t3_vs_ble_shared_ble_cache.npz}"

SYNC_TWEEDIES_RUN="$DATA_ROOT/outputs/sync_tweedies_conditions_full64"
OVERLAP_GUIDED_RUN="$DATA_ROOT/outputs/overlap_guided_conditions_full64"
CONSENSUS_RUN="$DATA_ROOT/outputs/consensus_equilibrium_conditions_full64"
TEMP_SYNC_TWEEDIES_RUN="$DATA_ROOT/outputs/temporal_sync_tweedies"
TEMP_OVERLAP_GUIDED_RUN="$DATA_ROOT/outputs/temporal_overlap_guided"
TEMP_CONSENSUS_RUN="$DATA_ROOT/outputs/temporal_consensus_equilibrium"

cd "$REPO"
test -x "$PYTHON"
test -f "$CHECKPOINT"
test -d "$SPATIAL_REFERENCE"
test -f "$BLE_DECODER"
mkdir -p "$DATA_ROOT/outputs"

submit_spatial() {
  local job_name="$1"
  local strategy="$2"
  local output_dir="$3"
  STRATEGY="$strategy" OUTPUT_DIR="$output_dir" \
  CHECKPOINT="$CHECKPOINT" PYTHON="$PYTHON" \
  sbatch --parsable --nodes=1 --job-name="$job_name" \
    --nodelist='dean-compute-[01-02]' \
    src/eval/windeval/generators/synchronized_diffusion/configs/generate_condition_set.sbatch
}

submit_temporal() {
  local job_name="$1"
  local strategy="$2"
  local output_dir="$3"
  local dependency="$4"
  STRATEGY="$strategy" OUTPUT_DIR="$output_dir" \
  CHECKPOINT="$CHECKPOINT" PYTHON="$PYTHON" \
  sbatch --parsable --nodes=1 --job-name="$job_name" \
    --nodelist='dean-compute-[01-02]' \
    --dependency="afterok:$dependency" \
    src/eval/windeval/generators/synchronized_diffusion/configs/generate_temporal_condition_set.sbatch
}

sync_space_job=$(submit_spatial sync-tweed sync_tweedies "$SYNC_TWEEDIES_RUN")
guided_space_job=$(submit_spatial sync-guide overlap_guided "$OVERLAP_GUIDED_RUN")
consensus_space_job=$(submit_spatial sync-ce consensus_equilibrium "$CONSENSUS_RUN")
sync_space_job="${sync_space_job%%;*}"
guided_space_job="${guided_space_job%%;*}"
consensus_space_job="${consensus_space_job%%;*}"

# Chain each method's shorter temporal run behind its spatial run. This limits the
# submission to at most three simultaneous GPUs while keeping methods independent.
sync_time_job=$(submit_temporal time-tweed sync_tweedies "$TEMP_SYNC_TWEEDIES_RUN" "$sync_space_job")
guided_time_job=$(submit_temporal time-guide overlap_guided "$TEMP_OVERLAP_GUIDED_RUN" "$guided_space_job")
consensus_time_job=$(submit_temporal time-ce consensus_equilibrium "$TEMP_CONSENSUS_RUN" "$consensus_space_job")
sync_time_job="${sync_time_job%%;*}"
guided_time_job="${guided_time_job%%;*}"
consensus_time_job="${consensus_time_job%%;*}"

spatial_benchmark_job=$(
  SPATIAL_REFERENCE="$SPATIAL_REFERENCE" \
  DIRECT_RUN="$DATA_ROOT/outputs/m2cond_conditions_direct_base" \
  T1_RUN="$OLD_RUN_ROOT/m2cond_conditions_t1" \
  T2_RUN="$OLD_RUN_ROOT/m2cond_conditions_t2_split9" \
  T3_RUN="$DATA_ROOT/outputs/m2cond_conditions_t3" \
  CFGD_RUN="$DATA_ROOT/outputs/cfgd_conditions_full64" \
  SYNC_TWEEDIES_RUN="$SYNC_TWEEDIES_RUN" \
  OVERLAP_GUIDED_RUN="$OVERLAP_GUIDED_RUN" \
  CONSENSUS_RUN="$CONSENSUS_RUN" \
  BLE_DECODER="$BLE_DECODER" \
  BLE_CACHE="$BLE_CACHE" \
  SPATIAL_OUTPUT="$DATA_ROOT/outputs/all_methods_plus_sync_shared.md" \
  PYTHON="$PYTHON" \
  sbatch --parsable --nodes=1 --nodelist='dean-compute-[01-02]' \
    --dependency="afterok:$sync_space_job:$guided_space_job:$consensus_space_job" \
    src/eval/windeval/generators/synchronized_diffusion/configs/benchmark_extended_shared.sbatch
)
spatial_benchmark_job="${spatial_benchmark_job%%;*}"

temporal_dependencies="$sync_time_job:$guided_time_job:$consensus_time_job"
if [[ -n "${TEMPORAL_BASELINES_JOB:-}" ]]; then
  temporal_dependencies="$TEMPORAL_BASELINES_JOB:$temporal_dependencies"
fi
if [[ -n "${TEMPORAL_REFERENCE_JOB:-}" ]]; then
  temporal_dependencies="$TEMPORAL_REFERENCE_JOB:$temporal_dependencies"
elif [[ ! -d "$TEMPORAL_REFERENCE" ]]; then
  echo "TEMPORAL_REFERENCE is incomplete; set TEMPORAL_REFERENCE_JOB to its running job ID" >&2
  exit 1
fi

temporal_benchmark_job=$(
  TEMPORAL_REFERENCE="$TEMPORAL_REFERENCE" \
  TEMP_DIRECT_RUN="$DATA_ROOT/outputs/temporal_direct_base" \
  TEMP_T1_RUN="$DATA_ROOT/outputs/temporal_idiff_t1" \
  TEMP_T2_RUN="$DATA_ROOT/outputs/temporal_idiff_t2_split9" \
  TEMP_T3_RUN="$DATA_ROOT/outputs/temporal_idiff_t3_split6_12" \
  TEMP_CFGD_RUN="$DATA_ROOT/outputs/temporal_cfgd" \
  TEMP_SYNC_TWEEDIES_RUN="$TEMP_SYNC_TWEEDIES_RUN" \
  TEMP_OVERLAP_GUIDED_RUN="$TEMP_OVERLAP_GUIDED_RUN" \
  TEMP_CONSENSUS_RUN="$TEMP_CONSENSUS_RUN" \
  TEMPORAL_OUTPUT="$DATA_ROOT/outputs/all_methods_plus_sync_temporal.md" \
  PYTHON="$PYTHON" \
  sbatch --parsable --nodes=1 --nodelist='dean-compute-[01-02]' \
    --dependency="afterok:$temporal_dependencies" \
    src/eval/windeval/generators/synchronized_diffusion/configs/benchmark_extended_temporal.sbatch
)
temporal_benchmark_job="${temporal_benchmark_job%%;*}"

printf 'SyncTweedies spatial:  %s\n' "$sync_space_job"
printf 'Overlap-guided spatial:%s\n' "$guided_space_job"
printf 'Consensus spatial:     %s\n' "$consensus_space_job"
printf 'SyncTweedies temporal: %s\n' "$sync_time_job"
printf 'Overlap-guided temporal:%s\n' "$guided_time_job"
printf 'Consensus temporal:    %s\n' "$consensus_time_job"
printf 'Extended spatial table:%s\n' "$spatial_benchmark_job"
printf 'Extended temporal table:%s\n' "$temporal_benchmark_job"
