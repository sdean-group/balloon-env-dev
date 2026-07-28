#!/bin/bash
# Submit matched 24-hour generation jobs and their dependent temporal benchmark.
set -euo pipefail

REPO="${REPO:-$PWD}"
PYTHON="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
CHECKPOINT="${CHECKPOINT:-$HOME/wind-idiff-checkpoint-eval/idiff_m2cond_latest.pt}"
DATA_ROOT="${DATA_ROOT:-/share/dean/$USER/balloon-research}"

REFERENCE="${REFERENCE:-$DATA_ROOT/era5/era5_temporal_24h.zarr}"
DIRECT_RUN="${DIRECT_RUN:-$DATA_ROOT/outputs/temporal_direct_base}"
T1_RUN="${T1_RUN:-$DATA_ROOT/outputs/temporal_idiff_t1}"
T2_RUN="${T2_RUN:-$DATA_ROOT/outputs/temporal_idiff_t2_split9}"
T3_RUN="${T3_RUN:-$DATA_ROOT/outputs/temporal_idiff_t3_split6_12}"
CFGD_RUN="${CFGD_RUN:-$DATA_ROOT/outputs/temporal_cfgd}"
OUTPUT="${OUTPUT:-$DATA_ROOT/outputs/all_methods_temporal.md}"

cd "$REPO"
test -x "$PYTHON"
test -f "$CHECKPOINT"
mkdir -p "$DATA_ROOT/era5" "$DATA_ROOT/outputs"

reference_job=$(
  REFERENCE="$REFERENCE" PYTHON="$PYTHON" \
  sbatch --parsable --nodelist='dean-compute-[01-02]' \
    src/eval/windeval/generators/infinite_diffusion/configs/download_temporal_reference.sbatch
)
reference_job="${reference_job%%;*}"

submit_generation() {
  local name="$1"
  local method="$2"
  local output_dir="$3"
  local depth="$4"
  local splits="$5"
  local limit="$6"
  METHOD="$method" \
  OUTPUT_DIR="$output_dir" \
  OUTER_DEPTH="$depth" \
  SPLIT_STEPS="$splits" \
  CHECKPOINT="$CHECKPOINT" \
  PYTHON="$PYTHON" \
  sbatch --parsable \
    --job-name="$name" \
    --time="$limit" \
    --nodelist='dean-compute-[01-02]' \
    src/eval/windeval/generators/infinite_diffusion/configs/generate_temporal_condition_set.sbatch
}

direct_job=$(submit_generation temp-direct direct "$DIRECT_RUN" 1 "" 01:00:00)
t1_job=$(submit_generation temp-t1 infinite "$T1_RUN" 1 "" 03:00:00)
t2_job=$(submit_generation temp-t2 infinite "$T2_RUN" 2 "9" 06:00:00)
t3_job=$(submit_generation temp-t3 infinite "$T3_RUN" 3 "6 12" 08:00:00)
cfgd_job=$(submit_generation temp-cfgd cfgd "$CFGD_RUN" 1 "" 04:00:00)

direct_job="${direct_job%%;*}"
t1_job="${t1_job%%;*}"
t2_job="${t2_job%%;*}"
t3_job="${t3_job%%;*}"
cfgd_job="${cfgd_job%%;*}"

benchmark_job=$(
  REFERENCE="$REFERENCE" \
  DIRECT_RUN="$DIRECT_RUN" \
  T1_RUN="$T1_RUN" \
  T2_RUN="$T2_RUN" \
  T3_RUN="$T3_RUN" \
  CFGD_RUN="$CFGD_RUN" \
  OUTPUT="$OUTPUT" \
  PYTHON="$PYTHON" \
  sbatch --parsable \
    --dependency="afterok:$reference_job:$direct_job:$t1_job:$t2_job:$t3_job:$cfgd_job" \
    --nodelist='dean-compute-[01-02]' \
    src/eval/windeval/generators/infinite_diffusion/configs/benchmark_temporal_all_methods.sbatch
)
benchmark_job="${benchmark_job%%;*}"

printf 'ERA5 reference: %s\n' "$reference_job"
printf 'Direct base:    %s\n' "$direct_job"
printf 'T=1:            %s\n' "$t1_job"
printf 'T=2:            %s\n' "$t2_job"
printf 'T=3:            %s\n' "$t3_job"
printf 'CFGD:           %s\n' "$cfgd_job"
printf 'Benchmark:      %s (waits for all six inputs)\n' "$benchmark_job"
printf 'Report:         %s\n' "$OUTPUT"
