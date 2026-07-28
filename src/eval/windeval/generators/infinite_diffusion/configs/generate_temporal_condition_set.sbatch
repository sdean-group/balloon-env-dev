#!/bin/bash
#SBATCH --job-name=wind-temporal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=wind-temporal-%j.out
#SBATCH --error=wind-temporal-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
PY="${PYTHON:-python}"
SPLIT_ARGS=()
if [[ -n "${SPLIT_STEPS:-}" ]]; then
  read -r -a values <<< "$SPLIT_STEPS"
  SPLIT_ARGS=(--split-steps "${values[@]}")
fi

"$PY" src/eval/windeval/generators/infinite_diffusion/generate_temporal_condition_set.py \
  --method "${METHOD:?set METHOD}" \
  --checkpoint "${CHECKPOINT:?set CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR:?set OUTPUT_DIR}" \
  --device cuda \
  --num-steps "${NUM_STEPS:-18}" \
  --outer-depth "${OUTER_DEPTH:-1}" \
  "${SPLIT_ARGS[@]}" \
  --query-frames "${QUERY_FRAMES:-24}" \
  --cache-gb "${CACHE_GB:-4}" \
  --max-cached-charts "${MAX_CACHED_CHARTS:-64}"
