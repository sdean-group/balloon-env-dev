#!/bin/bash
#SBATCH --job-name=idiff-cond-set
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=idiff-cond-set-%j.out
#SBATCH --error=idiff-cond-set-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
PY="${PYTHON:-python}"
SPLIT_ARGS=()
if [[ -n "${SPLIT_STEPS:-}" ]]; then
  read -r -a SPLIT_VALUES <<< "$SPLIT_STEPS"
  SPLIT_ARGS=(--split-steps "${SPLIT_VALUES[@]}")
elif [[ -n "${SPLIT_STEP:-}" ]]; then
  SPLIT_ARGS=(--split-step "$SPLIT_STEP")
fi
FILTER_ARGS=()
for spec in "MONTHS:--months" "DAYS:--days" "HOURS:--hours"; do
  variable="${spec%%:*}"
  option="${spec#*:}"
  value="${!variable:-}"
  if [[ -n "$value" ]]; then
    read -r -a values <<< "$value"
    FILTER_ARGS+=("$option" "${values[@]}")
  fi
done

"$PY" src/eval/windeval/generators/infinite_diffusion/generate_condition_set.py \
  --checkpoint "${CHECKPOINT:-idiff_m2cond_latest.pt}" \
  --output-dir "${OUTPUT_DIR:?set OUTPUT_DIR}" \
  --device cuda \
  --num-steps "${NUM_STEPS:-18}" \
  --outer-depth "${OUTER_DEPTH:-1}" \
  "${SPLIT_ARGS[@]}" \
  --num-seeds "${NUM_SEEDS:-2}" \
  "${FILTER_ARGS[@]}"
