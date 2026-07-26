#!/bin/bash
#SBATCH --job-name=cfgd-cond-set
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=cfgd-cond-set-%j.out
#SBATCH --error=cfgd-cond-set-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
PY="${PYTHON:-python}"
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

"$PY" src/eval/windeval/generators/canonical_factor_graph/generate_condition_set.py \
  --checkpoint "${CHECKPOINT:-idiff_m2cond_latest.pt}" \
  --output-dir "${OUTPUT_DIR:?set OUTPUT_DIR}" \
  --device cuda \
  --num-steps "${NUM_STEPS:-18}" \
  --num-seeds "${NUM_SEEDS:-2}" \
  --window-batch-size "${WINDOW_BATCH_SIZE:-1}" \
  "${FILTER_ARGS[@]}"
