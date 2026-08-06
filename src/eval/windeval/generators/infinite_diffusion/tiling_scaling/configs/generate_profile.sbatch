#!/bin/bash
#SBATCH --job-name=idiff-tiles
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=idiff-tiles-%j.out
#SBATCH --error=idiff-tiles-%j.err
#SBATCH --get-user-env
#SBATCH --requeue

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
PY="${PYTHON:-python}"
cd "$REPO"

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

SPLIT_ARGS=()
if [[ -n "${SPLIT_STEPS:-}" ]]; then
  read -r -a values <<< "$SPLIT_STEPS"
  SPLIT_ARGS=(--split-steps "${values[@]}")
fi

CROP_MATCH_ARGS=()
if [[ "${REQUIRE_CROP_MATCH:-0}" == "1" ]]; then
  CROP_MATCH_ARGS=(--require-crop-match)
fi

PYTHONPATH="$REPO/src/eval${PYTHONPATH:+:$PYTHONPATH}" \
"$PY" -m windeval.generators.infinite_diffusion.tiling_scaling.generate \
  --checkpoint "${CHECKPOINT:?set CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR:?set OUTPUT_DIR}" \
  --core-tiles "${CORE_TILES:?set CORE_TILES to 4, 16, or 64}" \
  --device cuda \
  --num-steps "${NUM_STEPS:-18}" \
  --outer-depth "${OUTER_DEPTH:-1}" \
  "${SPLIT_ARGS[@]}" \
  --num-seeds "${NUM_SEEDS:-2}" \
  --cache-gb "${CACHE_GB:-2}" \
  "${CROP_MATCH_ARGS[@]}" \
  "${FILTER_ARGS[@]}"
