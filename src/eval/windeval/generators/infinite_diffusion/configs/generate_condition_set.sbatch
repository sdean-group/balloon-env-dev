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

"$PY" src/eval/windeval/generators/infinite_diffusion/generate_condition_set.py \
  --checkpoint "${CHECKPOINT:-idiff_m2cond_latest.pt}" \
  --output-dir "${OUTPUT_DIR:?set OUTPUT_DIR}" \
  --device cuda \
  --num-steps "${NUM_STEPS:-18}" \
  --outer-depth "${OUTER_DEPTH:-1}" \
  --split-step "${SPLIT_STEP:-9}" \
  --num-seeds "${NUM_SEEDS:-2}"
