#!/bin/bash
#SBATCH --job-name=idiff-m2cond-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=idiff-m2cond-eval-%j.out
#SBATCH --error=idiff-m2cond-eval-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
PY="${PYTHON:-python}"

"$PY" src/eval/windeval/generators/infinite_diffusion/evaluate_spacetime_infinite.py \
  --checkpoint "${CHECKPOINT:-idiff_m2cond_latest.pt}" \
  --output-dir "${OUTPUT_DIR:-outputs/m2cond_infinite_full}" \
  --device cuda \
  --num-steps 18 \
  --outer-depth "${OUTER_DEPTH:-1}" \
  --split-step "${SPLIT_STEP:-9}" \
  --window 64 \
  --stride 32 \
  --time-stride 2 \
  --query-size 64 \
  --query-frames 4
