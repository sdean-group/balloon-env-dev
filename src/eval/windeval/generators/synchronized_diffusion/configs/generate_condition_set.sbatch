#!/bin/bash
#SBATCH --job-name=sync-cond-set
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=sync-cond-set-%j.out
#SBATCH --error=sync-cond-set-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
PY="${PYTHON:-python}"

"$PY" src/eval/windeval/generators/synchronized_diffusion/generate_condition_set.py \
  --strategy "${STRATEGY:?set STRATEGY}" \
  --checkpoint "${CHECKPOINT:?set CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR:?set OUTPUT_DIR}" \
  --device cuda \
  --num-steps "${NUM_STEPS:-18}" \
  --num-seeds "${NUM_SEEDS:-2}" \
  --query-size 64 \
  --query-frames 4 \
  --query-t0 2 \
  --query-y0 32 \
  --query-x0 32 \
  --lat-origin 25 \
  --lon-origin 225 \
  --window-batch-size "${WINDOW_BATCH_SIZE:-1}" \
  --max-cached-charts "${MAX_CACHED_CHARTS:-64}" \
  --guidance-strength "${GUIDANCE_STRENGTH:-0.15}" \
  --consensus-rounds "${CONSENSUS_ROUNDS:-2}" \
  --consensus-relaxation "${CONSENSUS_RELAXATION:-0.5}" \
  --dual-scale "${DUAL_SCALE:-0.25}"
