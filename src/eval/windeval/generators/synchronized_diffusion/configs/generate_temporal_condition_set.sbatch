#!/bin/bash
#SBATCH --job-name=sync-temporal
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=sync-temporal-%j.out
#SBATCH --error=sync-temporal-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
PY="${PYTHON:-python}"

"$PY" src/eval/windeval/generators/synchronized_diffusion/generate_temporal_condition_set.py \
  --strategy "${STRATEGY:?set STRATEGY}" \
  --checkpoint "${CHECKPOINT:?set CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR:?set OUTPUT_DIR}" \
  --device cuda \
  --num-steps "${NUM_STEPS:-18}" \
  --query-frames 24 \
  --max-cached-charts "${MAX_CACHED_CHARTS:-64}" \
  --guidance-strength "${GUIDANCE_STRENGTH:-0.15}" \
  --consensus-rounds "${CONSENSUS_ROUNDS:-2}" \
  --consensus-relaxation "${CONSENSUS_RELAXATION:-0.5}" \
  --dual-scale "${DUAL_SCALE:-0.25}"
