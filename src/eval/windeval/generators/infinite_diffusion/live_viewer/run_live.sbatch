#!/bin/bash
#SBATCH --job-name=idiff-live
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dean-compute-[01-02]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=idiff-live-%j.out
#SBATCH --error=idiff-live-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
PY="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
CHECKPOINT="${CHECKPOINT:-$HOME/wind-idiff-checkpoint-eval/idiff_m2cond_latest.pt}"
PORT="${PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

cd "$REPO"

SPLIT_ARGS=()
if [[ -n "${SPLIT_STEPS:-}" ]]; then
  read -r -a SPLIT_VALUES <<< "$SPLIT_STEPS"
  SPLIT_ARGS=(--split-steps "${SPLIT_VALUES[@]}")
elif [[ -n "${SPLIT_STEP:-}" ]]; then
  SPLIT_ARGS=(--split-step "$SPLIT_STEP")
fi

exec "$PY" -u \
  src/eval/windeval/generators/infinite_diffusion/live_viewer/server.py \
  --checkpoint "$CHECKPOINT" \
  --device cuda \
  --host 0.0.0.0 \
  --port "$PORT" \
  --num-steps "${NUM_STEPS:-18}" \
  --outer-depth "${OUTER_DEPTH:-1}" \
  "${SPLIT_ARGS[@]}" \
  --window 64 \
  --stride 32 \
  --time-stride 2 \
  --seed "${SEED:-7}" \
  --lat-origin "${LAT_ORIGIN:-25}" \
  --lon-origin "${LON_ORIGIN:-225}" \
  --time-origin "${TIME_ORIGIN:-2023-01-15T00}" \
  --start-level-index "${START_LEVEL_INDEX:-8}" \
  --playback-hours-per-second "${PLAYBACK_HOURS_PER_SECOND:-0.08}" \
  --cache-gb "${CACHE_GB:-4}"
