#!/bin/bash
#SBATCH --job-name=crop-preflight
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=crop-preflight-%j.out
#SBATCH --error=crop-preflight-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
PY="${PYTHON:-python}"
CONFIG_ROOT="src/eval/windeval/generators/infinite_diffusion/configs"
cd "$REPO"

PYTHONPATH="$REPO/src/eval${PYTHONPATH:+:$PYTHONPATH}" \
"$PY" -m windeval.generators.infinite_diffusion.tiling_scaling.training.preflight \
  --dataset "${DATASET:?set DATASET}" \
  --checkpoint-64 "${CHECKPOINT_64:?set CHECKPOINT_64}" \
  --config-64 "$CONFIG_ROOT/era5_2023_m2cond.yaml" \
  --config-32 "$CONFIG_ROOT/era5_2023_m2cond_crop32.yaml" \
  --config-16 "$CONFIG_ROOT/era5_2023_m2cond_crop16.yaml" \
  --output "${PREFLIGHT_REPORT:?set PREFLIGHT_REPORT}"
