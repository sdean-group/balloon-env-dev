#!/bin/bash
#SBATCH --job-name=idiff-tile-bench
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=idiff-tile-bench-%j.out
#SBATCH --error=idiff-tile-bench-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
PY="${PYTHON:-python}"
cd "$REPO"

PYTHONPATH="$REPO/src/eval${PYTHONPATH:+:$PYTHONPATH}" \
"$PY" -m windeval.generators.infinite_diffusion.tiling_scaling.benchmark \
  --reference "${REFERENCE:?set REFERENCE}" \
  --run 4 "${RUN_4:?set RUN_4}" \
  --run 16 "${RUN_16:?set RUN_16}" \
  --run 64 "${RUN_64:?set RUN_64}" \
  --output-dir "${OUTPUT_DIR:?set OUTPUT_DIR}"
