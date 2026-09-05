#!/bin/bash
#SBATCH --job-name=idiff-ensemble
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dean-compute-[01-02]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=idiff-ensemble-%j.out
#SBATCH --error=idiff-ensemble-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
PY="${PYTHON:-python}"
cd "$REPO"

PYTHONPATH="$REPO/src/eval${PYTHONPATH:+:$PYTHONPATH}" \
"$PY" -m windeval.generators.infinite_diffusion.tiling_scaling.ensemble_diagnostics \
  --reference "${REFERENCE:?set REFERENCE}" \
  --run 4 "${RUN_4:?set RUN_4}" \
  --run 16 "${RUN_16:?set RUN_16}" \
  --run 64 "${RUN_64:?set RUN_64}" \
  --output-dir "${OUTPUT_DIR:?set OUTPUT_DIR}"
