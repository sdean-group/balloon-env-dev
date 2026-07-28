#!/bin/bash
#SBATCH --job-name=era5-temporal
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=era5-temporal-%j.out
#SBATCH --error=era5-temporal-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
PY="${PYTHON:-python}"

PYTHONPATH="$REPO/src/eval${PYTHONPATH:+:$PYTHONPATH}" \
"$PY" -m windeval.download_arco_temporal_reference \
  --output "${REFERENCE:?set REFERENCE}" \
  --workers "${WORKERS:-1}"
