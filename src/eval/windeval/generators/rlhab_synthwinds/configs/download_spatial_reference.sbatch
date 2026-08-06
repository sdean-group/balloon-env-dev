#!/bin/bash
#SBATCH --job-name=era5-spatial-v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=era5-spatial-v2-%j.out
#SBATCH --error=era5-spatial-v2-%j.err

set -euo pipefail
REPO="${REPO:-$HOME/balloon-env-dev-code}"
PYTHON="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
DATA_ROOT="${DATA_ROOT:-/share/dean/$USER/balloon-research}"
REFERENCE="${REFERENCE:-$DATA_ROOT/era5/era5_spatial_v2_4h.zarr}"
TEMPORAL_PARTS="${TEMPORAL_PARTS:-$DATA_ROOT/era5/era5_temporal_24h.zarr.parts}"

cd "$REPO"
PYTHONPATH="$REPO/src/eval${PYTHONPATH:+:$PYTHONPATH}" \
"$PYTHON" -m windeval.download_arco_spatial_reference \
  --output "$REFERENCE" \
  --temporal-parts "$TEMPORAL_PARTS" \
  --workers "${WORKERS:-1}"
