#!/bin/bash
#SBATCH --job-name=rlhab-temporal
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=rlhab-temporal-%j.out
#SBATCH --error=rlhab-temporal-%j.err

set -euo pipefail
REPO="${REPO:-$HOME/balloon-env-dev-code}"
PYTHON="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
DATA_ROOT="${DATA_ROOT:-/share/dean/$USER/balloon-research}"
REFERENCE="${REFERENCE:-$DATA_ROOT/era5/era5_temporal_24h.zarr}"
SYNTH="${SYNTH:-$DATA_ROOT/rlhab/rlhab_synthwinds_heldout.zarr}"
OUTPUT="${OUTPUT:-$DATA_ROOT/outputs/rlhab_synthwinds_v2/rlhab_temporal_metrics.md}"

cd "$REPO"
export PYTHONPATH="$REPO/src/eval${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

"$PYTHON" -m windeval.generators.rlhab_synthwinds.score_temporal \
  --reference "$REFERENCE" \
  --synthwinds "$SYNTH" \
  --output "$OUTPUT"
