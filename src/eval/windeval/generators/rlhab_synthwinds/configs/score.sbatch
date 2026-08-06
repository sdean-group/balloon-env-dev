#!/bin/bash
#SBATCH --job-name=rlhab-score
#SBATCH --output=rlhab-score-%j.out
#SBATCH --error=rlhab-score-%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
REPO="${REPO:-$HOME/balloon-env-dev-code}"
PYTHON="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
DATA_ROOT="${DATA_ROOT:-/share/dean/$USER/balloon-research}"
REFERENCE="${REFERENCE:-$REPO/src/eval/windeval/data/era5_heldout.zarr}"
SYNTH="${SYNTH:-$DATA_ROOT/rlhab/rlhab_synthwinds_heldout.zarr}"
DZ_SOURCE="${DZ_SOURCE:-$REPO/src/eval/windeval/data/era5_real_stage2.zarr}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/outputs/rlhab_synthwinds}"

cd "$REPO"
export PYTHONPATH="$REPO/src/eval${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

"$PYTHON" -m windeval.generators.rlhab_synthwinds.score \
  --reference "$REFERENCE" \
  --synthwinds "$SYNTH" \
  --dz-source "$DZ_SOURCE" \
  --output-dir "$OUTPUT_DIR"
