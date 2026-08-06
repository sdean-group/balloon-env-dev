#!/bin/bash
#SBATCH --job-name=rlhab-prepare
#SBATCH --output=rlhab-prepare-%j.out
#SBATCH --error=rlhab-prepare-%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00

set -euo pipefail
REPO="${REPO:-$HOME/balloon-env-dev-code}"
PYTHON="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
DATA_ROOT="${DATA_ROOT:-/share/dean/$USER/balloon-research}"
REFERENCE="${REFERENCE:-$REPO/src/eval/windeval/data/era5_heldout.zarr}"
RAW_DIR="${RAW_DIR:-$DATA_ROOT/rlhab/raw_soundings_2023}"
SYNTH="${SYNTH:-$DATA_ROOT/rlhab/rlhab_synthwinds_heldout.zarr}"

cd "$REPO"
export PYTHONPATH="$REPO/src/eval${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

"$PYTHON" -c 'import bs4, numpy, pandas, requests, scipy, xarray, zarr'
"$PYTHON" -m windeval.generators.rlhab_synthwinds.download \
  --output "$RAW_DIR"
"$PYTHON" -m windeval.generators.rlhab_synthwinds.generate \
  --raw-dir "$RAW_DIR" \
  --reference "$REFERENCE" \
  --output "$SYNTH"
