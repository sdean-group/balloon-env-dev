#!/bin/bash
#SBATCH --job-name=idiff-10yr
#SBATCH --partition=gpu
#SBATCH --nodelist=dean-compute-02
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=idiff-10yr-%j.out
#SBATCH --error=idiff-10yr-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
PY="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the prepared ERA5 directory}"
OUT_DIR="${OUT_DIR:-$REPO/runs/idiff_m2cond_era5_10year}"
CONFIG="src/eval/windeval/generators/infinite_diffusion/configs/era5_10year_m2cond.yaml"

cd "$REPO"
test -d "$DATA_ROOT/era5_2012_2021.zarr"
test -f "$DATA_ROOT/era5_2012_2021_norm_stats.npz"
test -d "$DATA_ROOT/era5_2022.zarr"

"$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
"$PY" src/eval/windeval/generators/infinite_diffusion/train.py \
  --config "$CONFIG" \
  --set \
    data_path="$DATA_ROOT/era5_2012_2021.zarr" \
    stats_path="$DATA_ROOT/era5_2012_2021_norm_stats.npz" \
    val_data_path="$DATA_ROOT/era5_2022.zarr" \
    out_dir="$OUT_DIR"
