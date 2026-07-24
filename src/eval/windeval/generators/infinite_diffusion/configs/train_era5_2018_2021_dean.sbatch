#!/bin/bash
#SBATCH --job-name=idiff-4yr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dean-compute-02
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=idiff-4yr-%j.out
#SBATCH --error=idiff-4yr-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
PY="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
DATA_ROOT="${DATA_ROOT:-/share/dean/$USER/balloon-research/era5}"
OUT_DIR="${OUT_DIR:-/share/dean/$USER/balloon-research/runs/idiff_m2cond_era5_2018_2021}"
CONFIG="src/eval/windeval/generators/infinite_diffusion/configs/era5_2018_2021_m2cond.yaml"

cd "$REPO"
test -f "$DATA_ROOT/era5_2018_2021.complete.json"
test -f "$DATA_ROOT/era5_2022.complete.json"
test -d "$DATA_ROOT/era5_2018_2021.zarr"
test -f "$DATA_ROOT/era5_2018_2021_norm_stats.npz"
test -d "$DATA_ROOT/era5_2022.zarr"
mkdir -p "$OUT_DIR"

"$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

child_pid=""
forward_usr1() {
  if [[ -n "$child_pid" ]]; then
    echo "[slurm] forwarding time-limit signal to trainer process $child_pid"
    kill -USR1 "$child_pid"
  fi
}
trap forward_usr1 USR1

set +e
"$PY" src/eval/windeval/generators/infinite_diffusion/train.py \
  --config "$CONFIG" \
  --set \
    data_path="$DATA_ROOT/era5_2018_2021.zarr" \
    stats_path="$DATA_ROOT/era5_2018_2021_norm_stats.npz" \
    val_data_path="$DATA_ROOT/era5_2022.zarr" \
    out_dir="$OUT_DIR" &
child_pid=$!
while true; do
  wait "$child_pid"
  status=$?
  if kill -0 "$child_pid" 2>/dev/null; then
    continue
  fi
  break
done
set -e

if [[ "$status" -eq 75 ]]; then
  echo "[slurm] checkpoint complete; requeueing job $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
