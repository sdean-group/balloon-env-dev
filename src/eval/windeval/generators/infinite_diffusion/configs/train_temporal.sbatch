#!/bin/bash
#SBATCH --job-name=idiff-temporal
#SBATCH --output=/zooper2/shaurya.sen/work/balloon-env-dev/runs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#
# Train a TEMPORAL InfiniteDiffusion denoiser (Phase 4d) on Kahan — M3 (autoregressive) or
# M2 (joint spacetime). Parameterized: pass CONFIG via env (defaults to M3). The job name (-J)
# names the .out file via %x, so launch with a descriptive name:
#
#   # M3 autoregressive:
#   CONFIG=src/eval/windeval/generators/infinite_diffusion/configs/era5_temporal_m3.yaml \
#     sbatch -J idiff-m3 .../configs/train_temporal.sbatch
#   # M2 joint spacetime:
#   CONFIG=src/eval/windeval/generators/infinite_diffusion/configs/era5_temporal_m2.yaml \
#     sbatch -J idiff-m2 .../configs/train_temporal.sbatch
#
# Resumable: re-submitting picks up from <out_dir>/latest.pt (cfg.resume=true). out_dir is set
# inside the YAML (runs/idiff_m3 or runs/idiff_m2).
#
# Kahan conventions (see kahan-cluster-access): everything under /zooper2/$USER (compute node
# can't see $HOME/AFS); bare venv .venv-idiff (torch cu128 + numpy/xarray/zarr/pyyaml, NOT jax);
# login node kahanctrl has no AVX -> run compute on a GPU node via sbatch/srun; the data file
# era5_temporal.zarr must be copied into src/eval/windeval/data/ first.

set -euo pipefail
REPO="/zooper2/${USER}/work/balloon-env-dev"
cd "$REPO"
mkdir -p runs

# redirect caches off AFS (the #1 silent breakage on Kahan)
[ -f "/zooper2/${USER}/env.sh" ] && source "/zooper2/${USER}/env.sh" && cd "$REPO"

PY="$REPO/.venv-idiff/bin/python"
CONFIG="${CONFIG:-src/eval/windeval/generators/infinite_diffusion/configs/era5_temporal_m3.yaml}"
export PYTHONUNBUFFERED=1   # stream prints live to the .out file (don't block-buffer)

echo "[sbatch] training with CONFIG=$CONFIG"
# STANDALONE entry (not `-m src.eval...`) so we don't drag in the jax/gym harness stack.
"$PY" src/eval/windeval/generators/infinite_diffusion/train.py --config "$CONFIG"
