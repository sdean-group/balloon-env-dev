#!/bin/bash
#SBATCH --job-name=idiff-m1
#SBATCH --output=/zooper2/shaurya.sen/work/balloon-env-dev/runs/idiff_m1/slurm-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#
# Train the InfiniteDiffusion wind denoiser (Phase 2 / M1) on Kahan (Cooper Union EE).
# Resumable: re-submitting picks up from runs/idiff_m1/latest.pt (cfg.resume=true), so a
# 24 h wall-clock cap is fine for a longer run.
#
# Kahan conventions (see kahan-cluster-access):
#   - everything MUST live under /zooper2/$USER (compute node can't see $HOME / AFS)
#   - bare-metal venv works; Podman not required
#   - first GPU step prints ~1 min of cuDNN autotuning "slow_operation_alarm" lines (normal)
#
# One-time setup on the LOGIN node (kahanctrl, has internet):
#   cd /zooper2/$USER/work && git clone/scp the balloon-env-dev repo here
#   cd balloon-env-dev && cp <your era5_train.zarr + grib> into src/eval/windeval/data/
#   python3 -m venv .venv-idiff && source .venv-idiff/bin/activate
#   # Blackwell (sm_120) needs a recent torch (CUDA 12.8+):
#   pip install --index-url https://download.pytorch.org/whl/cu128 torch
#   pip install numpy xarray zarr pyyaml
#   # sanity (on a GPU node): srun --gres=gpu:1 --mem=8G --pty \
#   #   ./.venv-idiff/bin/python -c "import torch;print(torch.cuda.is_available())"

set -euo pipefail
REPO="/zooper2/${USER}/work/balloon-env-dev"
cd "$REPO"
mkdir -p runs/idiff_m1

# redirect caches off AFS (the #1 silent breakage on Kahan)
[ -f "/zooper2/${USER}/env.sh" ] && source "/zooper2/${USER}/env.sh" && cd "$REPO"

PY="$REPO/.venv-idiff/bin/python"
CONFIG="${CONFIG:-src/eval/windeval/generators/infinite_diffusion/configs/era5_sf.yaml}"
export PYTHONUNBUFFERED=1   # stream prints live to the .out file (don't block-buffer)

# Run the trainer as a STANDALONE script (not `-m src.eval...`) so we don't drag in the
# jax/gym harness stack — this entry only needs torch + numpy + xarray + zarr + pyyaml.
"$PY" src/eval/windeval/generators/infinite_diffusion/train.py --config "$CONFIG"
