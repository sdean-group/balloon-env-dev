#!/bin/bash
#SBATCH --job-name=idiff-temporal
#SBATCH --output=/zooper2/shaurya.sen/work/balloon-env-dev/runs/%x-%j.out
#SBATCH --gres=gpu:gpu48:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#
# Train a TEMPORAL InfiniteDiffusion denoiser (Phase 4d) on Kahan — M3 (autoregressive) or
# M2 (joint spacetime). Parameterized: pass CONFIG via env (defaults to M3). The job name (-J)
# names the .out file via %x, so launch with a descriptive name.
#
# Kahan GPUs are MIG-partitioned into memory POOLS (see wiki kahan-info#gpu_resource_types):
# gpu:gpu24 (24GB, x8), gpu:gpu48 (48GB, x2), gpu:gpu96 (96GB, x2). You MUST request a pool via
# --gres=gpu:<pool>:<count> (a bare `--gres=gpu:1` will NOT map to a MIG instance and the job dies
# at CUDA init). SLURM sets CUDA_VISIBLE_DEVICES to the allocated instance automatically — do NOT
# set it yourself. Default here is gpu48:1 (fits M3's ~28GB at batch 128); override per job:
#
#   # M3 autoregressive (~28GB -> 48GB pool):
#   CONFIG=src/eval/windeval/generators/infinite_diffusion/configs/era5_temporal_m3.yaml \
#     sbatch --gres=gpu:gpu48:1 --requeue -J idiff-m3 .../configs/train_temporal.sbatch
#   # M2 joint spacetime (heavier -> 96GB pool):
#   CONFIG=src/eval/windeval/generators/infinite_diffusion/configs/era5_temporal_m2.yaml \
#     sbatch --gres=gpu:gpu96:1 --requeue -J idiff-m2 .../configs/train_temporal.sbatch
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
