#!/bin/bash
#SBATCH --job-name=cbottle-smoke
#SBATCH --partition=dean
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
# Pretrained cBottle-3d (HPX64, ERA5-style variables) sampled with AMIP SST conditioning for
# a few held-out timestamps; no ERA5 zarr needed. Validates the HEALPix/EDM tooling on Unicorn.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
source ~/envs/cbottle/bin/activate
export AMIP_MID_MONTH_SST=$HOME/cbottle-weights/amip_midmonth_sst.nc
export PROJECT_ROOT=$HOME/cbottle-weights
CKPT=$HOME/cbottle-weights/cBottle-3d/training-state-009856000.checkpoint
OUT=$HOME/cbottle-weights/outputs/coarse_smoke_${SLURM_JOB_ID}
mkdir -p "$OUT"; cd ~/cBottle
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,earth2grid,cbottle;print(torch.__version__, torch.cuda.is_available(), earth2grid.__version__)"
time python scripts/inference_coarse.py "$CKPT" "$OUT" --dataset amip --sample.mode sample \
  --start_time 2023-01-11T00 --end_time 2023-01-11T12 --timestamp_frequency 12h \
  --sample.min_samples 2 --sample.batch_gpu 2 --sample.seed 0 --sample.bf16
echo "--- outputs:"; ls -la "$OUT"; find "$OUT" -name "*.nc" | head -3 | while read f; do python - "$f" <<'PY'
import sys, xarray as xr
ds = xr.open_dataset(sys.argv[1]); print(ds)
PY
done
