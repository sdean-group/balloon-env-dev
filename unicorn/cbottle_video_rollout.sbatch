#!/bin/bash
#SBATCH --job-name=cbottle-video
#SBATCH --partition=dean
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# One wind rollout from NVIDIA's pretrained cBottle-video, used exactly as released.
# The model makes 12 frames (6 h apart = 72 h) at a time and chains clips to reach DAYS days.
# Needs only the AMIP sea-surface-temperature file; no ERA5.
#   sbatch --export=ALL,START=2023-01-10,DAYS=14,SEED=0,TAG=jan_s0 cbottle_video_rollout.sbatch
set -uo pipefail
START=${START:-2023-01-10}; DAYS=${DAYS:-4}; SEED=${SEED:-0}; TAG=${TAG:-smoke}
END=$(date -u -d "$START + $DAYS days" +%Y-%m-%d)
export PATH=/usr/local/cuda-12.8/bin:$PATH
source ~/envs/cbottle/bin/activate
export AMIP_MID_MONTH_SST=$HOME/cbottle-weights/amip_midmonth_sst.nc
export PROJECT_ROOT=$HOME/cbottle-weights
CKPT=$HOME/cbottle-weights/cBottle-video/cBottle-video.checkpoint
OUT=$HOME/cbottle-weights/outputs/video_${TAG}_${SLURM_JOB_ID}
mkdir -p "$OUT"; cd ~/cBottle
echo "rollout $START -> $END ($DAYS days), seed $SEED, out $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# duration = DAYS+1 so the date range holds exactly one rollout start
time python scripts/inference_coarse_video.py "$CKPT" "$OUT" --dataset amip \
  --start_time "$START" --end_time "$END" \
  --sample.mode sample --sample.autoregression_duration $((DAYS + 1)) \
  --sample.seed "$SEED" --sample.bf16
echo "--- outputs:"; ls -la "$OUT"
f=$(find "$OUT" -name "*.nc" | head -1)
[ -n "$f" ] && python - "$f" <<'PY'
import sys, xarray as xr, numpy as np
ds = xr.open_dataset(sys.argv[1])
print(ds)
for v in ("U200", "V200", "U50"):
    if v in ds: print(v, "mean %.2f std %.2f" % (float(ds[v].mean()), float(ds[v].std())))
PY
