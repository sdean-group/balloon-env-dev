#!/bin/bash
#SBATCH --job-name=cbottle-video
#SBATCH --partition=dean
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# One wind rollout from NVIDIA's pretrained cBottle-video: released weights, NVIDIA's rollout script,
# plus a one-line fix to its clip chaining (see cbottle_video_patched.py). PATCH=0 runs it unfixed.
# The model makes 12 frames (6 h apart = 72 h) at a time and chains clips to reach DAYS days.
# Needs only the AMIP sea-surface-temperature file; no ERA5.
#   sbatch --export=ALL,START=2021-01-10,DAYS=14,SEED=0,TAG=jan_s0 cbottle_video_rollout.sbatch
# START must be 1940..2021 (the script's calendar). SEED=none leaves the seed unset. EXTRA = extra flags.
set -uo pipefail
START=${START:-2021-01-10}; DAYS=${DAYS:-4}; SEED=${SEED:-0}; TAG=${TAG:-smoke}; EXTRA=${EXTRA:-}; PATCH=${PATCH:-1}
RUNNER=$HOME/balloon-env-dev/unicorn/cbottle_video_patched.py; [ "$PATCH" = "0" ] && RUNNER=scripts/inference_coarse_video.py
SEEDFLAG="--sample.seed $SEED"; [ "$SEED" = "none" ] && SEEDFLAG=""
END=$(date -u -d "$START + $DAYS days" +%Y-%m-%d)
export PATH=/usr/local/cuda-12.8/bin:$PATH
source ~/envs/cbottle/bin/activate
export AMIP_MID_MONTH_SST=$HOME/cbottle-weights/amip_midmonth_sst.nc
export PROJECT_ROOT=$HOME/cbottle-weights
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))   # own port, so several rollouts can share a node
CKPT=$HOME/cbottle-weights/cBottle-video/cBottle-video.checkpoint
OUT=$HOME/cbottle-weights/outputs/video_${TAG}_${SLURM_JOB_ID}
mkdir -p "$OUT"; cd ~/cBottle
echo "rollout $START -> $END ($DAYS days), seed $SEED, extra [$EXTRA], chaining fix $PATCH, out $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# duration = DAYS+1 so the date range holds exactly one rollout start
time python "$RUNNER" "$CKPT" "$OUT" --dataset amip \
  --start_time "$START" --end_time "$END" \
  --sample.mode sample --sample.autoregression_duration $((DAYS + 1)) \
  $SEEDFLAG --sample.bf16 $EXTRA
echo "--- outputs:"; ls -la "$OUT"
f=$(find "$OUT" -name "*.nc" | head -1)
# Health check: wind variability per frame, and how similar each frame is to the one before.
# A healthy rollout keeps U200std steady (~18) and corr ~0.98 across clip boundaries (every 11 frames).
[ -n "$f" ] && python - "$f" <<'PY'
import sys, xarray as xr, numpy as np
ds = xr.open_dataset(sys.argv[1]); prev = None
print("frame time          U200std U200max U850std corr_with_prev")
for i in range(ds.sizes["time"]):
    u = ds["U200"].values[i]; c = np.corrcoef(u, prev)[0, 1] if prev is not None else float("nan")
    print("%4d  %s %7.2f %7.1f %7.2f %7.3f" % (i, str(ds.time.values[i])[:13], u.std(), np.abs(u).max(), ds["U850"].values[i].std(), c)); prev = u
PY
