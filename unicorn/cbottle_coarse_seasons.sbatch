#!/bin/bash
#SBATCH --job-name=cbottle-seasons
#SBATCH --partition=dean
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
set -uo pipefail
source ~/envs/cbottle/bin/activate
export AMIP_MID_MONTH_SST=$HOME/cbottle-weights/amip_midmonth_sst.nc PROJECT_ROOT=$HOME/cbottle-weights
CKPT=$HOME/cbottle-weights/cBottle-3d/training-state-009856000.checkpoint
cd ~/cBottle
for M in 01 07; do
  OUT=$HOME/cbottle-weights/outputs/coarse_2023${M}_${SLURM_JOB_ID}; mkdir -p "$OUT"
  python scripts/inference_coarse.py "$CKPT" "$OUT" --dataset amip --sample.mode sample \
    --start_time 2023-${M}-11T00 --end_time 2023-${M}-14T12 --timestamp_frequency 12h \
    --sample.min_samples 8 --sample.batch_gpu 4 --sample.seed 1 --sample.bf16 2>&1 | grep -E "Writing time|it/s\]" | tail -3
  echo "done $OUT"
done
