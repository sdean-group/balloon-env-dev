#!/bin/bash
#SBATCH --job-name=stage1-hpx32
#SBATCH --partition=dean
#SBATCH --nodelist=dean-compute-02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/%x-%j.out
# Stage 1 coarse generator. The trainer checkpoints on SIGUSR1 (sent 5 min before the limit)
# and exits 75; we then requeue ourselves so the run continues from latest.pt.
source ~/envs/cbottle/bin/activate
cd ~/balloon-env-dev
CONFIG=${CONFIG:-unicorn/configs/stage1_hpx32.yaml}
python src/eval/windeval/hpx/train_coarse.py --config "$CONFIG" "$@" &
PID=$!
trap 'kill -USR1 $PID' USR1
wait $PID; RC=$?
if [ "$RC" -eq 75 ]; then echo "[sbatch] paused at time limit -> requeue"; scontrol requeue "$SLURM_JOB_ID"; fi
exit $RC
