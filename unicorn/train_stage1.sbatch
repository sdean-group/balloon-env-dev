#!/bin/bash
#SBATCH --job-name=stage1-hpx32
#SBATCH --partition=dean
#SBATCH --nodelist=dean-compute-02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/%x-%j.out
# Stage 1 coarse generator. Slurm sends USR1 to this shell 5 min before the limit; we forward
# it to the trainer, which checkpoints and exits 75, and then requeue ourselves so the next
# leg resumes from latest.pt. Bash's `wait` returns 128+signal the moment the trap fires, so
# we must wait again for the trainer's real exit status -- the first version of this script
# exited at that point and Slurm killed the trainer before it could checkpoint (job 479348).
source ~/envs/cbottle/bin/activate
cd ~/balloon-env-dev
CONFIG=${CONFIG:-unicorn/configs/stage1_hpx32.yaml}
python src/eval/windeval/hpx/train_coarse.py --config "$CONFIG" "$@" &
PID=$!
trap 'echo "[sbatch] USR1 -> trainer $PID"; kill -USR1 $PID' USR1
wait $PID; RC=$?
while [ "$RC" -gt 128 ] && kill -0 "$PID" 2>/dev/null; do wait $PID; RC=$?; done
echo "[sbatch] trainer exited $RC"
if [ "$RC" -eq 75 ]; then
    echo "[sbatch] paused at time limit -> requeue"
    scontrol requeue "$SLURM_JOB_ID" && exit 0
fi
exit $RC
