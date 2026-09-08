#!/bin/bash
#SBATCH --job-name=eval-stage1
#SBATCH --partition=dean
#SBATCH --nodelist=dean-compute-02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# Stage 1 gate on every snapshot of a run (plus best.pt), same settings for all so the
# numbers form a curve. RUN=<run dir> CKPTS="step_50000.pt ..." (default: all snapshots + best)
# CHURNS="0 2 5 8" (EDM S_churn values; non-zero ones write to eval_<name>_churn<c>).
source ~/envs/cbottle/bin/activate
cd ~/balloon-env-dev
RUN=${RUN:-/scratch/sps252/runs/stage1_hpx32}
HELDOUT=${HELDOUT:-/scratch/sps252/era5_hpx_heldout_2023}
SEEDS=${SEEDS:-2}; BPM=${BPM:-6}; STEPS=${STEPS:-18}; CHURNS=${CHURNS:-0}
if [ -z "$CKPTS" ]; then CKPTS="$(cd $RUN && ls step_*.pt | sort -t_ -k2 -n) best.pt"; fi
for ck in $CKPTS; do for churn in $CHURNS; do
    name=${ck%.pt}; suffix=""; [ "$churn" != "0" ] && suffix="_churn$churn"; [ "$STEPS" != "18" ] && suffix="${suffix}_s$STEPS"
    echo "################ gate: $name churn=$churn ################"
    python src/eval/windeval/hpx/eval_coarse.py --ckpt $RUN/$ck --heldout $HELDOUT --churn $churn \
        --blocks-per-month $BPM --seeds $SEEDS --steps $STEPS --device cuda --out $RUN/eval_$name$suffix "$@"
done; done
echo "[sbatch] done"
