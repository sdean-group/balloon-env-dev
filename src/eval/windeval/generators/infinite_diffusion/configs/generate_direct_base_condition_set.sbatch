#!/bin/bash
#SBATCH --job-name=direct-base-set
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dean-compute-[01-02]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=direct-base-set-%j.out
#SBATCH --error=direct-base-set-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
PY="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"

cd "$REPO"
test -f "$CHECKPOINT"
mkdir -p "$OUTPUT_DIR"
"$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

"$PY" \
  src/eval/windeval/generators/infinite_diffusion/generate_direct_base_condition_set.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  --num-steps "${NUM_STEPS:-18}" \
  --num-seeds "${NUM_SEEDS:-2}"
