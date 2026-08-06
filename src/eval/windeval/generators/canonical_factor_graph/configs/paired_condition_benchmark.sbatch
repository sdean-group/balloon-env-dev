#!/bin/bash
#SBATCH --job-name=cfgd-vs-t1-s5
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dean-compute-02
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=cfgd-vs-t1-s5-%j.out
#SBATCH --error=cfgd-vs-t1-s5-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
PY="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT}"
REFERENCE="${REFERENCE:?set REFERENCE}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
NUM_SEEDS="${NUM_SEEDS:-5}"

INFINITE_DIR="$OUTPUT_ROOT/infinite_t1"
CFGD_DIR="$OUTPUT_ROOT/cfgd"
REPORT="$OUTPUT_ROOT/cfgd_vs_infinite_t1.md"

cd "$REPO"
mkdir -p "$OUTPUT_ROOT"
test -f "$CHECKPOINT"
test -d "$REFERENCE"
"$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

"$PY" src/eval/windeval/generators/infinite_diffusion/generate_condition_set.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$INFINITE_DIR" \
  --device cuda \
  --num-steps 18 \
  --outer-depth 1 \
  --num-seeds "$NUM_SEEDS"

"$PY" src/eval/windeval/generators/canonical_factor_graph/generate_condition_set.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$CFGD_DIR" \
  --device cuda \
  --num-steps 18 \
  --num-seeds "$NUM_SEEDS" \
  --window-batch-size 1

export PYTHONPATH="$REPO/src/eval${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m windeval.generators.canonical_factor_graph.benchmark_condition_sets \
  --reference "$REFERENCE" \
  --runs "$INFINITE_DIR" "$CFGD_DIR" \
  --names "InfiniteDiffusion T=1" "CFGD" \
  --output "$REPORT"

echo "[benchmark] report: $REPORT"
