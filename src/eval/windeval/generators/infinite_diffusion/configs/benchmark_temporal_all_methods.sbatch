#!/bin/bash
#SBATCH --job-name=wind-temp-bench
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=wind-temp-bench-%j.out
#SBATCH --error=wind-temp-bench-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
PY="${PYTHON:-python}"

PYTHONPATH="$REPO/src/eval${PYTHONPATH:+:$PYTHONPATH}" \
"$PY" -m windeval.benchmark_temporal_all_methods \
  --reference "${REFERENCE:?set REFERENCE}" \
  --model-runs \
    "${DIRECT_RUN:?set DIRECT_RUN}" \
    "${T1_RUN:?set T1_RUN}" \
    "${T2_RUN:?set T2_RUN}" \
    "${T3_RUN:?set T3_RUN}" \
    "${CFGD_RUN:?set CFGD_RUN}" \
  --model-names \
    "Direct base (independent 4h blocks)" \
    "InfiniteDiffusion T=1" \
    "InfiniteDiffusion T=2" \
    "InfiniteDiffusion T=3" \
    "CFGD" \
  --seam-periods 4 2 2 2 2 \
  --output "${OUTPUT:?set OUTPUT}"
