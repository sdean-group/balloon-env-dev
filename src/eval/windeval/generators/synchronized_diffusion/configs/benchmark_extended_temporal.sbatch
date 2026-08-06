#!/bin/bash
#SBATCH --job-name=sync-time-bench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=sync-time-bench-%j.out
#SBATCH --error=sync-time-bench-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
PY="${PYTHON:-python}"

PYTHONPATH="$REPO/src/eval${PYTHONPATH:+:$PYTHONPATH}" \
"$PY" -m windeval.benchmark_temporal_all_methods \
  --reference "${TEMPORAL_REFERENCE:?set TEMPORAL_REFERENCE}" \
  --model-runs \
    "${TEMP_DIRECT_RUN:?set TEMP_DIRECT_RUN}" \
    "${TEMP_T1_RUN:?set TEMP_T1_RUN}" \
    "${TEMP_T2_RUN:?set TEMP_T2_RUN}" \
    "${TEMP_T3_RUN:?set TEMP_T3_RUN}" \
    "${TEMP_CFGD_RUN:?set TEMP_CFGD_RUN}" \
    "${TEMP_SYNC_TWEEDIES_RUN:?set TEMP_SYNC_TWEEDIES_RUN}" \
    "${TEMP_OVERLAP_GUIDED_RUN:?set TEMP_OVERLAP_GUIDED_RUN}" \
    "${TEMP_CONSENSUS_RUN:?set TEMP_CONSENSUS_RUN}" \
  --model-names \
    "Direct base (independent 4h blocks)" \
    "InfiniteDiffusion T=1" \
    "InfiniteDiffusion T=2" \
    "InfiniteDiffusion T=3" \
    "CFGD" \
    "SyncTweedies adaptation" \
    "Overlap-guided adaptation" \
    "Consensus-equilibrium adaptation" \
  --seam-periods 4 2 2 2 2 2 2 2 \
  --output "${TEMPORAL_OUTPUT:?set TEMPORAL_OUTPUT}"
