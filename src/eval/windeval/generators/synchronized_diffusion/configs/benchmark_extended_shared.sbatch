#!/bin/bash
#SBATCH --job-name=sync-space-bench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=sync-space-bench-%j.out
#SBATCH --error=sync-space-bench-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
PY="${PYTHON:-python}"
export PYTHONPATH="$REPO/src/eval${PYTHONPATH:+:$PYTHONPATH}"

"$PY" -m windeval.benchmark_shared_baselines \
  --reference "${SPATIAL_REFERENCE:?set SPATIAL_REFERENCE}" \
  --model-runs \
    "${DIRECT_RUN:?set DIRECT_RUN}" \
    "${T1_RUN:?set T1_RUN}" \
    "${T2_RUN:?set T2_RUN}" \
    "${T3_RUN:?set T3_RUN}" \
    "${CFGD_RUN:?set CFGD_RUN}" \
    "${SYNC_TWEEDIES_RUN:?set SYNC_TWEEDIES_RUN}" \
    "${OVERLAP_GUIDED_RUN:?set OVERLAP_GUIDED_RUN}" \
    "${CONSENSUS_RUN:?set CONSENSUS_RUN}" \
  --model-names \
    "Direct base model" \
    "InfiniteDiffusion T=1" \
    "InfiniteDiffusion T=2" \
    "InfiniteDiffusion T=3" \
    "CFGD" \
    "SyncTweedies adaptation" \
    "Overlap-guided adaptation" \
    "Consensus-equilibrium adaptation" \
  --ble-decoder "${BLE_DECODER:?set BLE_DECODER}" \
  --ble-cache "${BLE_CACHE:?set BLE_CACHE}" \
  --output "${SPATIAL_OUTPUT:?set SPATIAL_OUTPUT}"
