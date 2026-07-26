#!/bin/bash
#SBATCH --job-name=t2-t3-ble
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dean-compute-[01-02]
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=t1-vs-ble-%j.out
#SBATCH --error=t1-vs-ble-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
PY="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
REFERENCE="${REFERENCE:?set REFERENCE}"
T2_RUN="${T2_RUN:?set T2_RUN}"
T3_RUN="${T3_RUN:?set T3_RUN}"
BLE_DECODER="${BLE_DECODER:?set BLE_DECODER}"
OUTPUT="${OUTPUT:?set OUTPUT}"
BLE_CACHE="${BLE_CACHE:-${OUTPUT%.*}_ble_cache.npz}"

cd "$REPO"
test -d "$REFERENCE"
test -d "$T2_RUN"
test -d "$T3_RUN"
test -f "$BLE_DECODER"
export PYTHONPATH="$REPO/src/eval${PYTHONPATH:+:$PYTHONPATH}"

"$PY" -m windeval.benchmark_shared_baselines \
  --reference "$REFERENCE" \
  --infinite-runs "$T2_RUN" "$T3_RUN" \
  --infinite-names "InfiniteDiffusion T=2" "InfiniteDiffusion T=3" \
  --ble-decoder "$BLE_DECODER" \
  --ble-cache "$BLE_CACHE" \
  --output "$OUTPUT"

echo "[benchmark] report: $OUTPUT"
