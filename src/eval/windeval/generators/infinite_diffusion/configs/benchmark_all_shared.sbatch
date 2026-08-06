#!/bin/bash
#SBATCH --job-name=all-wind-bench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dean-compute-[01-02]
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=all-wind-bench-%j.out
#SBATCH --error=all-wind-bench-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
PY="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
REFERENCE="${REFERENCE:?set REFERENCE}"
BASE_RUN="${BASE_RUN:?set BASE_RUN}"
T1_RUN="${T1_RUN:?set T1_RUN}"
T2_RUN="${T2_RUN:?set T2_RUN}"
T3_RUN="${T3_RUN:?set T3_RUN}"
CFGD_RUN="${CFGD_RUN:?set CFGD_RUN}"
BLE_DECODER="${BLE_DECODER:?set BLE_DECODER}"
OUTPUT="${OUTPUT:?set OUTPUT}"
BLE_CACHE="${BLE_CACHE:-${OUTPUT%.*}_ble_cache.npz}"

cd "$REPO"
for path in "$REFERENCE" "$BASE_RUN" "$T1_RUN" "$T2_RUN" "$T3_RUN" "$CFGD_RUN"; do
  test -d "$path"
done
test -f "$BLE_DECODER"
export PYTHONPATH="$REPO/src/eval${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c 'import msgpack, numpy, xarray; print("benchmark dependencies ready")'

"$PY" -m windeval.benchmark_shared_baselines \
  --reference "$REFERENCE" \
  --model-runs "$BASE_RUN" "$T1_RUN" "$T2_RUN" "$T3_RUN" "$CFGD_RUN" \
  --model-names \
    "Direct base model" \
    "InfiniteDiffusion T=1" \
    "InfiniteDiffusion T=2" \
    "InfiniteDiffusion T=3" \
    "CFGD" \
  --ble-decoder "$BLE_DECODER" \
  --ble-cache "$BLE_CACHE" \
  --output "$OUTPUT"

echo "[benchmark] report: $OUTPUT"
