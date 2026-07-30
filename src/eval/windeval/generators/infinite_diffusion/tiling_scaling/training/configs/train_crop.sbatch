#!/bin/bash
#SBATCH --job-name=idiff-crop
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=idiff-crop-%j.out
#SBATCH --error=idiff-crop-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
PY="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
CROP="${CROP:?set CROP to 32 or 16}"
DATASET="${DATASET:?set DATASET}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"
CONFIG="src/eval/windeval/generators/infinite_diffusion/configs/era5_2023_m2cond_crop${CROP}.yaml"

if [[ "$CROP" != "32" && "$CROP" != "16" ]]; then
  echo "CROP must be 32 or 16" >&2
  exit 2
fi

cd "$REPO"
test -d "$DATASET"
test -f "$CONFIG"
mkdir -p "$OUT_DIR"

"$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

child_pid=""
forward_usr1() {
  if [[ -n "$child_pid" ]]; then
    echo "[slurm] forwarding time-limit signal to trainer process $child_pid"
    kill -USR1 "$child_pid"
  fi
}
trap forward_usr1 USR1

set +e
"$PY" src/eval/windeval/generators/infinite_diffusion/train.py \
  --config "$CONFIG" \
  --set data_path="$DATASET" out_dir="$OUT_DIR" &
child_pid=$!
while true; do
  wait "$child_pid"
  status=$?
  if kill -0 "$child_pid" 2>/dev/null; then
    continue
  fi
  break
done
set -e

if [[ "$status" -eq 75 ]]; then
  echo "[slurm] checkpoint complete; requeueing job $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

"$PY" - "$OUT_DIR/latest.pt" "$CROP" "$OUT_DIR/complete.json" <<'PY'
import json
import sys
from pathlib import Path
import torch

checkpoint_path = Path(sys.argv[1])
crop = int(sys.argv[2])
output = Path(sys.argv[3])
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if int(checkpoint["step"]) != 100_000:
    raise SystemExit(f"training stopped at step {checkpoint['step']}, expected 100000")
if int(checkpoint["cfg"]["crop"]) != crop:
    raise SystemExit("checkpoint crop does not match requested crop")
payload = {
    "status": "complete",
    "crop": crop,
    "step": int(checkpoint["step"]),
    "checkpoint": str(checkpoint_path.resolve()),
}
temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(output)
print(json.dumps(payload, indent=2))
PY
