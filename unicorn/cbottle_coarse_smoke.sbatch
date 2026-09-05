#!/bin/bash
#SBATCH --job-name=cbottle-smoke
#SBATCH --partition=dean
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# One pretrained cBottle-3d sample (HPX64, ERA5-style variables, AMIP SST conditioning, no
# ERA5 zarr needed) to validate the HEALPix/EDM tooling on Unicorn. Weights live in
# ~/cbottle-weights (downloaded from huggingface.co/nvidia/cbottle after Shaurya OKs it).
set -euo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
source ~/envs/cbottle/bin/activate
export AMIP_MID_MONTH_SST=$HOME/cbottle-weights/amip_midmonth_sst.nc
export PROJECT_ROOT=$HOME/cbottle-weights
CKPT=$HOME/cbottle-weights/cBottle-3d/training-state-009856000.checkpoint
OUT=$HOME/cbottle-weights/outputs/coarse_smoke
mkdir -p "$OUT"; cd "$(dirname "$(python -c "import cbottle,os;print(os.path.dirname(cbottle.__file__))")")"/../..
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,earth2grid,cbottle;print(torch.__version__, torch.cuda.is_available(), earth2grid.__version__)"
python -m cbottle.scripts.inference_coarse 2>/dev/null || true
# the repo ships scripts/ outside the package; use the checkout of the script from the wheel dir if absent
SCRIPT=$(python - <<'PY'
import importlib.util, pathlib, cbottle
root = pathlib.Path(cbottle.__file__).resolve().parents[2]
for c in (root / "scripts" / "inference_coarse.py", pathlib.Path.home() / "cBottle" / "scripts" / "inference_coarse.py"):
    if c.exists(): print(c); break
PY
)
echo "script: $SCRIPT"
python "$SCRIPT" --state_path "$CKPT" --output_path "$OUT" --dataset amip --mode sample \
  --start_time 2023-01-11T00 --end_time 2023-01-11T00 --timestamp_frequency h \
  --min_samples 1 --batch_gpu 1 --seed 0 --bf16
ls -la "$OUT"
