#!/bin/bash
# Build the cBottle / earth2grid environment. Runs as a CPU-only SLURM job on a compute
# node because nvcc (needed if any wheel falls back to a source build) exists only there.
set -o pipefail
export PATH=/usr/local/cuda-12.8/bin:/usr/local/slurm/current/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PIP_CACHE_DIR=$HOME/.cache/pip TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
echo "host $(hostname)  python3 $(python3 --version)  nvcc: $(nvcc --version 2>/dev/null | tail -1)"
cd ~/envs
[ -d cbottle ] || /share/apps/software/anaconda3/bin/python3 -m venv cbottle
source cbottle/bin/activate
python -m pip install --upgrade pip wheel setuptools 2>&1 | tail -1
echo "== torch (cu128) =="
pip install "torch==2.7.*" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
echo "== earth2grid (prebuilt wheel, cu12 / torch 2.7) =="
pip install -f https://github.com/NVlabs/earth2grid/releases/expanded_assets/v2025.7.1 earth2grid 2>&1 | tail -3
echo "== cBottle (+ physicsnemo, fastgen) =="
pip install "git+https://github.com/NVlabs/cBottle.git" 2>&1 | tail -3
echo "== our stack =="
pip install healpy xarray zarr gcsfs cdsapi cfgrib eccodes pyyaml matplotlib pytest scipy "infinite-tensor>=0.3" opensimplex huggingface_hub 2>&1 | tail -2
echo "== verify =="
python - <<'PY'
import importlib
for m in ("torch", "earth2grid", "cbottle", "physicsnemo", "healpy", "xarray", "zarr", "gcsfs", "infinite_tensor"):
    try:
        mod = importlib.import_module(m); print(f"  {m:16s} OK  {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  {m:16s} FAIL {type(e).__name__}: {e}")
import torch; print("  torch cuda build:", torch.version.cuda, "| cuda available here (CPU job, expect False):", torch.cuda.is_available())
PY
echo "DONE $(date)"
