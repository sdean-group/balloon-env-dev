# Shared T=2, T=3, and BLE-VAE benchmark

This benchmark asks how InfiniteDiffusion T=2, InfiniteDiffusion T=3, and BLE-VAE score
when the metric
implementation, horizontal grid, pressure coordinates, ERA5 reference samples, and
generated sample count are controlled.

## Protocol

- Reference: held-out 2023 ERA5 timestamps shared by the T=2 and T=3 condition sets.
- Horizontal grid: 16x16 cells at 50 km, centered at 37.77 N, 237.58 E.
- Vertical grid: 60, 70, ..., 130 hPa.
- ERA5 and InfiniteDiffusion vertical conversion: linear interpolation from model levels
  49-66 using L137 pressure at a representative 1013.25 hPa surface pressure.
- InfiniteDiffusion sample: frame 0 from each condition file. T=2 and T=3 must contain
  identical conditions and seeds; the benchmark rejects mismatched or partial runs.
- BLE-VAE sample: central frame from one independent latent draw per condition file.
- Metrics: the same spatial spectra, marginal W1, and tail errors for every row.
- Conditional W1: reported for InfiniteDiffusion and the ERA5 floor, but N/A for BLE-VAE
  because BLE-VAE has no location or calendar conditioning.
- Temporal metrics: excluded. The BLE nine-frame axis and the model's real timestamps do
  not represent the same temporal protocol.

The 16x16 common grid gives the spectral metrics limited wavenumber resolution. The
distribution and tail metrics remain directly comparable. The pressure conversion is an
approximation; it is more defensible than pairing pressure levels with hybrid levels by
array index.

## Unicorn

Download the public BLE package without installing its old dependencies, then extract
only the decoder:

```bash
cd ~/balloon-env-dev-code
PY="$HOME/envs/idiff-eval-titan/bin/python"
DATA_ROOT="/share/dean/$USER/balloon-research"

mkdir -p "$DATA_ROOT/ble"
"$PY" -m pip download --no-deps balloon-learning-environment==1.0.1 \
  -d "$DATA_ROOT/ble"

"$PY" -c '
from pathlib import Path
import zipfile
root = Path("'"$DATA_ROOT"'/ble")
wheel = next(root.glob("balloon_learning_environment-1.0.1-*.whl"))
member = "balloon_learning_environment/models/offlineskies22_decoder.msgpack"
with zipfile.ZipFile(wheel) as archive:
    (root / "offlineskies22_decoder.msgpack").write_bytes(archive.read(member))
print(root / "offlineskies22_decoder.msgpack")
'
```

Generate the full T=3 condition set first. The default three-level split schedule is
steps 6 and 12:

```bash
CHECKPOINT="$HOME/wind-idiff-checkpoint-eval/idiff_m2cond_latest.pt" \
OUTPUT_DIR="$DATA_ROOT/outputs/m2cond_conditions_t3" \
OUTER_DEPTH=3 \
NUM_SEEDS=2 \
PYTHON="$PY" \
sbatch \
  --time=06:00:00 \
  --nodelist='dean-compute-[01-02]' \
  src/eval/windeval/generators/infinite_diffusion/configs/generate_condition_set.sbatch
```

After that job completes, submit the CPU-only scoring job:

```bash
REFERENCE="$DATA_ROOT/era5/era5_heldout_conditional.zarr" \
T2_RUN="$HOME/wind-idiff-checkpoint-eval/outputs/m2cond_conditions_t2_split9" \
T3_RUN="$DATA_ROOT/outputs/m2cond_conditions_t3" \
BLE_DECODER="$DATA_ROOT/ble/offlineskies22_decoder.msgpack" \
OUTPUT="$DATA_ROOT/outputs/t2_t3_vs_ble_shared.md" \
PYTHON="$PY" \
sbatch \
  src/eval/windeval/generators/infinite_diffusion/configs/benchmark_t2_t3_ble_shared.sbatch
```

The job writes `t2_t3_vs_ble_shared.md`, `t2_t3_vs_ble_shared.json`, and a reusable BLE
sample cache named `t2_t3_vs_ble_shared_ble_cache.npz`.
