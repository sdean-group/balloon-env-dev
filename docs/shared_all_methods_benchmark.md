# Shared all-methods wind benchmark

The final table contains:

- ERA5 self-split floor
- Direct finite base model
- InfiniteDiffusion T=1
- InfiniteDiffusion T=2, split step 9
- InfiniteDiffusion T=3, split steps 6 and 12
- Canonical Factor-Graph Diffusion
- BLE-VAE

Every conditional method uses the same 112 `(month, day, hour, seed)` samples.
Every method is scored on the same 16x16 grid at 50 km spacing and on the same
60-130 hPa pressure coordinates. Only frame zero is included; temporal evaluation
remains a separate experiment.

CFGD must generate a 64x64 source field. Its earlier 16x16 pilot cannot cover the
shared 50 km grid and must not be reused for this table.

## Unicorn

```bash
cd ~/balloon-env-dev-code
git switch main
git pull --ff-only origin main

PY="$HOME/envs/idiff-eval-titan/bin/python"
CHECKPOINT="$HOME/wind-idiff-checkpoint-eval/idiff_m2cond_latest.pt"
DATA_ROOT="/share/dean/$USER/balloon-research"
```

The checked-in submitter validates the existing T=1/T=2/T=3/reference/BLE paths,
submits direct-base and CFGD generation, and queues scoring with an `afterok`
dependency:

```bash
PYTHON="$PY" \
CHECKPOINT="$CHECKPOINT" \
DATA_ROOT="$DATA_ROOT" \
bash \
  src/eval/windeval/generators/infinite_diffusion/configs/submit_all_methods_benchmark.sh
```

The commands below show the equivalent manual sequence.

Submit the two missing condition sets:

```bash
DIRECT_JOB=$(
  CHECKPOINT="$CHECKPOINT" \
  OUTPUT_DIR="$DATA_ROOT/outputs/m2cond_conditions_direct_base" \
  NUM_SEEDS=2 \
  PYTHON="$PY" \
  sbatch --parsable \
    src/eval/windeval/generators/infinite_diffusion/configs/generate_direct_base_condition_set.sbatch
)

CFGD_JOB=$(
  CHECKPOINT="$CHECKPOINT" \
  OUTPUT_DIR="$DATA_ROOT/outputs/cfgd_conditions_full64" \
  QUERY_SIZE=64 \
  QUERY_T0=2 \
  QUERY_Y0=32 \
  QUERY_X0=32 \
  LAT_ORIGIN=25 \
  LON_ORIGIN=225 \
  NUM_SEEDS=2 \
  PYTHON="$PY" \
  sbatch --parsable \
    src/eval/windeval/generators/canonical_factor_graph/configs/generate_condition_set.sbatch
)

echo "direct-base: $DIRECT_JOB"
echo "CFGD: $CFGD_JOB"
```

Submit scoring now with a dependency on both generation jobs:

```bash
BENCH_JOB=$(
  REFERENCE="$DATA_ROOT/era5/era5_heldout_conditional.zarr" \
  BASE_RUN="$DATA_ROOT/outputs/m2cond_conditions_direct_base" \
  T1_RUN="$HOME/wind-idiff-checkpoint-eval/outputs/m2cond_conditions_t1" \
  T2_RUN="$HOME/wind-idiff-checkpoint-eval/outputs/m2cond_conditions_t2_split9" \
  T3_RUN="$DATA_ROOT/outputs/m2cond_conditions_t3" \
  CFGD_RUN="$DATA_ROOT/outputs/cfgd_conditions_full64" \
  BLE_DECODER="$DATA_ROOT/ble/offlineskies22_decoder.msgpack" \
  BLE_CACHE="$DATA_ROOT/outputs/t2_t3_vs_ble_shared_ble_cache.npz" \
  OUTPUT="$DATA_ROOT/outputs/all_methods_shared.md" \
  PYTHON="$PY" \
  sbatch --parsable \
    --dependency="afterok:$DIRECT_JOB:$CFGD_JOB" \
    src/eval/windeval/generators/infinite_diffusion/configs/benchmark_all_shared.sbatch
)

echo "benchmark: $BENCH_JOB"
```

The benchmark remains pending until both generation jobs complete successfully.
It then writes:

- `/share/dean/$USER/balloon-research/outputs/all_methods_shared.md`
- `/share/dean/$USER/balloon-research/outputs/all_methods_shared.json`
