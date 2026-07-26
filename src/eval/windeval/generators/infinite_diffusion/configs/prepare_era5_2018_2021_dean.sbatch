#!/bin/bash
#SBATCH --job-name=era5-4yr-data
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=dean-compute-[01-02]
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --requeue
#SBATCH --output=era5-4yr-data-%j.out
#SBATCH --error=era5-4yr-data-%j.err
#SBATCH --get-user-env

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
PY="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
DATA_ROOT="${DATA_ROOT:-/share/dean/$USER/balloon-research/era5}"
STORE="$DATA_ROOT/era5_2018_2021.zarr"
STATS="$DATA_ROOT/era5_2018_2021_norm_stats.npz"
MANIFEST="$DATA_ROOT/era5_2018_2021.complete.json"

cd "$REPO"
mkdir -p "$DATA_ROOT"
test -f "$HOME/.cdsapirc"
"$PY" -c 'import cdsapi, cfgrib, xarray, zarr'

run_prepare() {
  local years="$1"
  shift
  child_pid=""
  forward_usr1() {
    if [[ -n "$child_pid" ]]; then
      echo "[slurm] forwarding time-limit signal to preparation process $child_pid"
      kill -USR1 "$child_pid"
    fi
  }
  trap forward_usr1 USR1

  set +e
  # YEARS deliberately expands into separate integer arguments.
  # shellcheck disable=SC2086
  "$PY" src/eval/windeval/generators/infinite_diffusion/prepare_era5_multiyear.py \
    --years $years \
    --out "$STORE" \
    --work-dir "$DATA_ROOT/grib_tmp/era5_2018_2021" \
    "$@" &
  child_pid=$!
  while true; do
    wait "$child_pid"
    status=$?
    if kill -0 "$child_pid" 2>/dev/null; then
      continue
    fi
    break
  done
  child_pid=""
  set -e
  if [[ "$status" -eq 75 ]]; then
    echo "[slurm] preparation checkpoint complete; requeueing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  return "$status"
}

run_prepare "$(seq -s ' ' 2018 2021)" --stats-out "$STATS"

"$PY" src/eval/windeval/generators/infinite_diffusion/validate_era5_multiyear.py \
  --path "$STORE" \
  --first-year 2018 \
  --last-year 2021 \
  --stats "$STATS" \
  --manifest "$MANIFEST"
