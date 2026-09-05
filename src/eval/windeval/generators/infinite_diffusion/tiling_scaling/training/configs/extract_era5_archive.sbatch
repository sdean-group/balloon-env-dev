#!/bin/bash
#SBATCH --job-name=era5-extract
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=era5-extract-%j.out
#SBATCH --error=era5-extract-%j.err
#SBATCH --get-user-env

set -euo pipefail

ARCHIVE="${ARCHIVE:?set ARCHIVE}"
DESTINATION="${DESTINATION:?set DESTINATION}"
PY="${PYTHON:-$HOME/envs/idiff-eval-titan/bin/python}"
PARENT="$(dirname "$DESTINATION")"
STAGING="$PARENT/.era5_2023.extract.${SLURM_JOB_ID}"

test -f "$ARCHIVE"
if [[ -e "$DESTINATION" ]]; then
  echo "Destination already exists: $DESTINATION" >&2
  exit 2
fi
mkdir -p "$PARENT" "$STAGING"

cleanup() {
  if [[ -d "$STAGING" ]]; then
    rm -rf "$STAGING"
  fi
}
trap cleanup EXIT

echo "[extract] archive: $ARCHIVE"
echo "[extract] staging: $STAGING"
if [[ -f "$ARCHIVE.sha256" ]]; then
  echo "[extract] verifying SHA-256 on the compute node"
  (
    cd "$(dirname "$ARCHIVE")"
    sha256sum -c "$(basename "$ARCHIVE").sha256"
  )
fi
"$PY" - "$ARCHIVE" <<'PY'
import sys
import tarfile
from pathlib import PurePosixPath

with tarfile.open(sys.argv[1]) as archive:
    for member in archive:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive path: {member.name}")
        if member.issym() or member.islnk():
            raise SystemExit(f"archive links are not permitted: {member.name}")
print("[extract] archive path-safety check passed")
PY
tar -xf "$ARCHIVE" -C "$STAGING"

mapfile -t stores < <(find "$STAGING" -type d -name era5_2023.zarr -print)
if [[ "${#stores[@]}" -eq 1 ]]; then
  STORE="${stores[0]}"
elif [[ "${#stores[@]}" -eq 0 ]] && \
     { [[ -f "$STAGING/.zgroup" ]] || [[ -f "$STAGING/zarr.json" ]]; }; then
  STORE="$STAGING"
else
  echo "Expected exactly one era5_2023.zarr store, found ${#stores[@]}" >&2
  find "$STAGING" -maxdepth 3 -type f | head -40 >&2
  exit 2
fi

echo "[extract] validating readable Zarr metadata"
"$PY" - "$STORE" <<'PY'
import sys
import xarray as xr

path = sys.argv[1]
dataset = xr.open_zarr(path, consolidated=False)
print("sizes:", dict(dataset.sizes))
print("variables:", list(dataset.data_vars))
dataset.close()
PY

mv "$STORE" "$DESTINATION"
echo "[extract] installed: $DESTINATION"
du -sh "$DESTINATION"

if [[ "$STORE" == "$STAGING" ]]; then
  trap - EXIT
else
  cleanup
  trap - EXIT
fi
