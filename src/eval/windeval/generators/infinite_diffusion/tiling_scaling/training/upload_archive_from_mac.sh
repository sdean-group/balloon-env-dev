#!/bin/bash
# Validate and resumably upload the completed ERA5 archive from macOS to Unicorn.
set -euo pipefail

ARCHIVE="${1:-$HOME/Downloads/era5_2023.zarr.tar}"
REMOTE="${REMOTE:-unicorn}"
REMOTE_ROOT="${REMOTE_ROOT:-/share/dean/rs2656/balloon-research}"
REMOTE_ARCHIVE="$REMOTE_ROOT/incoming/era5_2023.zarr.tar"

test -f "$ARCHIVE" || {
  echo "Archive not found: $ARCHIVE" >&2
  exit 2
}
case "$ARCHIVE" in
  *.download/*|*.download)
    echo "Refusing an archive that is still inside a .download path: $ARCHIVE" >&2
    echo "Wait for the download to complete and move/rename it first." >&2
    exit 2
    ;;
esac

echo "[local] checking complete tar archive"
tar -tf "$ARCHIVE" >/dev/null
python3 - "$ARCHIVE" <<'PY'
import sys
import tarfile
from pathlib import PurePosixPath

archive = sys.argv[1]
markers = 0
with tarfile.open(archive) as stream:
    for member in stream:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive path: {member.name}")
        if member.issym() or member.islnk():
            raise SystemExit(f"archive links are not permitted: {member.name}")
        if "era5_2023.zarr" in path.parts or path.name in {".zgroup", "zarr.json"}:
            markers += 1
if markers == 0:
    raise SystemExit("archive does not appear to contain an era5_2023.zarr store")
print("archive structure check passed")
PY

echo "[remote] creating incoming directory"
ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT/incoming' '$REMOTE_ROOT/era5'"

echo "[upload] resumable transfer to $REMOTE:$REMOTE_ARCHIVE"
progress_args=(--progress)
if rsync --help 2>&1 | grep -q -- '--info'; then
  progress_args=(--info=progress2)
fi
rsync -ah --partial "${progress_args[@]}" "$ARCHIVE" "$REMOTE:$REMOTE_ARCHIVE"

echo "[verify] writing SHA-256 sidecar for compute-node verification"
local_sha="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
sidecar="$(mktemp)"
trap 'rm -f "$sidecar"' EXIT
printf '%s  %s\n' "$local_sha" "$(basename "$REMOTE_ARCHIVE")" > "$sidecar"
rsync -ah "$sidecar" "$REMOTE:$REMOTE_ARCHIVE.sha256"
echo "SHA-256: $local_sha"

echo
echo "Upload complete. Remote archive:"
ssh "$REMOTE" "ls -lh '$REMOTE_ARCHIVE'"
