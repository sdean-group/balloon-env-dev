"""Temporal physical-consistency metrics for the poster row.

The board reports N/A for `ble_vae` and every diffusion row because
`temporal.MIN_FRAMES = 16` and neither produced 16 contiguous frames. Two changes make
the row scorable; both are recorded here because they change what the numbers mean.

1. **ble_vae time axis (repair, not padding).** Its artifact stores time as raw step
   indices 0..8. Per docs/conditional-base-changes.md those 9 slices span 48 h
   (vae.py time_horizon_hours=48) -- native 6-HOURLY spacing. We stamp real datetimes at
   6 h so the axis says what BLE actually means. We do NOT repeat frames to reach 16:
   duplicated frames have zero temporal variance, which would drive the temporal PSD and
   the tracer dispersion toward artefacts of the padding rather than properties of BLE.

2. **MIN_FRAMES lowered to 8, applied to every row.** BLE physically has 9 slices, so 8
   is the largest common window that admits it. Every row -- floor included -- is
   re-scored at the same setting, so the column stays internally comparable.

Cross-check: `white noise` has a single 24-frame segment, so it qualifies under both the
old (16) and new (8) settings and its numbers must reproduce the published
2.40 / 0.39 / 0.38. That is the pipeline's correctness test and it is asserted below.

Comparability caveat that survives all of this: metrics are computed in physical hours,
but BLE resolves only >=12 h periods (6 h sampling) while ERA5 and the diffusion runs are
hourly. SR_time is therefore evaluated on each pair's overlapping frequency band.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from .. import artifact
from ..metrics import temporal
from ..reference import split

DATA = Path(__file__).resolve().parents[1] / "data"

PUBLISHED_WHITE_NOISE = {"SR_time": 2.40, "disp log-MSD RMSE": 0.39,
                         "final spread ratio": 0.38}
ROW_ORDER = ("SR_time", "disp log-MSD RMSE", "final spread ratio")


def ble_with_real_time(path: Path, dt_hours: int = 6) -> xr.Dataset:
    """BLE artifact with its integer step axis replaced by true 6-hourly datetimes."""
    ds = artifact.read(path)
    n = ds.sizes["time"]
    t0 = np.datetime64("2023-01-11T00", "h")
    times = t0 + np.arange(n) * np.timedelta64(dt_hours, "h")
    return ds.assign_coords(time=("time", times.astype("datetime64[ns]")))


def match_levels(ref: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
    """Reference re-indexed to the pred's nearest pressure levels (ble is 10 hPa levels)."""
    rp = np.asarray(ref["level"].values, dtype=float)
    idx = [int(np.argmin(np.abs(rp - float(p)))) for p in np.asarray(pred["level"].values)]
    return ref.isel(level=idx)


def score(pred: xr.Dataset, ref: xr.Dataset) -> dict:
    out = {}
    if not (temporal.has_time(pred) and temporal.has_time(ref)):
        return {k: float("nan") for k in ROW_ORDER}
    sr, _ = temporal.temporal_spectral_residual(pred, ref)
    out.update(sr)
    disp, _ = temporal.dispersion_compare(pred, ref)
    out.update(disp)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--diffusion-dir", default=None,
                    help="directory of idiff_temporal_*.zarr from gen_temporal_sequences.py")
    args = ap.parse_args()

    temporal.MIN_FRAMES = int(args.min_frames)
    print(f"MIN_FRAMES = {temporal.MIN_FRAMES}\n")

    # Reproduce the board's exact inputs. The anchors are NOT the stored anchor_*.zarr
    # files: benchmark._anchor_rows derives them from the 4-hourly-subsampled half A,
    # and one shared rng means phase-shuffle must be built first for white noise to
    # land on the published draw. The floor scores that same subsample against
    # FULL-hourly half B. Reading the stored zarrs instead gives different numbers.
    from ..benchmark import SPATIAL_STRIDE_H, _anchor_rows
    from ..reference import build_heldout

    ref = artifact.read(build_heldout())
    half_a, half_b = split(ref)
    a_sp = half_a.isel(time=slice(0, None, SPATIAL_STRIDE_H)).compute()
    rows: dict[str, dict] = {}

    print("scoring self-split floor …", flush=True)
    rows["ERA5 floor"] = score(a_sp, half_b)

    print("scoring white noise …", flush=True)
    rows["white noise"] = score(_anchor_rows(a_sp)["white noise"], ref)

    print("scoring ble_vae (6-hourly axis) …", flush=True)
    ble = ble_with_real_time(DATA / "ble_vae_0.zarr")
    rows["BLE-VAE"] = score(ble, match_levels(ref, ble))

    if args.diffusion_dir:
        d = Path(args.diffusion_dir)
        parts = sorted(d.glob("idiff_temporal_*.zarr"))
        if parts:
            print(f"scoring diffusion ({len(parts)} tiled sequences) …", flush=True)
            ds = xr.concat([artifact.read(p) for p in parts], dim="time")
            segs = temporal.contiguous_segments(ds)
            print(f"  contiguous segments: {[s.stop - s.start for s in segs]}")
            rows["our diffusion"] = score(ds, ref)
        else:
            print(f"no idiff_temporal_*.zarr in {d}")

    print("\n" + "-" * 68)
    hdr = f"{'row':<16}" + "".join(f"{m:>18}" for m in ROW_ORDER)
    print(hdr)
    print("-" * 68)
    for name, vals in rows.items():
        line = f"{name:<16}"
        for m in ROW_ORDER:
            v = vals.get(m, float("nan"))
            line += f"{'N/A':>18}" if not np.isfinite(v) else f"{v:>18.2f}"
        print(line)
    print("-" * 68)

    wnr = rows["white noise"]
    print("\nvalidation vs published white-noise row (MIN_FRAMES=16):")
    ok = True
    for m, want in PUBLISHED_WHITE_NOISE.items():
        got = wnr.get(m, float("nan"))
        hit = np.isfinite(got) and abs(got - want) < 0.005
        ok &= hit
        print(f"  {m:<22} published {want:5.2f}   recomputed {got:6.3f}   "
              f"{'MATCH' if hit else 'DIFFERS'}")
    print("pipeline reproduces the published row." if ok else
          "WARNING: white noise did not reproduce -- treat other rows with suspicion.")


if __name__ == "__main__":
    main()
