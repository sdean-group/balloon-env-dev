"""Compare one CFGD output block with its coordinate-matched ERA5 block."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
IDIFF_DIR = HERE.parent / "infinite_diffusion"
if str(IDIFF_DIR) not in sys.path:
    sys.path.insert(0, str(IDIFF_DIR))

from compare_spacetime_to_era5 import (  # noqa: E402
    _comparison,
    _field_metrics,
    _indices_exact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CFGD wind.npz to matched ERA5")
    parser.add_argument("--era5", required=True)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-origin", default="2023-01-15T02")
    parser.add_argument("--lat-origin", type=float, default=33.0)
    parser.add_argument("--lon-origin", type=float, default=233.0)
    parser.add_argument("--dlat", type=float, default=0.25)
    parser.add_argument("--dlon", type=float, default=0.25)
    args = parser.parse_args()

    generated = np.load(args.generated)
    u = np.asarray(generated["u"], dtype=np.float32)
    v = np.asarray(generated["v"], dtype=np.float32)
    levels = np.asarray(generated["levels"])
    if u.shape != v.shape or u.ndim != 4:
        raise ValueError(f"expected matching (time, level, y, x) arrays; got {u.shape}, {v.shape}")

    target_time = np.datetime64(args.time_origin) + np.arange(u.shape[0]) * np.timedelta64(1, "h")
    target_lat = args.lat_origin + args.dlat * np.arange(u.shape[2])
    target_lon = args.lon_origin + args.dlon * np.arange(u.shape[3])
    ds = xr.open_zarr(args.era5, consolidated=False)
    indices = {
        "time": _indices_exact(ds["time"].values, target_time),
        "level": _indices_exact(ds["level"].values, levels),
        "y": _indices_exact(ds["lat"].values, target_lat),
        "x": _indices_exact(ds["lon"].values, target_lon, circular=True),
    }
    selected = ds[["u", "v"]].isel(**{
        name: xr.DataArray(index, dims=name) for name, index in indices.items()
    }).load()
    ds.close()
    ru = selected["u"].values.astype(np.float32)
    rv = selected["v"].values.astype(np.float32)

    report = {
        "generated_path": str(Path(args.generated).resolve()),
        "reference_path": str(Path(args.era5).resolve()),
        "coordinates": {
            "time": [str(target_time[0]), str(target_time[-1])],
            "latitude": [float(target_lat[0]), float(target_lat[-1])],
            "longitude_east": [float(target_lon[0]), float(target_lon[-1])],
            "levels": levels.tolist(),
        },
        "era5": _field_metrics(ru, rv),
        "cfgd": {**_field_metrics(u, v), **_comparison(u, v, ru, rv)},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
