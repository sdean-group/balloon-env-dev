"""Download the held-out conditional ERA5 reference from public ARCO-ERA5.

The full benchmark reference is hourly and expensive to read from ARCO's global chunks.
The conditional M2 benchmark only needs days 8--14 at 00/12 UTC in Jan/Apr/Jul/Oct.
Those are the exact (month, hour) condition groups used by ``benchmark.py``.

The download is resumable by month.  Each completed monthly part is a normal local Zarr
store; after all four exist they are concatenated into ``era5_heldout_conditional.zarr``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

ARCO_URL = "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"
MONTHS = (1, 4, 7, 10)
DAYS = tuple(range(8, 15))
HOURS = (0, 12)


def _times(month: int) -> np.ndarray:
    return np.asarray(
        [np.datetime64(f"2023-{month:02d}-{day:02d}T{hour:02d}", "ns")
         for day in DAYS for hour in HOURS]
    )


def _encoding(ds: xr.Dataset) -> dict:
    chunks = (1, ds.sizes["level"], ds.sizes["y"], ds.sizes["x"])
    return {name: {"chunks": chunks} for name in ("u", "v")}


def _write_zarr(ds: xr.Dataset, path: Path) -> None:
    kwargs = dict(mode="w", consolidated=False, encoding=_encoding(ds))
    try:
        ds.to_zarr(path, zarr_format=2, **kwargs)
    except TypeError as exc:
        if "zarr_format" not in str(exc):
            raise
        ds.to_zarr(path, zarr_version=2, **kwargs)


def download(output: Path, *, workers: int = 4) -> Path:
    parts = output.with_suffix(output.suffix + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    source = xr.open_zarr(
        ARCO_URL,
        consolidated=True,
        chunks={},
        storage_options={"token": "anon"},
    )

    for month in MONTHS:
        part = parts / f"2023-{month:02d}.zarr"
        if part.exists():
            existing = xr.open_zarr(part, consolidated=False)
            if existing.sizes.get("time") == len(DAYS) * len(HOURS):
                print(f"[arco] {part.name}: complete, skipping", flush=True)
                continue
            raise RuntimeError(f"incomplete existing part: {part}")

        print(f"[arco] downloading 2023-{month:02d}, days 8-14 at 00/12 UTC", flush=True)
        selected = source.sel(
            time=_times(month),
            hybrid=slice(49, 66),
            latitude=slice(55, 25),
            longitude=slice(225, 255),
        )[["u_component_of_wind", "v_component_of_wind"]]
        selected = selected.rename(
            {
                "u_component_of_wind": "u",
                "v_component_of_wind": "v",
                "hybrid": "level",
                "latitude": "y",
                "longitude": "x",
            }
        ).rename_vars({"y": "lat", "x": "lon"})
        selected = selected.transpose("time", "level", "y", "x")
        selected = selected.load(scheduler="threads", num_workers=workers)
        if not bool(np.isfinite(selected[["u", "v"]].to_array().values).all()):
            raise RuntimeError(f"non-finite values in {month=}")
        selected.attrs = {
            "format_version": "v1",
            "generator": json.dumps({"name": "era5_arco_heldout_conditional"}),
            "capabilities": json.dumps(
                {"extent": "bounded", "temporally_evolving": True}
            ),
            "conditioning": json.dumps(
                {"year": 2023, "month": month, "days": [8, 14], "hours": [0, 12]}
            ),
            "model_levels": list(range(49, 67)),
            "units": "u,v:m/s",
            "source": ARCO_URL,
        }
        _write_zarr(selected, part)
        print(f"[arco] wrote {part}", flush=True)

    datasets = [xr.open_zarr(parts / f"2023-{month:02d}.zarr", consolidated=False)
                for month in MONTHS]
    combined = xr.concat(datasets, dim="time").sortby("time")
    if output.exists():
        existing = xr.open_zarr(output, consolidated=False)
        if existing.sizes.get("time") == len(MONTHS) * len(DAYS) * len(HOURS):
            print(f"[arco] combined reference already complete: {output}", flush=True)
            return output
        raise RuntimeError(f"incomplete existing output: {output}")
    _write_zarr(combined, output)
    print(f"[arco] complete: {output}", flush=True)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "era5_heldout_conditional.zarr",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    download(args.output, workers=args.workers)


if __name__ == "__main__":
    main()
