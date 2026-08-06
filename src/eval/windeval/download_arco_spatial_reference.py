"""Download the compact 4-hourly ERA5 reference used by spatial evaluation v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from .download_arco_conditional_reference import ARCO_URL, _write_zarr

MONTHS = (1, 4, 7, 10)
DAYS = tuple(range(8, 15))
HOURS = tuple(range(0, 24, 4))


def _times(month: int, day: int) -> np.ndarray:
    return np.asarray([
        np.datetime64(f"2023-{month:02d}-{day:02d}T{hour:02d}", "ns")
        for hour in HOURS
    ])


def _standardize(frame: xr.Dataset) -> xr.Dataset:
    return (
        frame.rename({
            "u_component_of_wind": "u",
            "v_component_of_wind": "v",
            "hybrid": "level",
            "latitude": "y",
            "longitude": "x",
        })
        .rename_vars({"y": "lat", "x": "lon"})
        .transpose("time", "level", "y", "x")
    )


def download(output: Path, *, temporal_parts: Path | None, workers: int = 1) -> Path:
    parts = output.with_suffix(output.suffix + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    source = None
    for month in MONTHS:
        for day in DAYS:
            part = parts / f"2023-{month:02d}-{day:02d}.zarr"
            if part.exists():
                existing = xr.open_zarr(part, consolidated=False)
                if existing.sizes.get("time") == len(HOURS):
                    print(f"[arco] {part.name}: complete, skipping", flush=True)
                    continue
                raise RuntimeError(f"incomplete existing part: {part}")

            reusable = (
                temporal_parts / f"2023-{month:02d}-{day:02d}.zarr"
                if temporal_parts else None
            )
            if reusable and reusable.exists():
                cached = xr.open_zarr(reusable, consolidated=False)
                selected = cached.sel(time=cached.time.dt.hour.isin(HOURS)).compute()
                if selected.sizes.get("time") == len(HOURS):
                    _write_zarr(selected, part)
                    print(f"[arco] reused {reusable.name}", flush=True)
                    continue

            if source is None:
                source = xr.open_zarr(
                    ARCO_URL,
                    consolidated=True,
                    chunks={},
                    storage_options={"token": "anon"},
                )
            print(f"[arco] downloading 2023-{month:02d}-{day:02d}", flush=True)
            frames = []
            for index, timestamp in enumerate(_times(month, day), 1):
                frame = source.sel(
                    time=[timestamp],
                    hybrid=slice(49, 66),
                    latitude=slice(55, 25),
                    longitude=slice(225, 255),
                )[["u_component_of_wind", "v_component_of_wind"]]
                frames.append(_standardize(frame).load(
                    scheduler="threads", num_workers=workers
                ))
                print(f"[arco]   frame {index}/{len(HOURS)}", flush=True)
            selected = xr.concat(frames, dim="time")
            if not bool(np.isfinite(selected[["u", "v"]].to_array().values).all()):
                raise RuntimeError(f"non-finite values in month={month}, day={day}")
            selected.attrs = {
                "format_version": "v1",
                "generator": json.dumps({"name": "era5_arco_spatial_v2"}),
                "capabilities": json.dumps({"extent": "bounded"}),
                "conditioning": json.dumps({
                    "year": 2023, "month": month, "day": day,
                    "hours": list(HOURS),
                }),
                "model_levels": list(range(49, 67)),
                "units": "u,v:m/s",
                "source": ARCO_URL,
            }
            _write_zarr(selected, part)

    part_paths = [
        parts / f"2023-{month:02d}-{day:02d}.zarr"
        for month in MONTHS for day in DAYS
    ]
    combined = xr.concat(
        [xr.open_zarr(path, consolidated=False) for path in part_paths],
        dim="time",
    ).sortby("time")
    expected = len(MONTHS) * len(DAYS) * len(HOURS)
    if combined.sizes.get("time") != expected:
        raise RuntimeError(f"combined reference has {combined.sizes.get('time')}/{expected} frames")
    if output.exists():
        existing = xr.open_zarr(output, consolidated=False)
        if existing.sizes.get("time") == expected:
            print(f"[arco] combined reference already complete: {output}", flush=True)
            return output
        raise RuntimeError(f"incomplete existing output: {output}")
    _write_zarr(combined, output)
    print(f"[arco] complete: {output}", flush=True)
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--temporal-parts", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    download(args.output, temporal_parts=args.temporal_parts, workers=args.workers)


if __name__ == "__main__":
    main()
