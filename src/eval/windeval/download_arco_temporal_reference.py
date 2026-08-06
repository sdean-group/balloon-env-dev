"""Download compact contiguous ERA5 episodes for the temporal wind benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from .download_arco_conditional_reference import ARCO_URL, _write_zarr

MONTHS = (1, 4, 7, 10)
DAYS = (8, 9)
HOURS = tuple(range(24))


def _times(month: int, day: int) -> np.ndarray:
    return np.asarray(
        [
            np.datetime64(f"2023-{month:02d}-{day:02d}T{hour:02d}", "ns")
            for hour in HOURS
        ]
    )


def download(output: Path, *, workers: int = 4) -> Path:
    """Download eight 24-hour episodes and combine them into one Zarr store."""
    parts = output.with_suffix(output.suffix + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    source = xr.open_zarr(
        ARCO_URL,
        consolidated=True,
        chunks={},
        storage_options={"token": "anon"},
    )

    expected_part_frames = len(HOURS)
    for month in MONTHS:
        for day in DAYS:
            part = parts / f"2023-{month:02d}-{day:02d}.zarr"
            if part.exists():
                existing = xr.open_zarr(part, consolidated=False)
                if existing.sizes.get("time") == expected_part_frames:
                    print(f"[arco] {part.name}: complete, skipping", flush=True)
                    continue
                raise RuntimeError(f"incomplete existing part: {part}")

            print(
                f"[arco] downloading 2023-{month:02d}-{day:02d}, all 24 hours",
                flush=True,
            )
            frames = []
            for hour, timestamp in enumerate(_times(month, day)):
                # ARCO is globally chunked. Loading all 24 timestamps together can
                # retain many full source chunks and exceed cluster memory even though
                # the final regional array is small.
                frame = source.sel(
                    time=[timestamp],
                    hybrid=slice(49, 66),
                    latitude=slice(55, 25),
                    longitude=slice(225, 255),
                )[["u_component_of_wind", "v_component_of_wind"]]
                frame = frame.rename(
                    {
                        "u_component_of_wind": "u",
                        "v_component_of_wind": "v",
                        "hybrid": "level",
                        "latitude": "y",
                        "longitude": "x",
                    }
                ).rename_vars({"y": "lat", "x": "lon"})
                frame = frame.transpose("time", "level", "y", "x")
                frames.append(
                    frame.load(scheduler="threads", num_workers=workers)
                )
                print(f"[arco]   hour {hour + 1}/24", flush=True)
            selected = xr.concat(frames, dim="time")
            if not bool(np.isfinite(selected[["u", "v"]].to_array().values).all()):
                raise RuntimeError(f"non-finite values in month={month}, day={day}")
            selected.attrs = {
                "format_version": "v1",
                "generator": json.dumps({"name": "era5_arco_temporal_reference"}),
                "capabilities": json.dumps(
                    {"extent": "bounded", "temporally_evolving": True}
                ),
                "conditioning": json.dumps(
                    {"year": 2023, "month": month, "day": day, "hours": [0, 23]}
                ),
                "model_levels": list(range(49, 67)),
                "units": "u,v:m/s",
                "source": ARCO_URL,
            }
            _write_zarr(selected, part)
            print(f"[arco] wrote {part}", flush=True)

    part_paths = [
        parts / f"2023-{month:02d}-{day:02d}.zarr"
        for month in MONTHS
        for day in DAYS
    ]
    datasets = [xr.open_zarr(path, consolidated=False) for path in part_paths]
    combined = xr.concat(datasets, dim="time").sortby("time")
    expected_frames = len(MONTHS) * len(DAYS) * len(HOURS)
    if output.exists():
        existing = xr.open_zarr(output, consolidated=False)
        if existing.sizes.get("time") == expected_frames:
            print(f"[arco] combined reference already complete: {output}", flush=True)
            return output
        raise RuntimeError(f"incomplete existing output: {output}")
    _write_zarr(combined, output)
    print(f"[arco] complete: {output}", flush=True)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    download(args.output, workers=args.workers)


if __name__ == "__main__":
    main()
