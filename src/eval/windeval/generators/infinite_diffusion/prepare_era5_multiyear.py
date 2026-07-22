"""Resumable ERA5 model-level preparation for the multi-year conditional run.

Downloads one calendar month at a time, appends it to a chunked Zarr store, and removes
the temporary GRIB by default. The bounded temporary footprint matters because ten years
of regional hourly u/v data is roughly 180 GB before compression.
"""
from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import numpy as np
import xarray as xr

try:
    from .data import _open_zarr, compute_zarr_stats
except ImportError:  # pragma: no cover - standalone cluster entrypoint
    from data import _open_zarr, compute_zarr_stats


def _month_specs(years: list[int]):
    for year in years:
        for month in range(1, 13):
            last = calendar.monthrange(year, month)[1]
            yield year, month, f"{year}-{month:02d}-01/to/{year}-{month:02d}-{last:02d}"


def _download(path: Path, *, dates: str, area: list[float], levels: tuple[int, int]) -> None:
    import cdsapi

    levelist = "/".join(str(level) for level in range(levels[0], levels[1] + 1))
    cdsapi.Client().retrieve(
        "reanalysis-era5-complete",
        {
            "class": "ea",
            "type": "an",
            "stream": "oper",
            "expver": "1",
            "levtype": "ml",
            "levelist": levelist,
            "param": "131/132",
            "date": dates,
            "time": "00/to/23/by/1",
            "area": area,
            "grid": "0.25/0.25",
            "format": "grib",
        },
        str(path),
    )


def _read_grib(path: Path) -> xr.Dataset:
    src = xr.open_dataset(path, engine="cfgrib")
    u = np.asarray(src["u"].values, dtype=np.float32)
    v = np.asarray(src["v"].values, dtype=np.float32)
    if u.ndim == 3:
        u, v = u[None], v[None]
    times = np.atleast_1d(src["valid_time"].values)
    order = np.argsort(times)
    ds = xr.Dataset(
        {"u": (("time", "level", "y", "x"), u[order]),
         "v": (("time", "level", "y", "x"), v[order])},
        coords={
            "time": ("time", times[order]),
            "level": ("level", np.asarray(src["hybrid"].values, dtype=np.int32)),
            "lat": ("y", np.asarray(src["latitude"].values, dtype=np.float64)),
            "lon": ("x", np.asarray(src["longitude"].values, dtype=np.float64)),
        },
        attrs={
            "source": "ERA5 complete model-level analysis",
            "variables": "u,v",
            "units": "m s-1",
        },
    )
    src.close()
    return ds


def _last_stored_time(path: Path) -> np.datetime64 | None:
    if not path.exists():
        return None
    ds = _open_zarr(path)
    last = np.asarray(ds["time"].values)[-1]
    ds.close()
    return last


def _append_month(ds: xr.Dataset, out: Path) -> None:
    def write(**kwargs) -> None:
        try:
            ds.to_zarr(out, consolidated=False, zarr_format=2, **kwargs)
        except TypeError:  # xarray versions before the zarr v3 transition
            ds.to_zarr(out, consolidated=False, **kwargs)

    if not out.exists():
        chunks = (4, int(ds.sizes["level"]), 64, 64)
        encoding = {name: {"chunks": chunks} for name in ("u", "v")}
        write(mode="w", encoding=encoding)
    else:
        write(mode="a", append_dim="time")


def prepare(args: argparse.Namespace) -> None:
    out = Path(args.out)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    levels = tuple(args.levels)
    area = [float(value) for value in args.area.split("/")]
    last_time = _last_stored_time(out)

    for year, month, dates in _month_specs(args.years):
        month_end = np.datetime64(f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}T23")
        if last_time is not None and month_end <= last_time:
            print(f"[prepare] {year}-{month:02d}: already stored", flush=True)
            continue

        grib = work / f"era5_ml_uv_{year}_{month:02d}.grib"
        if not grib.exists():
            partial = grib.with_suffix(".grib.part")
            print(f"[prepare] downloading {year}-{month:02d}", flush=True)
            _download(partial, dates=dates, area=area, levels=levels)
            partial.rename(grib)

        ds = _read_grib(grib)
        if last_time is not None and np.asarray(ds["time"].values)[0] <= last_time:
            raise ValueError(
                f"{grib} overlaps existing store ending at {last_time}; "
                "remove the incomplete final month before retrying"
            )
        expected_levels = np.arange(levels[0], levels[1] + 1)
        if not np.array_equal(np.asarray(ds["level"].values), expected_levels):
            raise ValueError(f"unexpected model levels in {grib}: {ds['level'].values}")
        _append_month(ds, out)
        last_time = np.asarray(ds["time"].values)[-1]
        print(f"[prepare] appended {year}-{month:02d}: {ds.sizes['time']} hours", flush=True)
        ds.close()

        if not args.keep_grib:
            grib.unlink()
            for index in work.glob(f"{grib.name}*.idx"):
                index.unlink()

    if args.stats_out:
        print("[prepare] scanning training store for normalization statistics", flush=True)
        stats = compute_zarr_stats(out, levels=levels, time_chunk=args.stats_time_chunk)
        stats.save(args.stats_out)
        print(f"[prepare] statistics -> {args.stats_out}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare chunked multi-year ERA5 training data")
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--stats-out", default=None,
                        help="write training-only NormStats after all months are present")
    parser.add_argument("--stats-time-chunk", type=int, default=168)
    parser.add_argument("--area", default="55/-135/25/-105", help="N/W/S/E")
    parser.add_argument("--levels", type=int, nargs=2, default=[49, 66])
    parser.add_argument("--keep-grib", action="store_true")
    args = parser.parse_args(argv)
    prepare(args)


if __name__ == "__main__":
    main()
