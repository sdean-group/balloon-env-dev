"""Validate a prepared hourly ERA5 store before expensive diffusion training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .data import NormStats, _open_zarr
except ImportError:  # pragma: no cover - standalone cluster entrypoint
    from data import NormStats, _open_zarr


def expected_hourly_times(first_year: int, last_year: int) -> np.ndarray:
    if last_year < first_year:
        raise ValueError("last_year must be greater than or equal to first_year")
    return np.arange(
        np.datetime64(f"{first_year}-01-01T00", "h"),
        np.datetime64(f"{last_year + 1}-01-01T00", "h"),
        np.timedelta64(1, "h"),
    )


def validate_hourly_times(
    times: np.ndarray,
    *,
    first_year: int,
    last_year: int,
) -> None:
    actual = np.asarray(times).astype("datetime64[h]")
    expected = expected_hourly_times(first_year, last_year)
    if np.array_equal(actual, expected):
        return
    shared = min(len(actual), len(expected))
    mismatch = np.flatnonzero(actual[:shared] != expected[:shared])
    detail = (
        f"first mismatch at index {int(mismatch[0])}"
        if len(mismatch)
        else f"length {len(actual)} != {len(expected)}"
    )
    raise ValueError(f"hourly timeline is incomplete or duplicated ({detail})")


def validate(
    path: Path,
    *,
    first_year: int,
    last_year: int,
    stats_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict:
    ds = _open_zarr(path)
    required = {"u", "v"}
    if not required.issubset(ds.data_vars):
        raise ValueError(f"{path}: expected variables {sorted(required)}")
    expected_levels = np.arange(49, 67)
    levels = np.asarray(ds["level"].values, dtype=int)
    if not np.array_equal(levels, expected_levels):
        raise ValueError(f"{path}: expected levels 49-66, found {levels.tolist()}")
    if int(ds.sizes["y"]) != 121 or int(ds.sizes["x"]) != 121:
        raise ValueError(f"{path}: expected a 121x121 regional grid, found {dict(ds.sizes)}")

    times = np.asarray(ds["time"].values).astype("datetime64[h]")
    try:
        validate_hourly_times(times, first_year=first_year, last_year=last_year)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc

    lat = np.asarray(ds["lat"].values, dtype=float)
    lon = np.mod(np.asarray(ds["lon"].values, dtype=float), 360.0)
    if not np.isclose(lat.max(), 55.0) or not np.isclose(lat.min(), 25.0):
        raise ValueError(f"{path}: latitude extent is not 25-55 degrees")
    if not np.isclose(lon.min(), 225.0) or not np.isclose(lon.max(), 255.0):
        raise ValueError(f"{path}: longitude extent is not 225-255 degrees")
    ds.close()

    if stats_path is not None:
        stats = NormStats.load(stats_path)
        if not np.array_equal(np.asarray(stats.levels, dtype=int), expected_levels):
            raise ValueError(f"{stats_path}: normalization levels do not match the store")
        arrays = (stats.mean_u, stats.std_u, stats.mean_v, stats.std_v)
        if not all(np.isfinite(values).all() for values in arrays):
            raise ValueError(f"{stats_path}: normalization statistics contain non-finite values")
        if np.any(stats.std_u <= 0) or np.any(stats.std_v <= 0):
            raise ValueError(f"{stats_path}: normalization standard deviations must be positive")

    summary = {
        "path": str(path.resolve()),
        "first_year": first_year,
        "last_year": last_year,
        "years": last_year - first_year + 1,
        "hours": len(times),
        "levels": [49, 66],
        "grid": [121, 121],
        "stats_path": str(stats_path.resolve()) if stats_path is not None else None,
        "complete": True,
    }
    if manifest_path is not None:
        manifest_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--first-year", required=True, type=int)
    parser.add_argument("--last-year", required=True, type=int)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    summary = validate(
        args.path,
        first_year=args.first_year,
        last_year=args.last_year,
        stats_path=args.stats,
        manifest_path=args.manifest,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
