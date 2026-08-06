"""Construct a benchmark artifact with the published RL-HAB SynthWinds method.

The public RL-HAB notebook uses nearest-station horizontal interpolation followed by a
Gaussian smoother with sigma 3 cells on its 1-degree grid. Its native vertical grid is
250 m. For this benchmark we interpolate each sounding directly to the benchmark
pressure levels first, then use the equivalent 3-degree smoother on the 0.25-degree
evaluation grid.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from bs4 import BeautifulSoup
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from ... import artifact
from ...benchmark import _level_hpa

TARGET_DAYS = tuple(range(8, 15))
TARGET_HOURS = (0, 12)


def _station_catalog(path: Path) -> dict[str, tuple[float, float]]:
    with path.open(newline="") as stream:
        return {
            row["wmo"]: (float(row["latitude"]), float(row["longitude"]))
            for row in csv.DictReader(stream)
        }


def parse_archive(path: Path, station_wmo: str, station_position) -> list[dict]:
    """Parse one cached University of Wyoming FM35 sounding response."""
    soup = BeautifulSoup(path.read_text(errors="replace"), "html.parser")
    block = next((item.get_text() for item in soup.find_all("pre") if "PRES" in item.get_text()), None)
    if block is None:
        return []
    table = pd.read_fwf(
        StringIO(block),
        widths=[7] * 11,
        skiprows=4,
        names=[
            "pressure", "height", "temperature", "dewpoint", "humidity", "mixing",
            "direction", "speed", "theta", "theta_e", "theta_v",
        ],
    )
    for column in ("pressure", "direction", "speed"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.dropna(subset=["pressure", "direction", "speed"])
    if len(table) < 4:
        return []
    timestamp = datetime.strptime(path.name.split("-")[0], "%Y%m%dT%H")
    pressure = table["pressure"].to_numpy(float)
    direction = np.deg2rad(table["direction"].to_numpy(float))
    speed = table["speed"].to_numpy(float)
    return [{
        "time": np.datetime64(timestamp, "ns"),
        "wmo": station_wmo,
        "lat": float(station_position[0]),
        "lon": float(station_position[1]),
        "pressure": pressure,
        "u": -speed * np.sin(direction),
        "v": -speed * np.cos(direction),
    }]


def _profile_at_levels(profile: dict, target_hpa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pressure = profile["pressure"]
    order = np.argsort(pressure)
    pressure = pressure[order]
    unique, unique_idx = np.unique(pressure, return_index=True)
    u = profile["u"][order][unique_idx]
    v = profile["v"][order][unique_idx]
    return (
        np.interp(target_hpa, unique, u, left=np.nan, right=np.nan),
        np.interp(target_hpa, unique, v, left=np.nan, right=np.nan),
    )


def _nearest_smoothed_field(
    profiles: list[dict], target_hpa: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    station_uv = [_profile_at_levels(profile, target_hpa) for profile in profiles]
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    query = np.column_stack([yy.ravel(), xx.ravel()])
    u = np.empty((len(target_hpa), len(lat), len(lon)))
    v = np.empty_like(u)
    for level_index, level in enumerate(target_hpa):
        valid = [
            i for i, (station_u, station_v) in enumerate(station_uv)
            if np.isfinite(station_u[level_index]) and np.isfinite(station_v[level_index])
        ]
        if len(valid) < 3:
            raise RuntimeError(
                f"only {len(valid)} radiosonde profiles cover {level:.1f} hPa"
            )
        points = np.array([[profiles[i]["lat"], profiles[i]["lon"]] for i in valid])
        nearest = cKDTree(points).query(query)[1]
        u[level_index] = np.array([station_uv[i][0][level_index] for i in valid])[nearest].reshape(
            len(lat), len(lon)
        )
        v[level_index] = np.array([station_uv[i][1][level_index] for i in valid])[nearest].reshape(
            len(lat), len(lon)
        )

    # RL-HAB uses sigma=3 cells at 1 degree. Preserve that 3-degree physical scale.
    dy = abs(float(np.median(np.diff(lat))))
    dx = abs(float(np.median(np.diff(lon))))
    sigma = (0.0, 3.0 / dy, 3.0 / dx)
    return gaussian_filter(u, sigma=sigma), gaussian_filter(v, sigma=sigma)


def generate_artifact(
    raw_dir: Path,
    reference_path: Path,
    output_path: Path,
    *,
    stations_path: Path,
) -> Path:
    ref = artifact.read(reference_path)
    target_hpa = _level_hpa(ref)
    lat = np.asarray(ref["lat"].values, dtype=float)
    lon = np.asarray(ref["lon"].values, dtype=float)
    if np.nanmean(lon) > 180:
        station_lon_transform = lambda x: x % 360.0
    else:
        station_lon_transform = lambda x: ((x + 180.0) % 360.0) - 180.0

    catalog = _station_catalog(stations_path)
    profiles = []
    for path in sorted(raw_dir.glob("*.html")):
        wmo = path.stem.rsplit("-", 1)[-1]
        if wmo not in catalog:
            continue
        station = (catalog[wmo][0], station_lon_transform(catalog[wmo][1]))
        profiles.extend(parse_archive(path, wmo, station))

    by_time: dict[np.datetime64, list[dict]] = {}
    for profile in profiles:
        by_time.setdefault(profile["time"], []).append(profile)
    ref_times = np.asarray(ref["time"].values).astype("datetime64[ns]")
    expected = [
        timestamp for timestamp in ref_times
        if int(pd.Timestamp(timestamp).day) in TARGET_DAYS
        and int(pd.Timestamp(timestamp).hour) in TARGET_HOURS
    ]
    missing = [str(t) for t in expected if t not in by_time]
    if missing:
        raise RuntimeError(f"missing all radiosonde profiles for {len(missing)} timestamps: {missing[:4]}")

    fields_u, fields_v = [], []
    for index, timestamp in enumerate(expected, 1):
        current = by_time[timestamp]
        u, v = _nearest_smoothed_field(current, target_hpa, lat, lon)
        fields_u.append(u)
        fields_v.append(v)
        print(f"[synth {index}/{len(expected)}] {timestamp}: {len(current)} stations", flush=True)

    ds = artifact.make_field(
        np.stack(fields_u),
        np.stack(fields_v),
        level=ref["level"].values,
        lat=lat,
        lon=lon,
        time=np.asarray(expected),
    )
    attrs = artifact.default_attrs(
        generator={"name": "rlhab_synthwinds", "version": "RL-HAB-public-notebook"},
        capabilities={"extent": "bounded", "temporally_evolving": True},
        conditioning={
            "source": "University of Wyoming radiosondes",
            "method": "nearest station plus 3-degree Gaussian smoothing",
            "observation_driven": True,
        },
        model_levels=ref["level"].values,
        dt_native="12h",
    )
    return artifact.write(ds, attrs, output_path)


def main(argv=None) -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate an RL-HAB SynthWinds artifact")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stations", type=Path, default=here / "stations.csv")
    args = parser.parse_args(argv)
    output = generate_artifact(
        args.raw_dir, args.reference, args.output, stations_path=args.stations
    )
    print(output)


if __name__ == "__main__":
    main()
