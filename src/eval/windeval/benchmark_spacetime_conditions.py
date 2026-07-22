"""Condition-matched ERA5 benchmark for T=1/T=2 space-time InfiniteDiffusion runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from . import artifact
from .metrics import run_suite
from .metrics.distributions import conditional_w1_grouped
from .metrics.suite import METRIC_INFO


def _field(u, v, levels, lat, lon, *, time=None):
    return artifact.make_field(u, v, level=levels, lat=lat, lon=lon, time=time)


def _reference_region(path: Path):
    ds = xr.open_zarr(path, consolidated=False)
    # Matches generate_condition_set: query begins at (y,x)=(32,32) from (25N,225E).
    lat = np.asarray(ds["lat"].values)
    lon = np.asarray(ds["lon"].values)
    yi = np.where((lat >= 33.0) & (lat <= 48.75))[0]
    xi = np.where((lon >= 233.0) & (lon <= 248.75))[0]
    region = ds.isel(y=yi, x=xi).sortby("lat")
    if region.sizes["y"] != 64 or region.sizes["x"] != 64:
        raise ValueError(f"expected 64x64 reference region, got {region.sizes}")
    return region.load()


def _load_generated(path: Path, ref) -> tuple[object, list]:
    files = sorted(path.glob("m*_d*_h*_s*.npz"))
    config = json.loads((path / "config.json").read_text())
    expected = int(config.get("conditions", 4 * 2 * 7 * int(config["num_seeds"])))
    if len(files) != expected:
        raise ValueError(f"{path}: expected {expected} blocks, found {len(files)}")

    records = []
    for file in files:
        z = np.load(file)
        records.append(
            {
                "u": z["u"][0],
                "v": z["v"][0],
                "levels": z["levels"],
                "lat": z["lat"],
                "lon": z["lon"],
                "month": int(z["month"]),
                "day": int(z["day"]),
                "hour": int(z["hour"]),
                "seed": int(z["seed"]),
            }
        )
    first = records[0]
    expected_shape = (ref.sizes["level"], ref.sizes["y"], ref.sizes["x"])
    if first["u"].shape != expected_shape:
        raise ValueError(f"{path}: generated frame shape {first['u'].shape} != {expected_shape}")
    if not np.array_equal(first["levels"].astype(int), ref["level"].values.astype(int)):
        raise ValueError(f"{path}: generated levels do not match ERA5")
    if not np.allclose(first["lat"], ref["lat"].values) or not np.allclose(
        first["lon"], ref["lon"].values
    ):
        raise ValueError(f"{path}: generated coordinates do not match ERA5 reference region")
    pooled = _field(
        np.stack([r["u"] for r in records]),
        np.stack([r["v"] for r in records]),
        first["levels"], first["lat"], first["lon"],
        time=np.arange(len(records)),
    )

    groups = []
    condition_groups = sorted({(r["month"], r["hour"]) for r in records})
    for month, hour in condition_groups:
            matching = [r for r in records if r["month"] == month and r["hour"] == hour]
            seeds = [_field(r["u"], r["v"], r["levels"], r["lat"], r["lon"])
                     for r in matching]
            tref = ref.sel(time=(ref.time.dt.month == month) & (ref.time.dt.hour == hour))
            groups.append((seeds, tref))
    return pooled, groups


def _integer_time(ds):
    return ds.assign_coords(time=np.arange(ds.sizes["time"]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--runs", required=True, nargs="+", type=Path)
    parser.add_argument("--names", required=True, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("outputs/condition_benchmark.md"))
    args = parser.parse_args(argv)
    if len(args.runs) != len(args.names):
        parser.error("--runs and --names must have equal lengths")

    ref = _reference_region(args.reference)
    ref_metric = _integer_time(ref)
    rows = {}

    # Same-distribution floor for interpreting every raw distance.
    early = _integer_time(ref.sel(time=ref.time.dt.day < 11))
    late = _integer_time(ref.sel(time=ref.time.dt.day >= 11))
    floor, _ = run_suite(early, late)
    floor_groups = []
    for month in (1, 4, 7, 10):
        for hour in (0, 12):
            a = ref.sel(time=(ref.time.dt.month == month) & (ref.time.dt.hour == hour)
                        & (ref.time.dt.day < 11))
            b = ref.sel(time=(ref.time.dt.month == month) & (ref.time.dt.hour == hour)
                        & (ref.time.dt.day >= 11))
            floor_groups.append(([a], b))
    floor.update(conditional_w1_grouped(floor_groups))
    rows["ERA5 self-split floor"] = floor

    for name, path in zip(args.names, args.runs):
        generated, groups = _load_generated(path, ref)
        scores, _ = run_suite(generated, ref_metric)
        scores.update(conditional_w1_grouped(groups))
        rows[name] = scores

    metrics = ["SR_E", "SR_div", "SR_vort", "L_eff (km)",
               "W1 u (m/s)", "W1 v (m/s)", "tail err 1% (m/s)",
               "tail err 0.1% (m/s)", "W1 cond (m/s)"]
    lines = [
        "# Condition-matched space-time InfiniteDiffusion benchmark",
        "",
        "Reference: public ARCO-ERA5, model levels 49-66, days 8-14 of "
        "Jan/Apr/Jul/Oct 2023 at 00/12 UTC. Generated and reference rows use the same "
        "64x64 location and exact (month, day, hour) conditions. Lower is better for all "
        "reported distances; compare model rows with the ERA5 self-split sampling floor.",
        "",
        "Only frame 0 of each generated four-frame block is scored here. This isolates "
        "field realism at exact conditions; temporal tiling is evaluated separately.",
        "",
        "| Metric | " + " | ".join(rows) + " |",
        "|---|" + "---|" * len(rows),
    ]
    for metric in metrics:
        direction, note = METRIC_INFO[metric]
        values = []
        for row in rows.values():
            value = row.get(metric, np.nan)
            values.append("N/A" if not np.isfinite(value) else f"{value:.4f}")
        lines.append(f"| {metric} ({direction}) | " + " | ".join(values) + " |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
