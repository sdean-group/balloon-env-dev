"""Shared-protocol ERA5, multiple InfiniteDiffusion depths, and BLE-VAE benchmark.

Every scored field is placed on the same horizontal grid and pressure coordinates before
the common metric suite is called. InfiniteDiffusion is evaluated at its true held-out
conditions. BLE-VAE is unconditional, so it is evaluated as an equal-size sample pool
and does not receive a conditional-W1 score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from . import artifact
from .l137 import full_level_pressure
from .metrics import run_suite
from .metrics.distributions import conditional_w1_grouped
from .metrics.suite import METRIC_INFO
from .reference import SPLIT_DAY, split
from .resample import common_grid, regrid

CENTER_LAT = 37.77
CENTER_LON = 237.58
GRID_SIZE = 16
GRID_SPACING_KM = 50.0
TARGET_PRESSURES_HPA = np.arange(60.0, 131.0, 10.0)
STANDARD_SURFACE_PRESSURE_PA = 101_325.0


def _standard_model_pressure_hpa(levels: np.ndarray) -> np.ndarray:
    """Representative pressure for ERA5 model levels in the upper-atmosphere band."""
    pressure = full_level_pressure(
        np.asarray(levels, dtype=int),
        np.asarray(STANDARD_SURFACE_PRESSURE_PA),
    )
    return np.asarray(pressure, dtype=np.float64).reshape(-1) / 100.0


def _vertical_interpolate(
    ds: xr.Dataset,
    source_pressure_hpa: np.ndarray,
    target_pressure_hpa: np.ndarray = TARGET_PRESSURES_HPA,
) -> xr.Dataset:
    """Linearly interpolate u/v along pressure without horizontal extrapolation."""
    source = np.asarray(source_pressure_hpa, dtype=np.float64)
    target = np.asarray(target_pressure_hpa, dtype=np.float64)
    order = np.argsort(source)
    source = source[order]
    if target.min() < source.min() or target.max() > source.max():
        raise ValueError(
            f"target pressures {target.min():g}-{target.max():g} hPa exceed source "
            f"range {source.min():g}-{source.max():g} hPa"
        )

    hi = np.searchsorted(source, target, side="left")
    hi = np.clip(hi, 1, len(source) - 1)
    lo = hi - 1
    weight = (target - source[lo]) / (source[hi] - source[lo])
    weight = weight.reshape(1, -1, 1, 1)

    values = {}
    for name in ("u", "v"):
        data = np.asarray(ds[name].values)[:, order]
        values[name] = data[:, lo] * (1.0 - weight) + data[:, hi] * weight
    return artifact.make_field(
        values["u"],
        values["v"],
        level=target,
        lat=ds["lat"].values,
        lon=ds["lon"].values,
        time=ds["time"].values,
    )


def _ensure_contains_grid(ds: xr.Dataset, lat: np.ndarray, lon: np.ndarray) -> None:
    src_lat = np.asarray(ds["lat"].values, dtype=np.float64)
    src_lon = np.asarray(ds["lon"].values, dtype=np.float64)
    if (
        lat.min() < src_lat.min()
        or lat.max() > src_lat.max()
        or lon.min() < src_lon.min()
        or lon.max() > src_lon.max()
    ):
        raise ValueError("source field does not contain the complete shared grid")


def _to_shared_grid(
    ds: xr.Dataset,
    source_pressure_hpa: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    source_spacing_km: float,
) -> xr.Dataset:
    _ensure_contains_grid(ds, lat, lon)
    horizontal = regrid(
        ds,
        lat,
        lon,
        src_spacing_km=source_spacing_km,
        target_spacing_km=GRID_SPACING_KM,
    )
    return _vertical_interpolate(horizontal, source_pressure_hpa)


def _condition_files(path: Path) -> list[Path]:
    files = sorted(path.glob("m*_d*_h*_s*.npz"))
    config_path = path / "config.json"
    if not config_path.exists():
        raise ValueError(f"{path}: missing config.json")
    expected = int(json.loads(config_path.read_text())["conditions"])
    if len(files) != expected:
        raise ValueError(f"{path}: expected {expected} condition files, found {len(files)}")
    return files


def _load_infinite_records(
    path: Path,
    lat: np.ndarray,
    lon: np.ndarray,
) -> list[dict]:
    records = []
    for file in _condition_files(path):
        with np.load(file) as data:
            field = artifact.make_field(
                data["u"][0],
                data["v"][0],
                level=data["levels"],
                lat=data["lat"],
                lon=data["lon"],
            )
            shared = _to_shared_grid(
                field,
                _standard_model_pressure_hpa(data["levels"]),
                lat,
                lon,
                source_spacing_km=27.83,
            )
            records.append(
                {
                    "u": shared["u"].values[0],
                    "v": shared["v"].values[0],
                    "month": int(data["month"]),
                    "day": int(data["day"]),
                    "hour": int(data["hour"]),
                    "seed": int(data["seed"]),
                }
            )
    return records


def _condition_keys(records: list[dict]) -> set[tuple[int, int, int, int]]:
    return {
        (record["month"], record["day"], record["hour"], record["seed"])
        for record in records
    }


def _validate_matching_conditions(records_by_name: dict[str, list[dict]]) -> None:
    names = list(records_by_name)
    expected = _condition_keys(records_by_name[names[0]])
    for name in names[1:]:
        actual = _condition_keys(records_by_name[name])
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"{name}: condition set differs from {names[0]}; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )


def _reference_for_records(
    path: Path,
    records: list[dict],
    lat: np.ndarray,
    lon: np.ndarray,
) -> xr.Dataset:
    source = xr.open_zarr(path, consolidated=False)
    wanted = {
        np.datetime64(
            f"2023-{record['month']:02d}-{record['day']:02d}T{record['hour']:02d}",
            "h",
        )
        for record in records
    }
    available = np.asarray(source["time"].values).astype("datetime64[h]")
    indices = np.array([index for index, value in enumerate(available) if value in wanted])
    if len(indices) != len(wanted):
        raise ValueError(
            f"reference contains {len(indices)} of {len(wanted)} requested timestamps"
        )
    selected = source.isel(time=indices).load()
    return _to_shared_grid(
        selected,
        _standard_model_pressure_hpa(selected["level"].values),
        lat,
        lon,
        source_spacing_km=27.83,
    )


def _pool(records: list[dict], lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    return artifact.make_field(
        np.stack([record["u"] for record in records]),
        np.stack([record["v"] for record in records]),
        level=TARGET_PRESSURES_HPA,
        lat=lat,
        lon=lon,
        time=np.arange(len(records)),
    )


def _conditional_groups(
    records: list[dict],
    reference: xr.Dataset,
    lat: np.ndarray,
    lon: np.ndarray,
) -> list[tuple[list[xr.Dataset], xr.Dataset]]:
    groups = []
    for month, hour in sorted({(r["month"], r["hour"]) for r in records}):
        matching = [r for r in records if r["month"] == month and r["hour"] == hour]
        samples = [
            artifact.make_field(
                record["u"],
                record["v"],
                level=TARGET_PRESSURES_HPA,
                lat=lat,
                lon=lon,
            )
            for record in matching
        ]
        ref = reference.sel(
            time=(reference.time.dt.month == month) & (reference.time.dt.hour == hour)
        )
        groups.append((samples, ref))
    return groups


def _ble_records(
    count: int,
    decoder_path: Path,
    cache_path: Path | None,
    lat: np.ndarray,
    lon: np.ndarray,
) -> list[dict]:
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as data:
            if int(data["count"]) != count:
                raise ValueError(
                    f"{cache_path}: cached {int(data['count'])} samples, expected {count}"
                )
            return [
                {"u": data["u"][index], "v": data["v"][index]}
                for index in range(count)
            ]

    from .generators import ble_vae

    params = ble_vae.load_params(decoder_path)
    native_lat, native_lon, native_pressure = ble_vae._coords()
    records = []
    for seed in range(count):
        decoded = ble_vae.sample(params, seed=seed)
        # The center BLE frame avoids treating its nine-frame block as nine independent draws.
        frame = decoded[:, :, :, ble_vae.N_TIME // 2, :]
        field = artifact.make_field(
            np.transpose(frame[..., 0], (2, 0, 1)),
            np.transpose(frame[..., 1], (2, 0, 1)),
            level=native_pressure,
            lat=native_lat,
            lon=native_lon,
        )
        shared = _to_shared_grid(
            field,
            native_pressure,
            lat,
            lon,
            source_spacing_km=50.0,
        )
        records.append({"u": shared["u"].values[0], "v": shared["v"].values[0]})
        print(f"[BLE] {seed + 1}/{count}", flush=True)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            count=count,
            u=np.stack([record["u"] for record in records]),
            v=np.stack([record["v"] for record in records]),
            lat=lat,
            lon=lon,
            pressure_hpa=TARGET_PRESSURES_HPA,
        )
    return records


def _score_floor(reference: xr.Dataset) -> dict:
    early, late = split(reference)
    if early.sizes["time"] == 0 or late.sizes["time"] == 0:
        raise ValueError("reference needs held-out days on both sides of day 11")
    scores, _ = run_suite(
        early.assign_coords(time=np.arange(early.sizes["time"])),
        late.assign_coords(time=np.arange(late.sizes["time"])),
    )
    floor_groups = []
    for month in sorted(set(reference.time.dt.month.values.tolist())):
        for hour in sorted(set(reference.time.dt.hour.values.tolist())):
            a = reference.sel(
                time=(reference.time.dt.month == month)
                & (reference.time.dt.hour == hour)
                & (reference.time.dt.day < SPLIT_DAY)
            )
            b = reference.sel(
                time=(reference.time.dt.month == month)
                & (reference.time.dt.hour == hour)
                & (reference.time.dt.day >= SPLIT_DAY)
            )
            floor_groups.append(([a], b))
    scores.update(conditional_w1_grouped(floor_groups))
    return scores


def _write_report(
    rows: dict[str, dict],
    output: Path,
    sample_count: int,
    infinite_names: list[str],
) -> None:
    metrics = [
        "SR_E",
        "SR_div",
        "SR_vort",
        "L_eff (km)",
        "W1 u (m/s)",
        "W1 v (m/s)",
        "tail err 1% (m/s)",
        "tail err 0.1% (m/s)",
        "W1 cond (m/s)",
    ]
    lines = [
        "# Shared-protocol InfiniteDiffusion depth and BLE-VAE benchmark",
        "",
        f"All rows use the same {GRID_SIZE}x{GRID_SIZE} SF-centered grid at "
        f"{GRID_SPACING_KM:g} km spacing and the same 60-130 hPa pressure coordinates. "
        "ERA5 and InfiniteDiffusion model levels are linearly interpolated using the "
        "L137 pressures at 1013.25 hPa surface pressure. No spatial extrapolation is used.",
        "",
        "The ERA5 reference uses exactly the held-out timestamps represented by every "
        "conditional model set. " + ", ".join(infinite_names) + " are evaluated "
        f"at the same conditions and seeds. BLE-VAE contributes {sample_count} independent "
        "latent samples, equal to each InfiniteDiffusion sample count, but cannot match "
        "timestamps because it is unconditional. Its conditional W1 is therefore N/A.",
        "",
        "Only the central BLE temporal frame and frame 0 of each InfiniteDiffusion block "
        "are scored. Temporal metrics are outside this comparison. Spectral metrics have "
        "limited wavenumber resolution on a 16x16 grid.",
        "",
        "| Metric | " + " | ".join(rows) + " |",
        "|---|" + "---|" * len(rows),
    ]
    for metric in metrics:
        direction, _ = METRIC_INFO[metric]
        values = []
        for row in rows.values():
            value = row.get(metric, np.nan)
            values.append("N/A" if not np.isfinite(value) else f"{value:.4f}")
        lines.append(f"| {metric} ({direction}) | " + " | ".join(values) + " |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument(
        "--model-runs",
        "--infinite-runs",
        dest="model_runs",
        required=True,
        nargs="+",
        type=Path,
    )
    parser.add_argument(
        "--model-names",
        "--infinite-names",
        dest="model_names",
        required=True,
        nargs="+",
    )
    parser.add_argument("--ble-decoder", required=True, type=Path)
    parser.add_argument("--ble-cache", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if len(args.model_runs) != len(args.model_names):
        parser.error("--model-runs and --model-names must have equal lengths")

    lat, lon = common_grid(
        CENTER_LAT,
        CENTER_LON,
        GRID_SPACING_KM,
        GRID_SIZE,
    )
    records_by_name = {
        name: _load_infinite_records(path, lat, lon)
        for name, path in zip(args.model_names, args.model_runs)
    }
    _validate_matching_conditions(records_by_name)
    first_records = records_by_name[args.model_names[0]]
    reference = _reference_for_records(args.reference, first_records, lat, lon)
    ble_records = _ble_records(
        len(first_records),
        args.ble_decoder,
        args.ble_cache,
        lat,
        lon,
    )

    rows = {"ERA5 self-split floor": _score_floor(reference)}
    for name in args.model_names:
        records = records_by_name[name]
        infinite_scores, _ = run_suite(
            _pool(records, lat, lon),
            reference.assign_coords(time=np.arange(reference.sizes["time"])),
        )
        infinite_scores.update(
            conditional_w1_grouped(
                _conditional_groups(records, reference, lat, lon)
            )
        )
        rows[name] = infinite_scores

    ble_scores, _ = run_suite(
        _pool(ble_records, lat, lon),
        reference.assign_coords(time=np.arange(reference.sizes["time"])),
    )
    ble_scores["W1 cond (m/s)"] = np.nan
    rows["BLE-VAE"] = ble_scores
    _write_report(
        rows,
        args.output,
        len(first_records),
        args.model_names,
    )
    print(args.output)


if __name__ == "__main__":
    main()
