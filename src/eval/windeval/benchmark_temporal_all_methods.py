"""Temporal benchmark for direct, InfiniteDiffusion, CFGD, and ERA5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from . import artifact
from .benchmark_shared_baselines import (
    CENTER_LAT,
    CENTER_LON,
    GRID_SIZE,
    GRID_SPACING_KM,
    _standard_model_pressure_hpa,
    _to_shared_grid,
)
from .metrics.temporal import (
    contiguous_segments,
    dispersion_compare,
    temporal_spectral_residual,
)
from .resample import common_grid


def _episode_files(path: Path) -> list[Path]:
    config_path = path / "config.json"
    if not config_path.exists():
        raise ValueError(f"{path}: missing config.json")
    expected = int(json.loads(config_path.read_text())["conditions"])
    files = sorted(path.glob("m*_d*_s*.npz"))
    if len(files) != expected:
        raise ValueError(f"{path}: expected {expected} episodes, found {len(files)}")
    return files


def _combine(episodes: list[xr.Dataset]) -> xr.Dataset:
    return xr.concat(episodes, dim="time").sortby("time")


def _load_run(path: Path, lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    episodes = []
    for file in _episode_files(path):
        with np.load(file) as data:
            field = artifact.make_field(
                data["u"],
                data["v"],
                level=data["levels"],
                lat=data["lat"],
                lon=data["lon"],
                time=data["times"].astype("datetime64[ns]"),
            )
            episodes.append(
                _to_shared_grid(
                    field,
                    _standard_model_pressure_hpa(data["levels"]),
                    lat,
                    lon,
                    source_spacing_km=27.83,
                )
            )
    combined = _combine(episodes)
    if len(contiguous_segments(combined)) != len(episodes):
        raise ValueError(f"{path}: expected {len(episodes)} contiguous 24-hour episodes")
    return combined


def _load_reference(
    path: Path,
    day: int,
    lat: np.ndarray,
    lon: np.ndarray,
) -> xr.Dataset:
    source = xr.open_zarr(path, consolidated=False)
    selected = source.sel(time=source.time.dt.day == day).load()
    shared = _to_shared_grid(
        selected,
        _standard_model_pressure_hpa(selected["level"].values),
        lat,
        lon,
        source_spacing_km=27.83,
    )
    if len(contiguous_segments(shared)) != 4:
        raise ValueError(f"ERA5 day {day} does not contain four contiguous episodes")
    return shared


def _mean_adjacent_change(ds: xr.Dataset) -> float:
    values = []
    for segment in contiguous_segments(ds):
        du = np.diff(ds["u"].values[segment], axis=0)
        dv = np.diff(ds["v"].values[segment], axis=0)
        values.append(np.sqrt(du**2 + dv**2).mean())
    return float(np.mean(values))


def _seam_jump_ratio(ds: xr.Dataset, period: int) -> float:
    seam_jumps, all_jumps = [], []
    for segment in contiguous_segments(ds):
        u = ds["u"].values[segment]
        v = ds["v"].values[segment]
        jumps = np.sqrt(np.diff(u, axis=0) ** 2 + np.diff(v, axis=0) ** 2)
        seam_indices = np.arange(period - 1, jumps.shape[0], period)
        seam_jumps.append(jumps[seam_indices].mean())
        all_jumps.append(jumps.mean())
    return float(np.mean(seam_jumps) / np.mean(all_jumps))


def _score(pred: xr.Dataset, reference: xr.Dataset, seam_period: int | None) -> dict:
    scores, _ = temporal_spectral_residual(pred, reference)
    dispersion, _ = dispersion_compare(pred, reference)
    scores.update(dispersion)
    scores["mean adjacent change (m/s)"] = _mean_adjacent_change(pred)
    scores["time seam jump ratio"] = (
        _seam_jump_ratio(pred, seam_period) if seam_period is not None else np.nan
    )
    return scores


def _write_report(rows: dict[str, dict], output: Path) -> None:
    metrics = [
        ("SR_time", "lower"),
        ("disp log-MSD RMSE", "lower"),
        ("final spread ratio", "near 1"),
        ("mean adjacent change (m/s)", "match ERA5"),
        ("time seam jump ratio", "near 1"),
    ]
    lines = [
        "# Shared-protocol temporal wind benchmark",
        "",
        "Four independent 24-hour episodes are evaluated: 8 January, April, July, "
        "and October 2023, beginning at 00 UTC. Every conditional method uses the "
        "same seed, physical conditions, 16x16 grid at 50 km spacing, and 60-130 hPa "
        "pressure coordinates. ERA5 day 8 is the reference; ERA5 day 9 provides an "
        "independent seasonal self-split floor.",
        "",
        "Direct base concatenates six independently sampled four-hour blocks. "
        "Every other conditional generator produces each full 24-hour query as one field. "
        "BLE-VAE is listed as N/A because its nine decoder slices have no established "
        "physical hourly spacing; assigning one would make the temporal comparison arbitrary.",
        "",
        "SR_time compares temporal power spectra. Dispersion metrics advect passive "
        "agents through each field. Mean adjacent change diagnoses over-smoothing. "
        "The seam ratio compares jumps at method boundaries with all hourly jumps; "
        "one means boundaries are not unusually discontinuous.",
        "",
        "| Metric | " + " | ".join(rows) + " |",
        "|---|" + "---|" * len(rows),
    ]
    for metric, direction in metrics:
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
    parser.add_argument("--model-runs", required=True, nargs="+", type=Path)
    parser.add_argument("--model-names", required=True, nargs="+")
    parser.add_argument("--seam-periods", required=True, nargs="+", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if not (
        len(args.model_runs) == len(args.model_names) == len(args.seam_periods)
    ):
        parser.error("--model-runs, --model-names, and --seam-periods must match")

    lat, lon = common_grid(
        CENTER_LAT,
        CENTER_LON,
        GRID_SPACING_KM,
        GRID_SIZE,
    )
    reference = _load_reference(args.reference, 8, lat, lon)
    floor = _load_reference(args.reference, 9, lat, lon)
    rows = {"ERA5 self-split floor": _score(floor, reference, None)}
    for name, path, period in zip(
        args.model_names,
        args.model_runs,
        args.seam_periods,
    ):
        rows[name] = _score(_load_run(path, lat, lon), reference, period)
    rows["BLE-VAE"] = {
        "SR_time": np.nan,
        "disp log-MSD RMSE": np.nan,
        "final spread ratio": np.nan,
        "mean adjacent change (m/s)": np.nan,
        "time seam jump ratio": np.nan,
    }
    _write_report(rows, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
