"""Evaluate RL-HAB SynthWinds on the shared 24-hour temporal protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from ... import artifact
from ...benchmark_shared_baselines import (
    CENTER_LAT,
    CENTER_LON,
    GRID_SIZE,
    GRID_SPACING_KM,
    _standard_model_pressure_hpa,
    _to_shared_grid,
)
from ...benchmark_temporal_all_methods import _load_reference, _score
from ...metrics.temporal import contiguous_segments
from ...resample import common_grid

MONTHS = (1, 4, 7, 10)


def _hourly_rlhab(source: xr.Dataset, lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    """Interpolate the native 12-hour RL-HAB updates into four 24-hour episodes."""
    shared = _to_shared_grid(
        source,
        _standard_model_pressure_hpa(source["level"].values),
        lat,
        lon,
        source_spacing_km=27.83,
    )
    episodes = []
    for month in MONTHS:
        start = np.datetime64(f"2023-{month:02d}-08T00", "ns")
        stop = np.datetime64(f"2023-{month:02d}-09T00", "ns")
        anchors = shared.sel(time=slice(start, stop)).load()
        expected = np.asarray(
            [start, start + np.timedelta64(12, "h"), stop], dtype="datetime64[ns]"
        )
        if not np.array_equal(anchors.time.values.astype("datetime64[ns]"), expected):
            raise ValueError(
                f"RL-HAB month {month} needs anchors at day 8 00/12 and day 9 00; "
                f"found {anchors.time.values}"
            )
        target = np.arange(start, stop, np.timedelta64(1, "h")).astype("datetime64[ns]")
        episodes.append(anchors.interp(time=target))
    combined = xr.concat(episodes, dim="time").sortby("time")
    if len(contiguous_segments(combined)) != len(MONTHS):
        raise ValueError("interpolated RL-HAB field does not contain four 24-hour episodes")
    return combined


def evaluate(reference_path: Path, synth_path: Path, output: Path) -> dict:
    lat, lon = common_grid(CENTER_LAT, CENTER_LON, GRID_SPACING_KM, GRID_SIZE)
    reference = _load_reference(reference_path, 8, lat, lon)
    floor = _load_reference(reference_path, 9, lat, lon)
    rlhab = _hourly_rlhab(artifact.read(synth_path), lat, lon)

    rows = {
        "ERA5 self-split floor": _score(floor, reference, None),
        "RL-HAB SynthWinds": _score(rlhab, reference, 12),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")

    metrics = [
        ("SR_time", "lower"),
        ("disp log-MSD RMSE", "lower"),
        ("final spread ratio", "near 1"),
        ("mean adjacent change (m/s)", "match ERA5"),
        ("time seam jump ratio", "near 1"),
    ]
    lines = [
        "# RL-HAB SynthWinds temporal evaluation",
        "",
        "The shared protocol uses four 24-hour episodes beginning at 00 UTC on "
        "8 January, April, July, and October 2023. ERA5 day 8 is the reference and "
        "day 9 is the seasonal self-split floor.",
        "",
        "RL-HAB SynthWinds updates from radiosondes every 12 hours. Its fields are "
        "linearly interpolated between day-8 00/12 UTC and day-9 00 UTC to obtain the "
        "hourly sequence required by the temporal metrics. These scores therefore "
        "measure the standard interpolation-based RL-HAB environment, not learned "
        "hourly dynamics.",
        "",
        "| Metric | ERA5 self-split floor | RL-HAB SynthWinds |",
        "|---|---:|---:|",
    ]
    for metric, direction in metrics:
        values = []
        for row in rows.values():
            value = row.get(metric, np.nan)
            values.append("N/A" if not np.isfinite(value) else f"{value:.4f}")
        lines.append(f"| {metric} ({direction}) | " + " | ".join(values) + " |")
    output.write_text("\n".join(lines) + "\n")
    print(output.read_text())
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--synthwinds", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    evaluate(args.reference, args.synthwinds, args.output)


if __name__ == "__main__":
    main()
