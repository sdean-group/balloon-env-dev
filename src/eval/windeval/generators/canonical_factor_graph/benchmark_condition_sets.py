"""Compare CFGD and Infinite Diffusion condition sets on their exact common ERA5 crop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from ... import artifact
from ...metrics import run_suite
from ...metrics.distributions import conditional_w1_grouped
from ...metrics.suite import METRIC_INFO
from .paired_statistics import bootstrap_summary


def _files(path: Path) -> list[Path]:
    files = sorted(path.glob("m*_d*_h*_s*.npz"))
    config = json.loads((path / "config.json").read_text())
    expected = int(config["conditions"])
    if len(files) != expected:
        raise ValueError(f"{path}: expected {expected} blocks, found {len(files)}")
    return files


def _common_coordinates(runs: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    latitudes = []
    longitudes = []
    for run in runs:
        first = np.load(_files(run)[0])
        latitudes.append(np.asarray(first["lat"], dtype=np.float64))
        longitudes.append(np.asarray(first["lon"], dtype=np.float64))
    common_lat = latitudes[0]
    common_lon = longitudes[0]
    for values in latitudes[1:]:
        common_lat = np.intersect1d(np.round(common_lat, 6), np.round(values, 6))
    for values in longitudes[1:]:
        common_lon = np.intersect1d(np.round(common_lon, 6), np.round(values, 6))
    if len(common_lat) < 8 or len(common_lon) < 8:
        raise ValueError("runs do not share a sufficiently large spatial region")
    return common_lat, common_lon


def _condition_key(file: Path) -> tuple[int, int, int, int]:
    z = np.load(file)
    return int(z["month"]), int(z["day"]), int(z["hour"]), int(z["seed"])


def _common_conditions(runs: list[Path]) -> set[tuple[int, int, int, int]]:
    key_sets = [{_condition_key(file) for file in _files(run)} for run in runs]
    common = set.intersection(*key_sets)
    if not common:
        raise ValueError("runs do not share any condition/seed combinations")
    return common


def _indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    distance = np.abs(source[:, None] - target[None, :])
    index = np.argmin(distance, axis=0)
    if np.any(distance[index, np.arange(len(target))] > 1e-5):
        raise ValueError("coordinate arrays do not contain the requested common crop")
    return index


def _reference(path: Path, lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    ds = xr.open_zarr(path, consolidated=False)
    yi = _indices(ds["lat"].values, lat)
    xi = _indices(ds["lon"].values, lon)
    return ds.isel(y=yi, x=xi).sortby("lat").load()


def _load_run(
    path: Path,
    ref: xr.Dataset,
    conditions: set[tuple[int, int, int, int]],
) -> list[dict]:
    records = []
    for file in _files(path):
        z = np.load(file)
        key = int(z["month"]), int(z["day"]), int(z["hour"]), int(z["seed"])
        if key not in conditions:
            continue
        yi = _indices(z["lat"], ref["lat"].values)
        xi = _indices(z["lon"], ref["lon"].values)
        u = np.take(np.take(z["u"][0], yi, axis=1), xi, axis=2)
        v = np.take(np.take(z["v"][0], yi, axis=1), xi, axis=2)
        records.append(
            {
                "u": np.asarray(u, dtype=np.float32),
                "v": np.asarray(v, dtype=np.float32),
                "levels": np.asarray(z["levels"]),
                "month": int(z["month"]),
                "day": int(z["day"]),
                "hour": int(z["hour"]),
                "seed": int(z["seed"]),
            }
        )
    if not records:
        raise ValueError(f"{path}: no records matched the common conditions")
    return records


def _score_records(records: list[dict], ref: xr.Dataset) -> dict:
    first = records[0]
    if not np.array_equal(first["levels"].astype(int), ref["level"].values.astype(int)):
        raise ValueError("generated levels do not match ERA5")
    pooled = artifact.make_field(
        np.stack([record["u"] for record in records]),
        np.stack([record["v"] for record in records]),
        level=first["levels"],
        lat=ref["lat"].values,
        lon=ref["lon"].values,
        time=np.arange(len(records)),
    )
    groups = []
    for month, hour in sorted({(record["month"], record["hour"]) for record in records}):
        matching = [
            record for record in records
            if record["month"] == month and record["hour"] == hour
        ]
        seeds = [
            artifact.make_field(
                record["u"],
                record["v"],
                level=record["levels"],
                lat=ref["lat"].values,
                lon=ref["lon"].values,
            )
            for record in matching
        ]
        reference = ref.sel(time=(ref.time.dt.month == month) & (ref.time.dt.hour == hour))
        groups.append((seeds, reference))
    scores, _ = run_suite(pooled, _integer_time(ref))
    scores.update(conditional_w1_grouped(groups))
    return scores


def _integer_time(ds: xr.Dataset) -> xr.Dataset:
    return ds.assign_coords(time=np.arange(ds.sizes["time"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--runs", required=True, nargs="+", type=Path)
    parser.add_argument("--names", required=True, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.runs) != len(args.names):
        parser.error("--runs and --names must have equal lengths")

    lat, lon = _common_coordinates(args.runs)
    common_conditions = _common_conditions(args.runs)
    ref = _reference(args.reference, lat, lon)
    rows = {}
    early = _integer_time(ref.sel(time=ref.time.dt.day < 11))
    late = _integer_time(ref.sel(time=ref.time.dt.day >= 11))
    floor, _ = run_suite(early, late)
    floor_groups = []
    for month in (1, 4, 7, 10):
        for hour in (0, 12):
            a = ref.sel(
                time=(ref.time.dt.month == month)
                & (ref.time.dt.hour == hour)
                & (ref.time.dt.day < 11)
            )
            b = ref.sel(
                time=(ref.time.dt.month == month)
                & (ref.time.dt.hour == hour)
                & (ref.time.dt.day >= 11)
            )
            floor_groups.append(([a], b))
    floor.update(conditional_w1_grouped(floor_groups))
    rows["ERA5 self-split floor"] = floor

    records_by_name = {}
    per_seed_scores = {}
    for name, run in zip(args.names, args.runs):
        records = _load_run(run, ref, common_conditions)
        records_by_name[name] = records
        rows[name] = _score_records(records, ref)
        per_seed_scores[name] = {
            seed: _score_records(
                [record for record in records if record["seed"] == seed],
                ref,
            )
            for seed in sorted({record["seed"] for record in records})
        }

    metrics = [
        "SR_E", "SR_div", "SR_vort", "L_eff (km)",
        "W1 u (m/s)", "W1 v (m/s)", "tail err 1% (m/s)",
        "tail err 0.1% (m/s)", "W1 cond (m/s)",
    ]
    lines = [
        "# Common-crop held-out wind benchmark",
        "",
        f"Exact common region: {len(lat)}x{len(lon)} cells, "
        f"{float(lat[0]):.2f}-{float(lat[-1]):.2f}N, "
        f"{float(lon[0]):.2f}-{float(lon[-1]):.2f}E. "
        f"Shared generated conditions: {len(common_conditions)}. "
        "Only frame 0 is scored. Distribution and tail metrics are directly useful; "
        "spectral and effective-resolution metrics have limited dynamic range on this crop.",
        "",
        "| Metric | " + " | ".join(rows) + " |",
        "|---|" + "---|" * len(rows),
    ]
    for metric in metrics:
        direction, _ = METRIC_INFO[metric]
        values = [
            "N/A" if not np.isfinite(row.get(metric, np.nan)) else f"{row[metric]:.4f}"
            for row in rows.values()
        ]
        lines.append(f"| {metric} ({direction}) | " + " | ".join(values) + " |")

    paired_report = None
    if len(args.names) == 2:
        baseline, alternative = args.names
        seeds = sorted(
            set(per_seed_scores[baseline]) & set(per_seed_scores[alternative])
        )
        if len(seeds) < 2:
            raise ValueError("paired comparison requires at least two common seeds")
        paired_metrics = {}
        for metric in metrics:
            deltas = [
                per_seed_scores[alternative][seed][metric]
                - per_seed_scores[baseline][seed][metric]
                for seed in seeds
            ]
            if all(np.isfinite(deltas)):
                paired_metrics[metric] = bootstrap_summary(deltas)
        paired_report = {
            "baseline": baseline,
            "alternative": alternative,
            "delta": f"{alternative} minus {baseline}",
            "interpretation": (
                "Negative is better for every reported lower-is-better metric."
            ),
            "seeds": seeds,
            "metrics": paired_metrics,
            "per_seed_scores": per_seed_scores,
        }
        lines.extend([
            "",
            f"## Paired seed analysis: {alternative} minus {baseline}",
            "",
            "Negative deltas favor the alternative. Intervals are paired percentile "
            "bootstrap intervals across seeds.",
            "",
            "| Metric | Mean delta | 95% CI |",
            "|---|---:|---:|",
        ])
        for metric in metrics:
            if metric not in paired_metrics:
                continue
            summary = paired_metrics[metric]
            lines.append(
                f"| {metric} | {summary['mean']:.4f} | "
                f"[{summary['ci95_low']:.4f}, {summary['ci95_high']:.4f}] |"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    if paired_report is not None:
        args.output.with_suffix(".paired.json").write_text(
            json.dumps(paired_report, indent=2) + "\n"
        )
    print(args.output)


if __name__ == "__main__":
    main()
