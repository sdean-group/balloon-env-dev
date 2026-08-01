"""Publication-quality matched visual comparisons for tile-scaling runs."""
from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import xarray as xr

try:
    from .protocol import PROFILES
except ImportError:  # standalone rendering/test execution
    from protocol import PROFILES


DEFAULT_MONTHS = (1, 4, 7, 10)
METHOD_LABELS = {
    4: "4 cores (64x64 windows)",
    16: "16 cores (32x32 windows)",
    64: "64 cores (16x16 windows)",
}


def _axis_indices(source: np.ndarray, target: np.ndarray, name: str) -> np.ndarray:
    indices = []
    for value in target:
        index = int(np.argmin(np.abs(source - value)))
        if not np.isclose(source[index], value, atol=1e-6):
            raise ValueError(f"reference has no exact {name} coordinate for {value}")
        indices.append(index)
    return np.asarray(indices)


def _load_records(path: Path) -> tuple[dict, list[dict]]:
    config = json.loads((path / "config.json").read_text())
    files = sorted(path.glob("m*_d*_h*_s*.npz"))
    if len(files) != int(config["conditions"]):
        raise ValueError(
            f"{path}: expected {config['conditions']} files, found {len(files)}"
        )
    records = []
    for file in files:
        with np.load(file) as data:
            records.append(
                {
                    "u": np.asarray(data["u"][0], dtype=np.float32),
                    "v": np.asarray(data["v"][0], dtype=np.float32),
                    "levels": np.asarray(data["levels"]),
                    "lat": np.asarray(data["lat"]),
                    "lon": np.asarray(data["lon"]),
                    "month": int(data["month"]),
                    "day": int(data["day"]),
                    "hour": int(data["hour"]),
                    "seed": int(data["seed"]),
                }
            )
    return config, records


def _record_lookup(records: Iterable[dict]) -> dict[tuple[int, int, int, int], dict]:
    return {
        (record["month"], record["day"], record["hour"], record["seed"]): record
        for record in records
    }


def _select_keys(
    records: list[dict],
    *,
    months: tuple[int, ...],
    day: int,
    hour: int,
    seed: int,
) -> list[tuple[int, int, int, int]]:
    available = _record_lookup(records)
    keys = [(month, day, hour, seed) for month in months]
    missing = [key for key in keys if key not in available]
    if missing:
        raise ValueError(f"missing requested visual condition: {missing[0]}")
    return keys


def _load_reference(
    path: Path,
    exemplar: dict,
    keys: list[tuple[int, int, int, int]],
) -> dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]]:
    source = xr.open_zarr(path, consolidated=False)
    try:
        yi = _axis_indices(
            np.asarray(source["lat"].values), exemplar["lat"], "latitude"
        )
        xi = _axis_indices(
            np.asarray(source["lon"].values), exemplar["lon"], "longitude"
        )
        times = np.asarray(source["time"].values).astype("datetime64[h]")
        time_lookup = {value: index for index, value in enumerate(times)}
        result = {}
        for month, day, hour, _ in keys:
            timestamp = np.datetime64(
                f"2023-{month:02d}-{day:02d}T{hour:02d}", "h"
            )
            if timestamp not in time_lookup:
                raise ValueError(f"ERA5 reference is missing {timestamp}")
            block = source.isel(time=time_lookup[timestamp], y=yi, x=xi)
            result[(month, day, hour)] = (
                np.asarray(block["u"].values, dtype=np.float32),
                np.asarray(block["v"].values, dtype=np.float32),
            )
        return result
    finally:
        source.close()


def _robust_limit(fields: Iterable[tuple[np.ndarray, np.ndarray]]) -> float:
    values = [np.hypot(u, v).ravel() for u, v in fields]
    return max(float(np.quantile(np.concatenate(values), 0.99)), 1.0)


def _draw_boundaries(
    ax: plt.Axes,
    *,
    stride: int,
    size: int,
    color: str = "white",
) -> None:
    for boundary in range(stride, size, stride):
        ax.axhline(
            boundary - 0.5, color=color, linewidth=0.75, alpha=0.75, zorder=3
        )
        ax.axvline(
            boundary - 0.5, color=color, linewidth=0.75, alpha=0.75, zorder=3
        )


def _draw_wind(
    ax: plt.Axes,
    u: np.ndarray,
    v: np.ndarray,
    *,
    limit: float,
    stride: int | None,
    title: str,
) -> None:
    speed = np.hypot(u, v)
    image = ax.imshow(
        speed,
        origin="lower",
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=limit),
        interpolation="nearest",
    )
    step = max(1, speed.shape[-1] // 12)
    y, x = np.mgrid[0:speed.shape[-2]:step, 0:speed.shape[-1]:step]
    ax.quiver(
        x,
        y,
        u[::step, ::step],
        v[::step, ::step],
        color="white",
        alpha=0.85,
        angles="xy",
        scale_units="xy",
        scale=max(limit / 1.8, 1.0),
        width=0.003,
        headwidth=3.5,
        zorder=2,
    )
    if stride is not None:
        _draw_boundaries(ax, stride=stride, size=speed.shape[-1])
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def _seasonal_atlas(
    records: dict[int, dict[tuple[int, int, int, int], dict]],
    reference: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
    keys: list[tuple[int, int, int, int]],
    *,
    level_index: int,
    output: Path,
) -> None:
    all_fields = []
    for month, day, hour, seed in keys:
        all_fields.append(
            tuple(values[level_index] for values in reference[(month, day, hour)])
        )
        for count in sorted(records):
            record = records[count][(month, day, hour, seed)]
            all_fields.append((record["u"][level_index], record["v"][level_index]))
    limit = _robust_limit(all_fields)

    fig, axes = plt.subplots(
        4,
        len(keys),
        figsize=(3.15 * len(keys), 11.0),
        constrained_layout=True,
    )
    row_labels = ["ERA5", *(METHOD_LABELS[count] for count in sorted(records))]
    last_image = None
    for column, (month, day, hour, seed) in enumerate(keys):
        fields = [
            tuple(values[level_index] for values in reference[(month, day, hour)]),
            *[
                (
                    records[count][(month, day, hour, seed)]["u"][level_index],
                    records[count][(month, day, hour, seed)]["v"][level_index],
                )
                for count in sorted(records)
            ],
        ]
        for row, (u, v) in enumerate(fields):
            count = None if row == 0 else sorted(records)[row - 1]
            last_image = _draw_wind(
                axes[row, column],
                u,
                v,
                limit=limit,
                stride=None if count is None else PROFILES[count].stride,
                title=(
                    f"2023-{month:02d}-{day:02d} {hour:02d}:00 UTC"
                    if row == 0
                    else ""
                ),
            )
            if column == 0:
                axes[row, column].set_ylabel(row_labels[row], fontsize=10)
    level = next(iter(records[4].values()))["levels"][level_index]
    fig.suptitle(
        f"Condition-matched wind fields at model level {level}; "
        "white lines are generated tile-core boundaries",
        fontsize=12,
    )
    fig.colorbar(
        last_image,
        ax=axes,
        orientation="horizontal",
        shrink=0.58,
        pad=0.025,
        label="Wind speed (m/s), common scale",
    )
    fig.savefig(output / "eye_test_seasons.png", dpi=220)
    plt.close(fig)


def _multilevel_atlas(
    records: dict[int, dict[tuple[int, int, int, int], dict]],
    reference: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
    key: tuple[int, int, int, int],
    *,
    level_indices: tuple[int, ...],
    output: Path,
) -> None:
    month, day, hour, seed = key
    methods = [None, *sorted(records)]
    all_fields = []
    for level_index in level_indices:
        all_fields.append(
            tuple(values[level_index] for values in reference[(month, day, hour)])
        )
        for count in sorted(records):
            record = records[count][key]
            all_fields.append((record["u"][level_index], record["v"][level_index]))
    limit = _robust_limit(all_fields)

    fig, axes = plt.subplots(
        len(level_indices),
        len(methods),
        figsize=(12.8, 3.05 * len(level_indices)),
        constrained_layout=True,
    )
    exemplar = next(iter(records[4].values()))
    last_image = None
    for row, level_index in enumerate(level_indices):
        for column, count in enumerate(methods):
            if count is None:
                u, v = (
                    values[level_index]
                    for values in reference[(month, day, hour)]
                )
                title = "ERA5"
                stride = None
            else:
                record = records[count][key]
                u, v = record["u"][level_index], record["v"][level_index]
                title = METHOD_LABELS[count]
                stride = PROFILES[count].stride
            last_image = _draw_wind(
                axes[row, column],
                u,
                v,
                limit=limit,
                stride=stride,
                title=title if row == 0 else "",
            )
            if column == 0:
                axes[row, column].set_ylabel(
                    f"Model level {exemplar['levels'][level_index]}", fontsize=10
                )
    fig.suptitle(
        f"Vertical eye test: 2023-{month:02d}-{day:02d} {hour:02d}:00 UTC",
        fontsize=12,
    )
    fig.colorbar(
        last_image,
        ax=axes,
        orientation="horizontal",
        shrink=0.58,
        pad=0.025,
        label="Wind speed (m/s), common scale",
    )
    fig.savefig(output / "eye_test_levels.png", dpi=220)
    plt.close(fig)


def _boundary_atlas(
    records: dict[int, dict[tuple[int, int, int, int], dict]],
    reference: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
    key: tuple[int, int, int, int],
    *,
    level_index: int,
    output: Path,
) -> None:
    month, day, hour, seed = key
    methods = [None, *sorted(records)]
    fields = [
        tuple(values[level_index] for values in reference[(month, day, hour)]),
        *[
            (
                records[count][key]["u"][level_index],
                records[count][key]["v"][level_index],
            )
            for count in sorted(records)
        ],
    ]
    limit = _robust_limit(fields)
    fig, axes = plt.subplots(1, len(methods), figsize=(13.0, 3.4), constrained_layout=True)
    last_image = None
    for ax, count, (u, v) in zip(axes, methods, fields):
        last_image = _draw_wind(
            ax,
            u,
            v,
            limit=limit,
            stride=None if count is None else PROFILES[count].stride,
            title="ERA5" if count is None else METHOD_LABELS[count],
        )
        ax.set_xlim(15.5, 47.5)
        ax.set_ylim(15.5, 47.5)
    fig.suptitle(
        f"Boundary-focused center crop: 2023-{month:02d}-{day:02d} "
        f"{hour:02d}:00 UTC",
        fontsize=12,
    )
    fig.colorbar(
        last_image,
        ax=axes,
        orientation="horizontal",
        shrink=0.6,
        pad=0.04,
        label="Wind speed (m/s), common scale",
    )
    fig.savefig(output / "eye_test_boundaries.png", dpi=240)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("CORE_TILES", "PATH"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--day", type=int, default=12)
    parser.add_argument("--hour", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level-index", type=int, default=9)
    args = parser.parse_args(argv)

    records_by_count = {}
    for count_text, path_text in args.run:
        count = int(count_text)
        if count not in PROFILES:
            parser.error(f"unsupported core tile count {count}")
        _, loaded = _load_records(Path(path_text))
        records_by_count[count] = _record_lookup(loaded)
    if set(records_by_count) != set(PROFILES):
        parser.error("provide exactly one --run for 4, 16, and 64 core tiles")

    exemplar_records = list(records_by_count[4].values())
    keys = _select_keys(
        exemplar_records,
        months=DEFAULT_MONTHS,
        day=args.day,
        hour=args.hour,
        seed=args.seed,
    )
    exemplar = exemplar_records[0]
    level_count = exemplar["u"].shape[0]
    if not 0 <= args.level_index < level_count:
        parser.error(f"--level-index must be in [0, {level_count - 1}]")
    reference = _load_reference(args.reference, exemplar, keys)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _seasonal_atlas(
        records_by_count,
        reference,
        keys,
        level_index=args.level_index,
        output=args.output_dir,
    )
    _multilevel_atlas(
        records_by_count,
        reference,
        keys[0],
        level_indices=(0, level_count // 2, level_count - 1),
        output=args.output_dir,
    )
    _boundary_atlas(
        records_by_count,
        reference,
        keys[0],
        level_index=args.level_index,
        output=args.output_dir,
    )
    for name in (
        "eye_test_seasons.png",
        "eye_test_levels.png",
        "eye_test_boundaries.png",
    ):
        print(args.output_dir / name)


if __name__ == "__main__":
    main()
