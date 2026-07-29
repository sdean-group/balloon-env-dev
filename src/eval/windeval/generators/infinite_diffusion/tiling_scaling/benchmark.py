"""Score and plot fixed-domain 4/16/64-core-tile InfiniteDiffusion runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from windeval import artifact
from windeval.metrics.distributions import (
    conditional_w1_grouped,
    extreme_quantile_error,
    marginal_w1,
)
from windeval.metrics.spectra import spatial_spectral_suite
from windeval.reference import SPLIT_DAY

from .protocol import PROFILES, boundary_coherence, coherence_length_km

QUALITY_METRICS = (
    "SR_E",
    "SR_div",
    "SR_vort",
    "L_eff (km)",
    "W1 u (m/s)",
    "W1 v (m/s)",
    "tail err 1% (m/s)",
    "tail err 0.1% (m/s)",
    "W1 cond (m/s)",
)


def _condition_files(path: Path) -> tuple[dict, list[Path]]:
    config = json.loads((path / "config.json").read_text())
    files = sorted(path.glob("m*_d*_h*_s*.npz"))
    if len(files) != int(config["conditions"]):
        raise ValueError(f"{path}: expected {config['conditions']} files, found {len(files)}")
    return config, files


def _load_records(path: Path) -> tuple[dict, list[dict]]:
    config, files = _condition_files(path)
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
                    "seconds": float(data["seconds"]),
                    "peak_memory_mb": float(data["peak_memory_mb"]),
                    "model_window_calls": int(data["model_window_calls"]),
                    "model_forward_evaluations": int(data["model_forward_evaluations"]),
                    "final_windows": int(data["final_windows"]),
                    "file": file,
                }
            )
    return config, records


def _keys(records: list[dict]) -> set[tuple[int, int, int, int]]:
    return {(r["month"], r["day"], r["hour"], r["seed"]) for r in records}


def _axis_indices(source: np.ndarray, target: np.ndarray, name: str) -> np.ndarray:
    indices = []
    for value in target:
        index = int(np.argmin(np.abs(source - value)))
        if not np.isclose(source[index], value, atol=1e-6):
            raise ValueError(f"reference has no exact {name} coordinate for {value}")
        indices.append(index)
    return np.asarray(indices)


def _reference_for_records(path: Path, records: list[dict]) -> xr.Dataset:
    source = xr.open_zarr(path, consolidated=False)
    lat = records[0]["lat"]
    lon = records[0]["lon"]
    yi = _axis_indices(np.asarray(source["lat"].values), lat, "latitude")
    xi = _axis_indices(np.asarray(source["lon"].values), lon, "longitude")
    wanted = sorted(
        {
            np.datetime64(
                f"2023-{r['month']:02d}-{r['day']:02d}T{r['hour']:02d}",
                "h",
            )
            for r in records
        }
    )
    available = np.asarray(source["time"].values).astype("datetime64[h]")
    lookup = {value: index for index, value in enumerate(available)}
    missing = [value for value in wanted if value not in lookup]
    if missing:
        raise ValueError(f"reference is missing timestamps, first={missing[0]}")
    ti = np.asarray([lookup[value] for value in wanted])
    selected = source.isel(time=ti, y=yi, x=xi).load()
    result = artifact.make_field(
        selected["u"].values,
        selected["v"].values,
        level=selected["level"].values,
        lat=lat,
        lon=lon,
        time=selected["time"].values,
    )
    source.close()
    return result


def _pool(records: list[dict]) -> xr.Dataset:
    return artifact.make_field(
        np.stack([r["u"] for r in records]),
        np.stack([r["v"] for r in records]),
        level=records[0]["levels"],
        lat=records[0]["lat"],
        lon=records[0]["lon"],
        time=np.arange(len(records)),
    )


def _condition_groups(records: list[dict], reference: xr.Dataset):
    groups = []
    for month, hour in sorted({(r["month"], r["hour"]) for r in records}):
        matching = [r for r in records if r["month"] == month and r["hour"] == hour]
        samples = [
            artifact.make_field(
                r["u"],
                r["v"],
                level=r["levels"],
                lat=r["lat"],
                lon=r["lon"],
            )
            for r in matching
        ]
        ref = reference.sel(
            time=(reference.time.dt.month == month) & (reference.time.dt.hour == hour)
        )
        groups.append((samples, ref))
    return groups


def _score(prediction: xr.Dataset, reference: xr.Dataset) -> tuple[dict, dict]:
    scores, detail = spatial_spectral_suite(prediction, reference)
    marginal, marginal_detail = marginal_w1(prediction, reference)
    scores.update(marginal)
    scores.update(extreme_quantile_error(prediction, reference))
    detail["marginals"] = marginal_detail
    return scores, detail


def _floor(reference: xr.Dataset) -> dict:
    early = reference.sel(time=reference.time.dt.day < SPLIT_DAY)
    late = reference.sel(time=reference.time.dt.day >= SPLIT_DAY)
    scores, _ = _score(early, late)
    groups = []
    for month in sorted(set(reference.time.dt.month.values.tolist())):
        for hour in sorted(set(reference.time.dt.hour.values.tolist())):
            a = early.sel(
                time=(early.time.dt.month == month) & (early.time.dt.hour == hour)
            )
            b = late.sel(
                time=(late.time.dt.month == month) & (late.time.dt.hour == hour)
            )
            groups.append(([a], b))
    scores.update(conditional_w1_grouped(groups))
    return scores


def _performance(records: list[dict]) -> dict:
    seconds = np.asarray([r["seconds"] for r in records])
    memory = np.asarray([r["peak_memory_mb"] for r in records])
    return {
        "samples": len(records),
        "median seconds": float(np.median(seconds)),
        "mean seconds": float(np.mean(seconds)),
        "p10 seconds": float(np.quantile(seconds, 0.1)),
        "p90 seconds": float(np.quantile(seconds, 0.9)),
        "median peak GPU memory (MB)": (
            float(np.nanmedian(memory)) if np.isfinite(memory).any() else np.nan
        ),
        "median model window calls": float(
            np.median([r["model_window_calls"] for r in records])
        ),
        "median model forward evaluations": float(
            np.median([r["model_forward_evaluations"] for r in records])
        ),
        "final overlapping windows": int(records[0]["final_windows"]),
    }


def _effective_spacing_km(lat: np.ndarray, lon: np.ndarray) -> float:
    """Mean physical spacing of the latitude and longitude grid directions."""
    dy = float(np.mean(np.abs(np.diff(lat)))) * 111.32
    dx = (
        float(np.mean(np.abs(np.diff(lon))))
        * 111.32
        * np.cos(np.deg2rad(float(np.mean(lat))))
    )
    return 0.5 * (dx + dy)


def _fmt(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.4f}"


def _plot_runtime(performance: dict[int, dict], output: Path) -> None:
    tiles = np.asarray(sorted(performance))
    median = np.asarray([performance[n]["median seconds"] for n in tiles])
    low = median - np.asarray([performance[n]["p10 seconds"] for n in tiles])
    high = np.asarray([performance[n]["p90 seconds"] for n in tiles]) - median
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.errorbar(tiles, median, yerr=np.stack((low, high)), marker="o", capsize=4)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(tiles, [str(n) for n in tiles])
    ax.set_xlabel("Non-overlapping core tiles in fixed 64x64 output")
    ax.set_ylabel("Cold generation time per sample (s)")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "runtime_vs_tiles.png", dpi=180)
    plt.close(fig)


def _plot_quality(scores: dict[int, dict], floor: dict, output: Path) -> None:
    tiles = np.asarray(sorted(scores))
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 4.5), constrained_layout=True
    )
    for metric in ("SR_E", "SR_div", "SR_vort"):
        line, = axes[0].plot(
            tiles, [scores[n][metric] for n in tiles], marker="o", label=metric
        )
        axes[0].axhline(
            floor[metric], color=line.get_color(), linestyle="--", alpha=0.45
        )
    for metric in ("W1 u (m/s)", "W1 v (m/s)"):
        line, = axes[1].plot(
            tiles, [scores[n][metric] for n in tiles], marker="o", label=metric
        )
        axes[1].axhline(
            floor[metric], color=line.get_color(), linestyle="--", alpha=0.45
        )
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(tiles, [str(n) for n in tiles])
        ax.set_xlabel("Core tiles")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Spectral residual (lower is better)")
    axes[1].set_ylabel("Wasserstein-1 (m/s; lower is better)")
    fig.suptitle("Solid: generated; dashed: ERA5 self-split floor", fontsize=10)
    fig.savefig(output / "quality_vs_tiles.png", dpi=180)
    plt.close(fig)


def _plot_coherence(coherence: dict[int, dict], output: Path) -> None:
    tiles = np.asarray(sorted(coherence))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(
        tiles,
        [coherence[n]["model"]["boundary jump ratio"] for n in tiles],
        marker="o",
        label="Generated",
    )
    axes[0].plot(
        tiles,
        [coherence[n]["ERA5"]["boundary jump ratio"] for n in tiles],
        marker="o",
        label="ERA5 at same lines",
    )
    axes[0].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[0].set_ylabel("Boundary jump ratio (ideal near 1)")
    model_lengths = [coherence[n]["model"]["coherence length 0.5 (km)"] for n in tiles]
    ref_lengths = [coherence[n]["ERA5"]["coherence length 0.5 (km)"] for n in tiles]
    axes[1].plot(tiles, model_lengths, marker="o", label="Generated")
    axes[1].plot(tiles, ref_lengths, marker="o", label="ERA5")
    axes[1].set_ylabel("Vector correlation length at 0.5 (km)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(tiles, [str(n) for n in tiles])
        ax.set_xlabel("Core tiles")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "coherence_vs_tiles.png", dpi=180)
    plt.close(fig)


def _plot_tradeoff(scores: dict[int, dict], performance: dict[int, dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for tiles in sorted(scores):
        x = performance[tiles]["median seconds"]
        y = scores[tiles]["SR_E"]
        ax.scatter(x, y, s=65)
        ax.annotate(f"{tiles} cores", (x, y), xytext=(6, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Median cold generation time (s)")
    ax.set_ylabel("SR_E (lower is better)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "quality_runtime_tradeoff.png", dpi=180)
    plt.close(fig)


def _plot_samples(records: dict[int, list[dict]], output: Path) -> None:
    tiles = sorted(records)
    samples = []
    for count in tiles:
        record = records[count][0]
        level = record["u"].shape[0] // 2
        samples.append(np.hypot(record["u"][level], record["v"][level]))
    vmin = min(float(speed.min()) for speed in samples)
    vmax = max(float(speed.max()) for speed in samples)
    fig, axes = plt.subplots(1, len(tiles), figsize=(13, 4.2), constrained_layout=True)
    for ax, count, speed in zip(axes, tiles, samples):
        image = ax.imshow(
            speed, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
        )
        stride = PROFILES[count].stride
        for boundary in range(stride, speed.shape[0], stride):
            ax.axhline(boundary - 0.5, color="white", alpha=0.6, linewidth=0.7)
            ax.axvline(boundary - 0.5, color="white", alpha=0.6, linewidth=0.7)
        ax.set_title(
            f"{count} cores\n{PROFILES[count].window}x{PROFILES[count].window} windows"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, label="m/s")
    fig.savefig(output / "matched_sample_montage.png", dpi=180)
    plt.close(fig)


def _write_csv(
    scores: dict[int, dict],
    performance: dict[int, dict],
    coherence: dict[int, dict],
    output: Path,
) -> None:
    fields = [
        "core_tiles", "window", "stride", "final_windows",
        "median_seconds", "median_forward_evaluations",
        "median_peak_gpu_memory_mb",
        "SR_E", "SR_div", "SR_vort", "L_eff_km", "W1_u", "W1_v",
        "boundary_jump_ratio", "ERA5_boundary_jump_ratio",
        "coherence_length_km", "ERA5_coherence_length_km",
    ]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for count in sorted(scores):
            writer.writerow(
                {
                    "core_tiles": count,
                    "window": PROFILES[count].window,
                    "stride": PROFILES[count].stride,
                    "final_windows": performance[count]["final overlapping windows"],
                    "median_seconds": performance[count]["median seconds"],
                    "median_forward_evaluations": performance[count][
                        "median model forward evaluations"
                    ],
                    "median_peak_gpu_memory_mb": performance[count][
                        "median peak GPU memory (MB)"
                    ],
                    "SR_E": scores[count]["SR_E"],
                    "SR_div": scores[count]["SR_div"],
                    "SR_vort": scores[count]["SR_vort"],
                    "L_eff_km": scores[count]["L_eff (km)"],
                    "W1_u": scores[count]["W1 u (m/s)"],
                    "W1_v": scores[count]["W1 v (m/s)"],
                    "boundary_jump_ratio": coherence[count]["model"][
                        "boundary jump ratio"
                    ],
                    "ERA5_boundary_jump_ratio": coherence[count]["ERA5"][
                        "boundary jump ratio"
                    ],
                    "coherence_length_km": coherence[count]["model"][
                        "coherence length 0.5 (km)"
                    ],
                    "ERA5_coherence_length_km": coherence[count]["ERA5"][
                        "coherence length 0.5 (km)"
                    ],
                }
            )


def _write_report(
    configs: dict[int, dict],
    floor: dict,
    scores: dict[int, dict],
    performance: dict[int, dict],
    coherence: dict[int, dict],
    output: Path,
) -> None:
    lines = [
        "# InfiniteDiffusion fixed-domain tile-scaling benchmark",
        "",
        "Every row generates the same 64x64 location and timestamp conditions with 50% "
        "window overlap. The 4/16/64 labels count non-overlapping core regions. Because "
        "overlap halos are required, the actual final denoiser-window counts are 9/25/81.",
        "",
    ]
    checkpoints = {config["checkpoint"] for config in configs.values()}
    if len(checkpoints) == 1:
        lines += [
            "**Checkpoint control:** all rows use the same checkpoint, which was trained "
            "on 64x64 crops. The 32x32 and 16x16 rows are inference-only distribution-shift "
            "pilots and must be repeated with crop-matched models before attributing changes "
            "solely to tile size.",
            "",
        ]
    lines += [
        "## Geometry and inference cost",
        "",
        "| Core tiles | Model window | Stride | Final windows with halos | Median time (s) | "
        "Median model forwards | Peak GPU memory (MB) | Relative speed vs 64 cores |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    slow = performance[64]["median seconds"]
    for count in sorted(scores):
        profile = PROFILES[count]
        perf = performance[count]
        lines.append(
            f"| {count} | {profile.window}x{profile.window} | {profile.stride} | "
            f"{perf['final overlapping windows']} | {perf['median seconds']:.3f} | "
            f"{perf['median model forward evaluations']:.0f} | "
            f"{_fmt(perf['median peak GPU memory (MB)'])} | "
            f"{slow / perf['median seconds']:.2f}x |"
        )
    lines += [
        "",
        "## ERA5 spatial quality",
        "",
        "| Metric | ERA5 self-split floor | 4 cores | 16 cores | 64 cores |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in QUALITY_METRICS:
        lines.append(
            f"| {metric} | {_fmt(floor.get(metric, np.nan))} | "
            + " | ".join(_fmt(scores[count].get(metric, np.nan)) for count in sorted(scores))
            + " |"
        )
    lines += [
        "",
        "## Boundary and spatial coherence",
        "",
        "| Metric | 4 cores | 16 cores | 64 cores | Interpretation |",
        "|---|---:|---:|---:|---|",
    ]
    coherence_rows = (
        ("boundary jump ratio", "Near 1 means tile lines look like ordinary neighbors"),
        ("boundary squared-jump ratio", "Near 1; more sensitive to rare large seams"),
        ("boundary direction gap", "Near 0; positive means worse alignment at boundaries"),
        ("coherence length 0.5 (km)", "Compare with ERA5 at the same conditions"),
        ("boundary jump excess vs ERA5", "Near 0"),
    )
    for metric, note in coherence_rows:
        lines.append(
            f"| {metric} | "
            + " | ".join(_fmt(coherence[count]["model"][metric]) for count in sorted(scores))
            + f" | {note} |"
        )
    lines += [
        "",
        "### ERA5 boundary controls",
        "",
        "| Metric | 4-core lines | 16-core lines | 64-core lines |",
        "|---|---:|---:|---:|",
    ]
    for metric in (
        "boundary jump ratio",
        "boundary squared-jump ratio",
        "boundary direction gap",
        "coherence length 0.5 (km)",
    ):
        lines.append(
            f"| {metric} | "
            + " | ".join(_fmt(coherence[count]["ERA5"][metric]) for count in sorted(scores))
            + " |"
        )
    lines += [
        "",
        "## Figures",
        "",
        "![Runtime versus tile count](runtime_vs_tiles.png)",
        "",
        "![Quality versus tile count](quality_vs_tiles.png)",
        "",
        "![Coherence versus tile count](coherence_vs_tiles.png)",
        "",
        "![Quality-runtime tradeoff](quality_runtime_tradeoff.png)",
        "",
        "![Matched generated samples](matched_sample_montage.png)",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--run", action="append", nargs=2, metavar=("CORE_TILES", "PATH"), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs: dict[int, dict] = {}
    records_by_count: dict[int, list[dict]] = {}
    for count_text, path_text in args.run:
        count = int(count_text)
        if count not in PROFILES:
            parser.error(f"unsupported core tile count {count}")
        config, records = _load_records(Path(path_text))
        if int(config["core_tiles"]) != count:
            raise ValueError(f"{path_text}: config core_tiles does not equal {count}")
        configs[count] = config
        records_by_count[count] = records
    if set(records_by_count) != set(PROFILES):
        parser.error("provide exactly one --run for each of 4, 16, and 64 core tiles")
    expected_keys = _keys(records_by_count[4])
    for count, records in records_by_count.items():
        if _keys(records) != expected_keys:
            raise ValueError(f"{count}-core run has different conditions")

    reference = _reference_for_records(args.reference, records_by_count[4])
    floor_scores = _floor(reference)
    scores: dict[int, dict] = {}
    performance: dict[int, dict] = {}
    coherence: dict[int, dict] = {}
    reference_u = np.asarray(reference["u"].values)
    reference_v = np.asarray(reference["v"].values)
    spacing_km = _effective_spacing_km(
        np.asarray(reference["lat"].values),
        np.asarray(reference["lon"].values),
    )
    reference_length, reference_curve = coherence_length_km(
        reference_u, reference_v, spacing_km=spacing_km
    )

    details = {}
    for count in sorted(records_by_count):
        records = records_by_count[count]
        prediction = _pool(records)
        scores[count], details[count] = _score(prediction, reference)
        scores[count].update(
            conditional_w1_grouped(_condition_groups(records, reference))
        )
        performance[count] = _performance(records)
        model_u = np.stack([record["u"] for record in records])
        model_v = np.stack([record["v"] for record in records])
        model_boundary = boundary_coherence(model_u, model_v, PROFILES[count].stride)
        reference_boundary = boundary_coherence(
            reference_u, reference_v, PROFILES[count].stride
        )
        model_length, model_curve = coherence_length_km(
            model_u, model_v, spacing_km=spacing_km
        )
        model_boundary["coherence length 0.5 (km)"] = model_length
        reference_boundary["coherence length 0.5 (km)"] = reference_length
        model_boundary["boundary jump excess vs ERA5"] = (
            model_boundary["boundary jump ratio"]
            - reference_boundary["boundary jump ratio"]
        )
        reference_boundary["boundary jump excess vs ERA5"] = 0.0
        coherence[count] = {
            "model": model_boundary,
            "ERA5": reference_boundary,
            "model correlation curve": model_curve.tolist(),
            "ERA5 correlation curve": reference_curve.tolist(),
        }

    payload = {
        "protocol": {
            "query_size": 64,
            "effective_grid_spacing_km": spacing_km,
            "profiles": {str(key): vars(value) for key, value in PROFILES.items()},
        },
        "configs": configs,
        "ERA5 self-split floor": floor_scores,
        "quality": scores,
        "performance": performance,
        "coherence": coherence,
    }
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(scores, performance, coherence, args.output_dir)
    _write_report(
        configs, floor_scores, scores, performance, coherence, args.output_dir
    )
    _plot_runtime(performance, args.output_dir)
    _plot_quality(scores, floor_scores, args.output_dir)
    _plot_coherence(coherence, args.output_dir)
    _plot_tradeoff(scores, performance, args.output_dir)
    _plot_samples(records_by_count, args.output_dir)
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
