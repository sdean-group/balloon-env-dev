"""Ensemble visual and structural diagnostics for tile-scaling runs."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import xarray as xr

try:
    from .eye_test import _axis_indices, _load_records
    from .protocol import PROFILES, spatial_correlation_curve
except ImportError:  # standalone rendering/test execution
    from eye_test import _axis_indices, _load_records
    from protocol import PROFILES, spatial_correlation_curve


METHOD_LABELS = {
    4: "4 cores",
    16: "16 cores",
    64: "64 cores",
}
LAGS = (1, 2, 4, 8, 16)


def _condition_key(record: dict) -> tuple[int, int, int]:
    return record["month"], record["day"], record["hour"]


def _load_reference(
    path: Path,
    records: list[dict],
) -> tuple[dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]], np.ndarray]:
    source = xr.open_zarr(path, consolidated=False)
    try:
        exemplar = records[0]
        yi = _axis_indices(
            np.asarray(source["lat"].values), exemplar["lat"], "latitude"
        )
        xi = _axis_indices(
            np.asarray(source["lon"].values), exemplar["lon"], "longitude"
        )
        times = np.asarray(source["time"].values).astype("datetime64[h]")
        time_lookup = {value: index for index, value in enumerate(times)}
        keys = sorted({_condition_key(record) for record in records})
        result = {}
        for month, day, hour in keys:
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
        return result, np.asarray(exemplar["levels"])
    finally:
        source.close()


def _stack_reference(
    reference: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(reference)
    return (
        np.stack([reference[key][0] for key in keys]),
        np.stack([reference[key][1] for key in keys]),
    )


def _stack_records(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([record["u"] for record in records]),
        np.stack([record["v"] for record in records]),
    )


def _reference_normalization(
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axes = (0, 2, 3)
    u_mean = u.mean(axis=axes, keepdims=True)
    v_mean = v.mean(axis=axes, keepdims=True)
    u_std = np.maximum(u.std(axis=axes, keepdims=True), 1e-6)
    v_std = np.maximum(v.std(axis=axes, keepdims=True), 1e-6)
    return u_mean, u_std, v_mean, v_std


def _average_pool(values: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return values
    n, levels, height, width = values.shape
    if height % factor or width % factor:
        raise ValueError("pooling factor must divide both spatial dimensions")
    return values.reshape(
        n, levels, height // factor, factor, width // factor, factor
    ).mean(axis=(3, 5))


def _sample_patches(
    u: np.ndarray,
    v: np.ndarray,
    *,
    patch: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n, levels, height, width = u.shape
    if patch > min(height, width):
        raise ValueError("patch exceeds field dimensions")
    samples = np.empty((count, 2 * patch * patch), dtype=np.float32)
    for index in range(count):
        sample = int(rng.integers(n))
        level = int(rng.integers(levels))
        y = int(rng.integers(height - patch + 1))
        x = int(rng.integers(width - patch + 1))
        samples[index] = np.concatenate(
            (
                u[sample, level, y:y + patch, x:x + patch].ravel(),
                v[sample, level, y:y + patch, x:x + patch].ravel(),
            )
        )
    return samples


def multiscale_patch_swd(
    prediction: tuple[np.ndarray, np.ndarray],
    reference: tuple[np.ndarray, np.ndarray],
    *,
    factors: tuple[int, ...] = (1, 2, 4),
    patch: int = 5,
    patches: int = 2048,
    projections: int = 64,
    seed: int = 17,
) -> tuple[float, dict[int, float]]:
    """Sliced-Wasserstein distance between standardized local vector patches."""
    pred_u, pred_v = prediction
    ref_u, ref_v = reference
    u_mean, u_std, v_mean, v_std = _reference_normalization(ref_u, ref_v)
    pred_u = (pred_u - u_mean) / u_std
    pred_v = (pred_v - v_mean) / v_std
    ref_u = (ref_u - u_mean) / u_std
    ref_v = (ref_v - v_mean) / v_std

    detail = {}
    for factor in factors:
        rng_ref = np.random.default_rng(seed + factor)
        rng_pred = np.random.default_rng(seed + 100 + factor)
        ref_patches = _sample_patches(
            _average_pool(ref_u, factor),
            _average_pool(ref_v, factor),
            patch=patch,
            count=patches,
            rng=rng_ref,
        )
        pred_patches = _sample_patches(
            _average_pool(pred_u, factor),
            _average_pool(pred_v, factor),
            patch=patch,
            count=patches,
            rng=rng_pred,
        )
        rng_projection = np.random.default_rng(seed + 1000 + factor)
        directions = rng_projection.normal(
            size=(ref_patches.shape[1], projections)
        ).astype(np.float32)
        directions /= np.maximum(
            np.linalg.norm(directions, axis=0, keepdims=True), 1e-12
        )
        ref_projected = np.sort(ref_patches @ directions, axis=0)
        pred_projected = np.sort(pred_patches @ directions, axis=0)
        detail[factor] = float(np.mean(np.abs(ref_projected - pred_projected)))
    return float(np.mean(list(detail.values()))), detail


def vector_structure_function(
    u: np.ndarray,
    v: np.ndarray,
    *,
    lags: tuple[int, ...] = LAGS,
) -> np.ndarray:
    """Mean squared vector increment as a function of spatial lag."""
    if any(lag < 1 or lag >= min(u.shape[-2:]) for lag in lags):
        raise ValueError("every structure-function lag must lie inside the field")
    result = []
    for lag in lags:
        dx = (u[..., lag:] - u[..., :-lag]) ** 2
        dx += (v[..., lag:] - v[..., :-lag]) ** 2
        dy = (u[..., lag:, :] - u[..., :-lag, :]) ** 2
        dy += (v[..., lag:, :] - v[..., :-lag, :]) ** 2
        result.append(0.5 * (float(dx.mean()) + float(dy.mean())))
    return np.asarray(result)


def structure_log_rmse(
    prediction: tuple[np.ndarray, np.ndarray],
    reference: tuple[np.ndarray, np.ndarray],
) -> tuple[float, np.ndarray, np.ndarray]:
    pred_curve = vector_structure_function(*prediction)
    ref_curve = vector_structure_function(*reference)
    error = np.sqrt(
        np.mean((np.log(np.maximum(pred_curve, 1e-12))
                 - np.log(np.maximum(ref_curve, 1e-12))) ** 2)
    )
    return float(error), pred_curve, ref_curve


def correlation_curve_rmse(
    prediction: tuple[np.ndarray, np.ndarray],
    reference: tuple[np.ndarray, np.ndarray],
) -> tuple[float, np.ndarray, np.ndarray]:
    pred_curve = spatial_correlation_curve(*prediction)
    ref_curve = spatial_correlation_curve(*reference)
    return (
        float(np.sqrt(np.mean((pred_curve - ref_curve) ** 2))),
        pred_curve,
        ref_curve,
    )


def _sample_vector_jumps(
    u: np.ndarray,
    v: np.ndarray,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    x = np.hypot(np.diff(u, axis=3), np.diff(v, axis=3)).ravel()
    y = np.hypot(np.diff(u, axis=2), np.diff(v, axis=2)).ravel()
    values = np.concatenate((x, y))
    if len(values) <= count:
        return values
    rng = np.random.default_rng(seed)
    return values[rng.choice(len(values), size=count, replace=False)]


def gradient_w1(
    prediction: tuple[np.ndarray, np.ndarray],
    reference: tuple[np.ndarray, np.ndarray],
    *,
    samples: int = 250_000,
) -> float:
    pred = np.sort(_sample_vector_jumps(*prediction, count=samples, seed=31))
    ref = np.sort(_sample_vector_jumps(*reference, count=samples, seed=37))
    quantiles = np.linspace(0.0, 1.0, min(len(pred), len(ref)))
    pred_q = np.quantile(pred, quantiles)
    ref_q = np.quantile(ref, quantiles)
    return float(np.mean(np.abs(pred_q - ref_q)))


def ensemble_spread_ratio(
    records: list[dict],
    reference: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, float]:
    """Generated two-seed spread divided by ERA5 within-month day-to-day spread."""
    grouped: dict[tuple[int, int, int], list[dict]] = {}
    for record in records:
        grouped.setdefault(_condition_key(record), []).append(record)
    generated = []
    for samples in grouped.values():
        if len(samples) < 2:
            continue
        for first, second in combinations(samples, 2):
            generated.append(
                np.sqrt(
                    np.mean(
                        (first["u"] - second["u"]) ** 2
                        + (first["v"] - second["v"]) ** 2
                    )
                )
            )

    reference_grouped: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    for (month, _, hour), values in reference.items():
        reference_grouped.setdefault((month, hour), []).append(values)
    observed = []
    for samples in reference_grouped.values():
        for first, second in combinations(samples, 2):
            observed.append(
                np.sqrt(
                    np.mean(
                        (first[0] - second[0]) ** 2
                        + (first[1] - second[1]) ** 2
                    )
                )
            )
    generated_mean = float(np.mean(generated))
    observed_mean = float(np.mean(observed))
    return generated_mean / observed_mean, generated_mean, observed_mean


def _plot_examples(
    records: dict[int, dict[tuple[int, int, int, int], dict]],
    reference: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
    *,
    level_index: int,
    output: Path,
) -> None:
    keys = (
        (1, 8, 0, 0),
        (1, 13, 12, 1),
        (4, 10, 0, 0),
        (7, 12, 12, 1),
        (10, 9, 0, 1),
        (10, 14, 12, 0),
    )
    methods = [None, *sorted(records)]
    fields = []
    for key in keys:
        month, day, hour, _ = key
        fields.append(
            tuple(values[level_index] for values in reference[(month, day, hour)])
        )
        for count in sorted(records):
            record = records[count][key]
            fields.append((record["u"][level_index], record["v"][level_index]))
    limit = float(np.quantile(np.concatenate(
        [np.hypot(u, v).ravel() for u, v in fields]
    ), 0.99))

    fig, axes = plt.subplots(
        len(methods), len(keys), figsize=(15.2, 10.2), constrained_layout=True
    )
    image = None
    for column, key in enumerate(keys):
        month, day, hour, seed = key
        column_fields = [
            tuple(values[level_index] for values in reference[(month, day, hour)]),
            *[
                (
                    records[count][key]["u"][level_index],
                    records[count][key]["v"][level_index],
                )
                for count in sorted(records)
            ],
        ]
        for row, ((u, v), method) in enumerate(zip(column_fields, methods)):
            image = axes[row, column].imshow(
                np.hypot(u, v),
                origin="lower",
                cmap="viridis",
                norm=Normalize(0.0, limit),
            )
            step = 8
            y, x = np.mgrid[0:u.shape[0]:step, 0:u.shape[1]:step]
            axes[row, column].quiver(
                x,
                y,
                u[::step, ::step],
                v[::step, ::step],
                color="white",
                angles="xy",
                scale_units="xy",
                scale=max(limit / 1.8, 1.0),
                width=0.004,
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(
                    f"{month:02d}-{day:02d} {hour:02d}Z", fontsize=9
                )
            if column == 0:
                axes[row, column].set_ylabel(
                    "ERA5" if method is None else METHOD_LABELS[method],
                    fontsize=10,
                )
    fig.suptitle(
        "Six matched conditions; generated columns alternate seeds 0 and 1",
        fontsize=12,
    )
    fig.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        shrink=0.55,
        pad=0.025,
        label="Wind speed (m/s), common scale",
    )
    fig.savefig(output / "ensemble_examples.png", dpi=220)
    plt.close(fig)


def _plot_summary_maps(
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    level_index: int,
    output: Path,
) -> None:
    summaries = {}
    for name, (u, v) in arrays.items():
        speed = np.hypot(u[:, level_index], v[:, level_index])
        summaries[name] = (
            speed.mean(axis=0),
            speed.std(axis=0),
            np.quantile(speed, 0.95, axis=0),
        )
    limits = [
        max(float(summary[column].max()) for summary in summaries.values())
        for column in range(3)
    ]
    labels = ("Mean speed", "Speed standard deviation", "95th-percentile speed")
    fig, axes = plt.subplots(
        len(summaries), 3, figsize=(10.2, 11.0), constrained_layout=True
    )
    for row, (name, summary) in enumerate(summaries.items()):
        for column, values in enumerate(summary):
            image = axes[row, column].imshow(
                values,
                origin="lower",
                cmap="viridis",
                norm=Normalize(0.0, limits[column]),
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(labels[column], fontsize=10)
            if column == 0:
                axes[row, column].set_ylabel(name, fontsize=10)
            fig.colorbar(image, ax=axes[row, column], fraction=0.046, label="m/s")
    fig.suptitle("Ensemble spatial summaries over all matched samples", fontsize=12)
    fig.savefig(output / "ensemble_summary_maps.png", dpi=220)
    plt.close(fig)


def _plot_scale_curves(
    diagnostics: dict[str, dict],
    *,
    reference_structure: np.ndarray,
    reference_correlation: np.ndarray,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    axes[0].plot(LAGS, reference_structure, marker="o", label="ERA5", linewidth=2)
    axes[1].plot(
        np.arange(len(reference_correlation)),
        reference_correlation,
        label="ERA5",
        linewidth=2,
    )
    for name, values in diagnostics.items():
        axes[0].plot(LAGS, values["structure curve"], marker="o", label=name)
        axes[1].plot(values["correlation curve"], label=name)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(LAGS, [str(lag) for lag in LAGS])
    axes[0].set_xlabel("Spatial lag (grid cells)")
    axes[0].set_ylabel("Mean squared vector increment")
    axes[1].set_xlabel("Spatial lag (grid cells)")
    axes[1].set_ylabel("Normalized vector correlation")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(output / "ensemble_scale_curves.png", dpi=200)
    plt.close(fig)


def _write_report(metrics: dict[str, dict], output: Path) -> None:
    lines = [
        "# Ensemble structural diagnostics",
        "",
        "All statistics use every available held-out condition. Generated runs contain "
        "two seeds per timestamp; ERA5 contains one observed realization per timestamp. "
        "No pointwise generated-to-ERA5 image error is used.",
        "",
        "| Method | Multiscale patch SWD | Structure log-RMSE | Correlation curve RMSE | "
        "Gradient W1 (m/s) | Spread ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        lines.append(
            f"| {name} | {values['multiscale patch SWD']:.4f} | "
            f"{values['structure log-RMSE']:.4f} | "
            f"{values['correlation curve RMSE']:.4f} | "
            f"{values['gradient W1 (m/s)']:.4f} | "
            f"{values['spread ratio']:.4f} |"
        )
    lines += [
        "",
        "Lower is better for every metric except spread ratio, whose target is near 1. "
        "Patch SWD is dimensionless because u and v are standardized per level using "
        "ERA5 statistics.",
        "",
        "- **Multiscale patch SWD:** unpaired local vector-texture distance at native, "
        "2x-pooled, and 4x-pooled scales.",
        "- **Structure log-RMSE:** mismatch in the scale-dependent mean squared wind-vector "
        "increment.",
        "- **Correlation curve RMSE:** mismatch across the full two-point spatial "
        "correlation curve, rather than one threshold crossing.",
        "- **Gradient W1:** Wasserstein distance between neighboring-cell vector-jump "
        "distributions.",
        "- **Spread ratio:** two-seed generated RMS separation divided by ERA5's "
        "within-season day-to-day RMS separation.",
        "",
        "## Figures",
        "",
        "![Multiple matched examples](ensemble_examples.png)",
        "",
        "![Ensemble summary maps](ensemble_summary_maps.png)",
        "",
        "![Scale-dependent diagnostics](ensemble_scale_curves.png)",
        "",
    ]
    (output / "ensemble_report.md").write_text("\n".join(lines))


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
    parser.add_argument("--level-index", type=int, default=9)
    args = parser.parse_args(argv)

    records_by_count = {}
    record_lookup = {}
    for count_text, path_text in args.run:
        count = int(count_text)
        if count not in PROFILES:
            parser.error(f"unsupported core tile count {count}")
        _, records = _load_records(Path(path_text))
        records_by_count[count] = records
        record_lookup[count] = {
            (record["month"], record["day"], record["hour"], record["seed"]): record
            for record in records
        }
    if set(records_by_count) != set(PROFILES):
        parser.error("provide exactly one --run for 4, 16, and 64 core tiles")

    reference, levels = _load_reference(args.reference, records_by_count[4])
    ref_arrays = _stack_reference(reference)
    arrays = {
        "ERA5": ref_arrays,
        **{
            METHOD_LABELS[count]: _stack_records(records_by_count[count])
            for count in sorted(records_by_count)
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    diagnostics = {}
    ref_structure = vector_structure_function(*ref_arrays)
    ref_correlation = spatial_correlation_curve(*ref_arrays)
    for count in sorted(records_by_count):
        name = METHOD_LABELS[count]
        prediction = arrays[name]
        patch_score, patch_detail = multiscale_patch_swd(prediction, ref_arrays)
        structure_error, structure_curve, _ = structure_log_rmse(
            prediction, ref_arrays
        )
        correlation_error, correlation_curve, _ = correlation_curve_rmse(
            prediction, ref_arrays
        )
        spread_ratio, generated_spread, era5_spread = ensemble_spread_ratio(
            records_by_count[count], reference
        )
        metrics[name] = {
            "multiscale patch SWD": patch_score,
            "patch SWD by pooling factor": {
                str(key): value for key, value in patch_detail.items()
            },
            "structure log-RMSE": structure_error,
            "correlation curve RMSE": correlation_error,
            "gradient W1 (m/s)": gradient_w1(prediction, ref_arrays),
            "spread ratio": spread_ratio,
            "generated two-seed RMS spread (m/s)": generated_spread,
            "ERA5 within-season RMS spread (m/s)": era5_spread,
        }
        diagnostics[name] = {
            "structure curve": structure_curve.tolist(),
            "correlation curve": correlation_curve.tolist(),
        }

    payload = {
        "protocol": {
            "generated_samples_per_method": len(records_by_count[4]),
            "reference_timestamps": len(reference),
            "levels": levels.tolist(),
            "visual_level_index": args.level_index,
            "visual_level": int(levels[args.level_index]),
        },
        "metrics": metrics,
        "curves": {
            "lags": list(LAGS),
            "ERA5 structure curve": ref_structure.tolist(),
            "ERA5 correlation curve": ref_correlation.tolist(),
            **diagnostics,
        },
    }
    (args.output_dir / "ensemble_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    _write_report(metrics, args.output_dir)
    _plot_examples(
        record_lookup,
        reference,
        level_index=args.level_index,
        output=args.output_dir,
    )
    _plot_summary_maps(
        arrays,
        level_index=args.level_index,
        output=args.output_dir,
    )
    _plot_scale_curves(
        diagnostics,
        reference_structure=ref_structure,
        reference_correlation=ref_correlation,
        output=args.output_dir,
    )
    print(args.output_dir / "ensemble_report.md")


if __name__ == "__main__":
    main()
