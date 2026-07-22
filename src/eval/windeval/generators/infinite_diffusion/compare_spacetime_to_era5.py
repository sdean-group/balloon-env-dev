"""Compare paired InfiniteDiffusion runs with the coordinate-matched ERA5 block."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def _indices_exact(source: np.ndarray, target: np.ndarray, *, circular: bool = False) -> np.ndarray:
    source = np.asarray(source)
    target = np.asarray(target)
    if np.issubdtype(source.dtype, np.datetime64):
        distance = np.abs(source[:, None] - target[None, :]) / np.timedelta64(1, "s")
        tolerance = 1.0
    else:
        distance = np.abs(source.astype(float)[:, None] - target.astype(float)[None, :])
        if circular:
            distance = np.minimum(distance, 360.0 - np.mod(distance, 360.0))
        tolerance = 1e-4
    indices = np.argmin(distance, axis=0)
    misses = distance[indices, np.arange(len(target))] > tolerance
    if np.any(misses):
        raise ValueError(f"reference coordinates do not contain targets: {target[misses][:5]}")
    return indices.astype(np.int64)


def _load_reference(path: Path, levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ds = xr.open_zarr(path, consolidated=False)
    target_time = np.datetime64("2023-01-15T02") + np.arange(4) * np.timedelta64(1, "h")
    target_lat = 33.0 + 0.25 * np.arange(64)
    target_lon = 233.0 + 0.25 * np.arange(64)
    ti = _indices_exact(ds["time"].values, target_time)
    li = _indices_exact(ds["level"].values, levels)
    yi = _indices_exact(ds["lat"].values, target_lat)
    xi = _indices_exact(ds["lon"].values, target_lon, circular=True)
    selected = ds[["u", "v"]].isel(
        time=xr.DataArray(ti, dims="time"),
        level=xr.DataArray(li, dims="level"),
        y=xr.DataArray(yi, dims="y"),
        x=xr.DataArray(xi, dims="x"),
    ).load()
    return selected["u"].values.astype(np.float32), selected["v"].values.astype(np.float32)


def _vector_jumps(u: np.ndarray, v: np.ndarray, axis: int) -> np.ndarray:
    return np.sqrt(np.diff(u, axis=axis) ** 2 + np.diff(v, axis=axis) ** 2)


def _seam_ratio(u: np.ndarray, v: np.ndarray, axis: int, index: int) -> float:
    jumps = _vector_jumps(u, v, axis)
    seam = np.take(jumps, index, axis=axis)
    return float(seam.mean() / max(float(jumps.mean()), 1e-12))


def _temporal_correlations(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    values = []
    for t in range(u.shape[0] - 1):
        for level in range(u.shape[1]):
            left = np.concatenate([u[t, level].ravel(), v[t, level].ravel()])
            right = np.concatenate([u[t + 1, level].ravel(), v[t + 1, level].ravel()])
            values.append(np.corrcoef(left, right)[0, 1])
    return np.asarray(values)


def _spectral_summary(u: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    fy = np.fft.fftfreq(u.shape[-2])
    fx = np.fft.fftfreq(u.shape[-1])
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    high = radius >= 0.25
    fractions, centroids = [], []
    for t in range(u.shape[0]):
        for level in range(u.shape[1]):
            du = u[t, level] - u[t, level].mean()
            dv = v[t, level] - v[t, level].mean()
            power = np.abs(np.fft.fft2(du)) ** 2 + np.abs(np.fft.fft2(dv)) ** 2
            total = max(float(power.sum()), 1e-12)
            fractions.append(float(power[high].sum()) / total)
            centroids.append(float((radius * power).sum()) / total)
    return float(np.mean(fractions)), float(np.mean(centroids))


def _marginal_w1(u: np.ndarray, v: np.ndarray, ru: np.ndarray, rv: np.ndarray) -> float:
    values = []
    q = np.linspace(0.0, 1.0, 1001)
    for level in range(u.shape[1]):
        for generated, reference in ((u[:, level], ru[:, level]), (v[:, level], rv[:, level])):
            values.append(float(np.mean(np.abs(
                np.quantile(generated, q) - np.quantile(reference, q)
            ))))
    return float(np.mean(values))


def _field_metrics(u: np.ndarray, v: np.ndarray) -> dict:
    speed = np.sqrt(u * u + v * v)
    temporal = _vector_jumps(u, v, 0)
    high_frequency, spectral_centroid = _spectral_summary(u, v)
    du_dx, du_dy = np.gradient(u, axis=3), np.gradient(u, axis=2)
    dv_dx, dv_dy = np.gradient(v, axis=3), np.gradient(v, axis=2)
    return {
        "u_mean_mps": float(u.mean()),
        "u_std_mps": float(u.std()),
        "v_mean_mps": float(v.mean()),
        "v_std_mps": float(v.std()),
        "speed_mean_mps": float(speed.mean()),
        "speed_p95_mps": float(np.quantile(speed, 0.95)),
        "speed_max_mps": float(speed.max()),
        "x_boundary_ratio": _seam_ratio(u, v, axis=3, index=31),
        "y_boundary_ratio": _seam_ratio(u, v, axis=2, index=31),
        "time_boundary_ratio": _seam_ratio(u, v, axis=0, index=1),
        "mean_temporal_vector_change_mps": float(temporal.mean()),
        "mean_adjacent_frame_correlation": float(_temporal_correlations(u, v).mean()),
        "high_frequency_power_fraction": high_frequency,
        "spectral_centroid_cycles_per_pixel": spectral_centroid,
        "grid_divergence_std": float(np.std(du_dx + dv_dy)),
        "grid_vorticity_std": float(np.std(dv_dx - du_dy)),
    }


def _comparison(u: np.ndarray, v: np.ndarray, ru: np.ndarray, rv: np.ndarray) -> dict:
    generated = np.stack([u, v])
    reference = np.stack([ru, rv])
    error = generated - reference
    return {
        "component_rmse_mps": float(np.sqrt(np.mean(error ** 2))),
        "component_mae_mps": float(np.mean(np.abs(error))),
        "vector_correlation": float(np.corrcoef(generated.ravel(), reference.ravel())[0, 1]),
        "mean_per_level_marginal_w1_mps": _marginal_w1(u, v, ru, rv),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--era5", required=True)
    parser.add_argument("--t1", required=True)
    parser.add_argument("--t2", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    t1 = np.load(args.t1)
    t2 = np.load(args.t2)
    levels = np.asarray(t1["levels"])
    if not np.array_equal(levels, t2["levels"]):
        raise ValueError("T=1 and T=2 level coordinates differ")
    ru, rv = _load_reference(Path(args.era5), levels)
    u1, v1 = t1["u"], t1["v"]
    u2, v2 = t2["u"], t2["v"]
    expected = (4, len(levels), 64, 64)
    if any(array.shape != expected for array in (u1, v1, u2, v2, ru, rv)):
        raise ValueError(f"expected every field to have shape {expected}")

    report = {
        "reference": {
            "path": str(Path(args.era5).resolve()),
            "time": ["2023-01-15T02", "2023-01-15T05"],
            "latitude": [33.0, 48.75],
            "longitude_east": [233.0, 248.75],
            "levels": levels.tolist(),
        },
        "era5": _field_metrics(ru, rv),
        "t1": {**_field_metrics(u1, v1), **_comparison(u1, v1, ru, rv)},
        "t2": {**_field_metrics(u2, v2), **_comparison(u2, v2, ru, rv)},
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "era5_reference.npz", u=ru, v=rv, levels=levels)
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
