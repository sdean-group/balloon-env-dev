"""Compare a multi-seed CFGD/Infinite benchmark with a local ERA5 NPZ block."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _temporal_correlations(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    values = []
    for t in range(u.shape[0] - 1):
        for level in range(u.shape[1]):
            left = np.concatenate((u[t, level].ravel(), v[t, level].ravel()))
            right = np.concatenate((u[t + 1, level].ravel(), v[t + 1, level].ravel()))
            values.append(np.corrcoef(left, right)[0, 1])
    return np.asarray(values)


def _spectral_summary(u: np.ndarray, v: np.ndarray) -> float:
    fy = np.fft.fftfreq(u.shape[-2])
    fx = np.fft.fftfreq(u.shape[-1])
    high = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) >= 0.25
    values = []
    for t in range(u.shape[0]):
        for level in range(u.shape[1]):
            du = u[t, level] - u[t, level].mean()
            dv = v[t, level] - v[t, level].mean()
            power = np.abs(np.fft.fft2(du)) ** 2 + np.abs(np.fft.fft2(dv)) ** 2
            values.append(float(power[high].sum()) / max(float(power.sum()), 1e-12))
    return float(np.mean(values))


def _field_metrics(u: np.ndarray, v: np.ndarray) -> dict:
    speed = np.sqrt(u * u + v * v)
    temporal = np.sqrt(np.diff(u, axis=0) ** 2 + np.diff(v, axis=0) ** 2)
    du_dx, du_dy = np.gradient(u, axis=3), np.gradient(u, axis=2)
    dv_dx, dv_dy = np.gradient(v, axis=3), np.gradient(v, axis=2)
    return {
        "speed_mean_mps": float(speed.mean()),
        "speed_p95_mps": float(np.quantile(speed, 0.95)),
        "temporal_change_mps": float(temporal.mean()),
        "adjacent_frame_correlation": float(_temporal_correlations(u, v).mean()),
        "high_frequency_power_fraction": _spectral_summary(u, v),
        "grid_divergence_std": float(np.std(du_dx + dv_dy)),
        "grid_vorticity_std": float(np.std(dv_dx - du_dy)),
    }


def _marginal_w1(u: np.ndarray, v: np.ndarray, ru: np.ndarray, rv: np.ndarray) -> float:
    quantiles = np.linspace(0.0, 1.0, 1001)
    values = []
    for level in range(u.shape[1]):
        for generated, reference in ((u[:, level], ru[:, level]), (v[:, level], rv[:, level])):
            values.append(float(np.mean(np.abs(
                np.quantile(generated, quantiles) - np.quantile(reference, quantiles)
            ))))
    return float(np.mean(values))


def _comparison(u: np.ndarray, v: np.ndarray, ru: np.ndarray, rv: np.ndarray) -> dict:
    generated = np.stack((u, v))
    reference = np.stack((ru, rv))
    error = generated - reference
    return {
        "component_rmse_mps": float(np.sqrt(np.mean(error * error))),
        "component_mae_mps": float(np.mean(np.abs(error))),
        "vector_correlation": float(np.corrcoef(generated.ravel(), reference.ravel())[0, 1]),
        "per_level_marginal_w1_mps": _marginal_w1(u, v, ru, rv),
    }


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    sem = std / np.sqrt(len(array))
    return {
        "mean": float(array.mean()),
        "sample_std": std,
        "ci95_low": float(array.mean() - 1.96 * sem),
        "ci95_high": float(array.mean() + 1.96 * sem),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiseed wind samples to ERA5 NPZ")
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--era5", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    benchmark = json.loads((benchmark_dir / "benchmark.json").read_text())
    era5 = np.load(args.era5)
    levels = np.asarray(era5["levels"])
    if era5["u"].shape != (4, len(levels), 64, 64):
        raise ValueError("this comparator expects the saved January 15 ERA5 reference block")

    # ERA5 is 02:00--05:00 from 33N, 233E. The generated benchmark is 01:00--04:00
    # from 35N, 235E, so their exact overlap is 02:00--04:00 and spatial offset 8:24.
    ru = np.asarray(era5["u"][:3, :, 8:24, 8:24], dtype=np.float32)
    rv = np.asarray(era5["v"][:3, :, 8:24, 8:24], dtype=np.float32)
    reference_metrics = _field_metrics(ru, rv)
    records = []
    for seed in benchmark["seeds"]:
        record = {"seed": seed}
        for method in ("cfgd", "infinite"):
            sample = np.load(benchmark_dir / f"seed_{seed}" / method / "wind.npz")
            if not np.array_equal(sample["levels"], levels):
                raise ValueError(f"seed {seed} {method}: pressure levels do not match ERA5")
            u = np.asarray(sample["u"][1:4], dtype=np.float32)
            v = np.asarray(sample["v"][1:4], dtype=np.float32)
            record[method] = {**_field_metrics(u, v), **_comparison(u, v, ru, rv)}
        records.append(record)

    metric_names = list(records[0]["cfgd"])
    aggregate = {}
    for metric in metric_names:
        cfgd = [float(record["cfgd"][metric]) for record in records]
        infinite = [float(record["infinite"][metric]) for record in records]
        aggregate[metric] = {
            "era5": reference_metrics.get(metric),
            "cfgd": _summary(cfgd),
            "infinite": _summary(infinite),
            "paired_delta_cfgd_minus_infinite": _summary(
                [a - b for a, b in zip(cfgd, infinite)]
            ),
        }

    report = {
        "scope": {
            "time": ["2023-01-15T02", "2023-01-15T04"],
            "latitude": [35.0, 38.75],
            "longitude_east": [235.0, 238.75],
            "shape": list(ru.shape),
            "warning": (
                "January 15 overlaps the checkpoint training-date range. This is a "
                "condition-matched sanity check, not a held-out generalization benchmark."
            ),
        },
        "era5": reference_metrics,
        "records": records,
        "aggregate": aggregate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"scope": report["scope"], "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
