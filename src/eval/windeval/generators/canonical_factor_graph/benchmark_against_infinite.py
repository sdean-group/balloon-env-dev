"""Multi-seed, matched-compute comparison of CFGD and Infinite Diffusion."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
IDIFF_DIR = HERE.parent / "infinite_diffusion"
for directory in (HERE, IDIFF_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from core import CanonicalFactorGraphField, ChartConfig, SpaceTimeGrid  # noqa: E402
from spacetime import SpaceTimeSampler  # noqa: E402
from spacetime_infinite import (  # noqa: E402
    InfiniteSpaceTimeDiffusion,
    SpaceTimeGrid as InfiniteGrid,
)


def _boundaries(start: int, length: int, spacing: int) -> list[int]:
    first = ((start // spacing) + 1) * spacing
    return [point - start for point in range(first, start + length, spacing)]


def _field_metrics(
    u: np.ndarray,
    v: np.ndarray,
    *,
    bounds: tuple[int, int, int, int, int, int],
    time_spacing: int,
    spatial_spacing: int,
) -> dict:
    t0, _, y0, _, x0, _ = bounds
    speed = np.sqrt(u * u + v * v)
    temporal = np.sqrt(np.diff(u, axis=0) ** 2 + np.diff(v, axis=0) ** 2)
    spatial_x = np.sqrt(np.diff(u, axis=3) ** 2 + np.diff(v, axis=3) ** 2)
    spatial_y = np.sqrt(np.diff(u, axis=2) ** 2 + np.diff(v, axis=2) ** 2)
    transition_means = temporal.mean(axis=(1, 2, 3)) if len(u) > 1 else np.empty(0)
    temporal_boundary_positions = _boundaries(t0, len(u), time_spacing)
    seam_indices = [position - 1 for position in temporal_boundary_positions]
    nonseam_indices = [i for i in range(len(transition_means)) if i not in seam_indices]
    seam_mean = float(transition_means[seam_indices].mean()) if seam_indices else None
    nonseam_mean = float(transition_means[nonseam_indices].mean()) if nonseam_indices else None
    temporal_mean = float(temporal.mean()) if temporal.size else None

    def seam_ratio(jumps: np.ndarray, start: int, spacing: int, axis: int) -> float | None:
        positions = _boundaries(start, jumps.shape[axis] + 1, spacing)
        indices = [position - 1 for position in positions]
        if not indices:
            return None
        return float(np.take(jumps, indices, axis=axis).mean() / max(float(jumps.mean()), 1e-12))

    return {
        "finite": bool(np.isfinite(u).all() and np.isfinite(v).all()),
        "speed_mean_mps": float(speed.mean()),
        "speed_p95_mps": float(np.quantile(speed, 0.95)),
        "speed_max_mps": float(speed.max()),
        "temporal_change_mean_mps": temporal_mean,
        "temporal_transition_means_mps": transition_means.tolist(),
        "temporal_seam_mean_mps": seam_mean,
        "temporal_nonseam_mean_mps": nonseam_mean,
        "temporal_seam_ratio": (
            None if seam_mean is None or temporal_mean is None
            else seam_mean / max(temporal_mean, 1e-12)
        ),
        "temporal_seam_to_nonseam": (
            None if seam_mean is None or nonseam_mean is None
            else seam_mean / max(nonseam_mean, 1e-12)
        ),
        "spatial_x_change_mean_mps": float(spatial_x.mean()),
        "spatial_y_change_mean_mps": float(spatial_y.mean()),
        "spatial_x_seam_ratio": seam_ratio(spatial_x, x0, spatial_spacing, 3),
        "spatial_y_seam_ratio": seam_ratio(spatial_y, y0, spatial_spacing, 2),
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


def _release_device_cache(device: str) -> None:
    gc.collect()
    if device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched-compute, multi-seed CFGD versus Infinite Diffusion benchmark"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/cfgd_vs_infd_mps")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--num-steps", type=int, default=18)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29, 43, 61])
    parser.add_argument("--query-size", type=int, default=16)
    parser.add_argument("--query-frames", type=int, default=4)
    parser.add_argument("--window-batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sampler = SpaceTimeSampler(
        args.checkpoint, num_steps=args.num_steps, device=args.device, use_ema=True
    )

    # These two integer queries cover the same physical coordinates. Their alignment was
    # selected because both architectures require exactly 12 windows/charts-factors.
    cfgd_bounds = (1, 1 + args.query_frames, 24, 24 + args.query_size,
                   24, 24 + args.query_size)
    infd_bounds = (2, 2 + args.query_frames, 32, 32 + args.query_size,
                   32, 32 + args.query_size)
    cfgd_grid = SpaceTimeGrid(
        lat_origin=29.0, lon_origin=229.0, time_origin="2023-01-15T00"
    )
    infd_grid = InfiniteGrid(
        lat_origin=27.0, lon_origin=227.0, time_origin="2023-01-14T23"
    )
    chart_config = ChartConfig(
        core_time=2, core_size=64, halo_time=1, halo_size=16,
        window_size=64, window_stride=32, time_stride=2,
        window_batch_size=args.window_batch_size,
    )

    records: list[dict] = []
    for seed_index, seed in enumerate(args.seeds):
        methods = ["cfgd", "infinite"] if seed_index % 2 == 0 else ["infinite", "cfgd"]
        seed_records: dict[str, dict] = {}
        for method in methods:
            if method == "cfgd":
                field = CanonicalFactorGraphField(
                    sampler, config=chart_config, grid=cfgd_grid, seed=seed,
                    max_cached_charts=16,
                )
                bounds = cfgd_bounds
                spacing = chart_config.core_time
            else:
                field = InfiniteSpaceTimeDiffusion(
                    sampler, grid=infd_grid, window=64, stride=32,
                    time_stride=2, seed=seed, outer_depth=1,
                )
                bounds = infd_bounds
                spacing = 2

            started = time.perf_counter()
            u, v = field.field_uv(*bounds)
            seconds = time.perf_counter() - started
            u_repeat, v_repeat = field.field_uv(*bounds)
            repeat_error = float(max(np.max(np.abs(u - u_repeat)), np.max(np.abs(v - v_repeat))))
            metrics = _field_metrics(
                u, v, bounds=bounds, time_spacing=spacing, spatial_spacing=64 if method == "cfgd" else 32,
            )
            evaluations = (
                field.model_window_evaluations if method == "cfgd"
                else field.model_forward_evaluations
            )
            method_output = output / f"seed_{seed}" / method
            method_output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(method_output / "wind.npz", u=u, v=v, levels=sampler.stats.levels)
            seed_records[method] = {
                **metrics,
                "seconds": seconds,
                "model_forward_evaluations": evaluations,
                "repeat_max_abs_mps": repeat_error,
                "exact_cached_repeat": repeat_error == 0.0,
            }
            (method_output / "metrics.json").write_text(
                json.dumps(seed_records[method], indent=2) + "\n"
            )
            del field, u, v, u_repeat, v_repeat
            _release_device_cache(args.device)

        record = {"seed": seed, **seed_records}
        records.append(record)
        print(json.dumps(record, indent=2), flush=True)

    metric_names = [
        "temporal_seam_ratio", "temporal_seam_to_nonseam",
        "temporal_change_mean_mps", "speed_mean_mps", "speed_p95_mps",
        "speed_max_mps", "spatial_x_change_mean_mps", "spatial_y_change_mean_mps",
        "seconds", "model_forward_evaluations",
    ]
    aggregate = {"n_seeds": len(records), "metrics": {}}
    for name in metric_names:
        cfgd_values = [float(record["cfgd"][name]) for record in records]
        infd_values = [float(record["infinite"][name]) for record in records]
        aggregate["metrics"][name] = {
            "cfgd": _summary(cfgd_values),
            "infinite": _summary(infd_values),
            "paired_delta_cfgd_minus_infinite": _summary(
                [a - b for a, b in zip(cfgd_values, infd_values)]
            ),
        }

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "edm_steps": args.num_steps,
        "seeds": args.seeds,
        "physical_start": {
            "time": "2023-01-15T01", "latitude": 35.0, "longitude": 235.0,
        },
        "query_shape": [args.query_frames, sampler.n_levels, args.query_size, args.query_size],
        "method_order_alternates_by_seed": True,
        "matched_noise_realization": False,
        "comparison_note": (
            "Methods cover identical physical coordinates and use equal forward-evaluation "
            "budgets, but their differently aligned lattices do not expose identical noise tensors."
        ),
        "records": records,
        "aggregate": aggregate,
    }
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
