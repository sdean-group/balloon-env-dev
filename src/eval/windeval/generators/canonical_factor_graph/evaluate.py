"""Evaluate Canonical Factor-Graph Diffusion with a frozen wind checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
IDIFF_DIR = HERE.parent / "infinite_diffusion"
for directory in (HERE, IDIFF_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from core import CanonicalFactorGraphField, ChartConfig, SpaceTimeGrid  # noqa: E402
from spacetime import SpaceTimeSampler  # noqa: E402


def _vector_jumps(u: np.ndarray, v: np.ndarray, axis: int) -> np.ndarray:
    return np.sqrt(np.diff(u, axis=axis) ** 2 + np.diff(v, axis=axis) ** 2)


def _seam_ratio(jumps: np.ndarray, boundaries: list[int], axis: int) -> float | None:
    indices = [boundary - 1 for boundary in boundaries if 0 < boundary <= jumps.shape[axis]]
    if not indices:
        return None
    seam = np.take(jumps, indices, axis=axis)
    return float(seam.mean() / max(float(jumps.mean()), 1e-12))


def _internal_boundaries(origin: int, length: int, spacing: int) -> list[int]:
    first = ((origin // spacing) + 1) * spacing
    return [value - origin for value in range(first, origin + length, spacing)]


def _metrics(u: np.ndarray, v: np.ndarray, *, bounds, config: ChartConfig) -> dict:
    t0, _, y0, _, x0, _ = bounds
    speed = np.sqrt(u * u + v * v)
    x_jumps = _vector_jumps(u, v, axis=3)
    y_jumps = _vector_jumps(u, v, axis=2)
    t_jumps = _vector_jumps(u, v, axis=0) if len(u) > 1 else None
    return {
        "finite": bool(np.isfinite(u).all() and np.isfinite(v).all()),
        "shape": list(u.shape),
        "u_mean_mps": float(u.mean()),
        "u_std_mps": float(u.std()),
        "v_mean_mps": float(v.mean()),
        "v_std_mps": float(v.std()),
        "speed_mean_mps": float(speed.mean()),
        "speed_p95_mps": float(np.quantile(speed, 0.95)),
        "speed_max_mps": float(speed.max()),
        "chart_x_seam_jump_ratio": _seam_ratio(
            x_jumps, _internal_boundaries(x0, u.shape[3], config.core_size), axis=3
        ),
        "chart_y_seam_jump_ratio": _seam_ratio(
            y_jumps, _internal_boundaries(y0, u.shape[2], config.core_size), axis=2
        ),
        "chart_time_seam_jump_ratio": None if t_jumps is None else _seam_ratio(
            t_jumps, _internal_boundaries(t0, u.shape[0], config.core_time), axis=0
        ),
        "mean_temporal_vector_change_mps": (
            None if t_jumps is None else float(t_jumps.mean())
        ),
    }


def _plot(u: np.ndarray, v: np.ndarray, path: Path, level: int) -> None:
    speed = np.sqrt(u[:, level] ** 2 + v[:, level] ** 2)
    count = min(4, speed.shape[0])
    panel = max(192, 4 * max(speed.shape[-2:]))
    margin, gap, label_h = 12, 10, 28
    canvas = Image.new(
        "RGB", (2 * margin + count * panel + (count - 1) * gap,
                panel + label_h + 2 * margin), "white"
    )
    draw = ImageDraw.Draw(canvas)
    vmax = float(np.quantile(speed[:count], 0.99))
    for frame in range(count):
        normalized = np.clip(speed[frame] / max(vmax, 1e-12), 0.0, 1.0)
        rgb = np.stack(
            [np.clip(1.7 * normalized - 0.45, 0, 1),
             np.clip(np.sin(np.pi * normalized), 0, 1) ** 0.75,
             np.clip(1.15 - 1.4 * normalized, 0, 1)],
            axis=-1,
        )
        image = Image.fromarray((255 * rgb).astype(np.uint8), mode="RGB")
        image = image.resize((panel, panel), Image.Resampling.BILINEAR)
        left = margin + frame * (panel + gap)
        canvas.paste(image, (left, margin + label_h))
        draw.text((left, margin), f"frame {frame}, level index {level}", fill="black")
    canvas.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical Factor-Graph Diffusion evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/cfgd")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--core-size", type=int, default=64)
    parser.add_argument("--halo-size", type=int, default=32)
    parser.add_argument("--core-time", type=int, default=2)
    parser.add_argument("--halo-time", type=int, default=1)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--time-stride", type=int, default=2)
    parser.add_argument("--window-batch-size", type=int, default=1)
    parser.add_argument("--max-cached-charts", type=int, default=64)
    parser.add_argument("--query-size", type=int, default=64)
    parser.add_argument("--query-frames", type=int, default=2)
    parser.add_argument("--query-t0", type=int, default=1)
    parser.add_argument("--query-y0", type=int, default=32)
    parser.add_argument("--query-x0", type=int, default=32)
    parser.add_argument("--lat-origin", type=float, default=25.0)
    parser.add_argument("--lon-origin", type=float, default=225.0)
    parser.add_argument("--time-origin", default="2023-01-15T00")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sampler = SpaceTimeSampler(
        args.checkpoint, num_steps=args.num_steps, device=args.device, use_ema=True
    )
    config = ChartConfig(
        core_time=args.core_time,
        core_size=args.core_size,
        halo_time=args.halo_time,
        halo_size=args.halo_size,
        window_size=args.window,
        window_stride=args.stride,
        time_stride=args.time_stride,
        window_batch_size=args.window_batch_size,
    )
    grid = SpaceTimeGrid(
        lat_origin=args.lat_origin, lon_origin=args.lon_origin, time_origin=args.time_origin
    )
    field = CanonicalFactorGraphField(
        sampler, config=config, grid=grid, seed=args.seed,
        max_cached_charts=args.max_cached_charts,
    )
    bounds = (
        args.query_t0, args.query_t0 + args.query_frames,
        args.query_y0, args.query_y0 + args.query_size,
        args.query_x0, args.query_x0 + args.query_size,
    )
    required_charts = field.chart_keys_for_query(*bounds)

    started = time.perf_counter()
    u, v = field.field_uv(*bounds)
    cold_seconds = time.perf_counter() - started
    cold_charts = field.charts_generated
    cold_windows = field.model_window_evaluations
    cold_batch_calls = field.model_batch_calls

    started = time.perf_counter()
    u_repeat, v_repeat = field.field_uv(*bounds)
    warm_seconds = time.perf_counter() - started
    repeat_max_abs = float(max(np.max(np.abs(u - u_repeat)), np.max(np.abs(v - v_repeat))))

    report = _metrics(u, v, bounds=bounds, config=config)
    report.update({
        "architecture": "canonical_factor_graph_diffusion",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": sampler.step,
        "edm_steps": args.num_steps,
        "seed": args.seed,
        "chart_config": config.__dict__,
        "query_bounds": list(bounds),
        "required_chart_count": len(required_charts),
        "required_chart_keys": [list(key) for key in required_charts],
        "cold_seconds": cold_seconds,
        "cold_charts_generated": cold_charts,
        "cold_model_window_evaluations": cold_windows,
        "cold_model_batch_calls": cold_batch_calls,
        "warm_seconds": warm_seconds,
        "warm_extra_charts_generated": field.charts_generated - cold_charts,
        "warm_extra_model_window_evaluations": field.model_window_evaluations - cold_windows,
        "warm_extra_model_batch_calls": field.model_batch_calls - cold_batch_calls,
        "repeat_max_abs_mps": repeat_max_abs,
        "exact_cached_repeat": repeat_max_abs == 0.0,
    })
    np.savez_compressed(output / "wind.npz", u=u, v=v, levels=sampler.stats.levels)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    _plot(u, v, output / "wind.png", level=sampler.n_levels // 2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
