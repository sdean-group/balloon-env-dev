"""Run and evaluate the conditional M2 checkpoint through InfiniteDiffusion."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from spacetime import SpaceTimeSampler  # noqa: E402
from spacetime_infinite import InfiniteSpaceTimeDiffusion, SpaceTimeGrid  # noqa: E402


def _vector_jumps(u: np.ndarray, v: np.ndarray, axis: int) -> np.ndarray:
    return np.sqrt(np.diff(u, axis=axis) ** 2 + np.diff(v, axis=axis) ** 2)


def _seam_ratio(jumps: np.ndarray, boundaries: list[int], axis: int) -> float | None:
    indices = [b - 1 for b in boundaries if 0 < b <= jumps.shape[axis]]
    if not indices:
        return None
    seam = np.take(jumps, indices, axis=axis)
    return float(seam.mean() / max(float(jumps.mean()), 1e-12))


def _metrics(
    u: np.ndarray,
    v: np.ndarray,
    *,
    t0: int,
    y0: int,
    x0: int,
    time_stride: int,
    stride: int,
) -> dict:
    speed = np.sqrt(u * u + v * v)
    spatial_x = _vector_jumps(u, v, axis=3)
    spatial_y = _vector_jumps(u, v, axis=2)
    temporal = _vector_jumps(u, v, axis=0) if u.shape[0] > 1 else None
    x_bounds = [x - x0 for x in range(((x0 // stride) + 1) * stride, x0 + u.shape[3], stride)]
    y_bounds = [y - y0 for y in range(((y0 // stride) + 1) * stride, y0 + u.shape[2], stride)]
    t_bounds = [t - t0 for t in range(((t0 // time_stride) + 1) * time_stride, t0 + u.shape[0], time_stride)]
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
        "x_seam_jump_ratio": _seam_ratio(spatial_x, x_bounds, axis=3),
        "y_seam_jump_ratio": _seam_ratio(spatial_y, y_bounds, axis=2),
        "time_seam_jump_ratio": None if temporal is None else _seam_ratio(temporal, t_bounds, axis=0),
        "mean_temporal_vector_change_mps": None if temporal is None else float(temporal.mean()),
    }


def _plot(u: np.ndarray, v: np.ndarray, path: Path, level: int) -> None:
    speed = np.sqrt(u[:, level] ** 2 + v[:, level] ** 2)
    count = min(4, speed.shape[0])
    vmax = float(np.quantile(speed[:count], 0.99))
    panel = max(192, 4 * max(speed.shape[-2:]))
    margin, gap, label_h = 12, 10, 28
    canvas = Image.new("RGB", (2 * margin + count * panel + (count - 1) * gap, panel + label_h + 2 * margin), "white")
    draw = ImageDraw.Draw(canvas)
    for t in range(count):
        normalized = np.clip(speed[t] / max(vmax, 1e-12), 0.0, 1.0)
        red = np.clip(1.7 * normalized - 0.45, 0.0, 1.0)
        green = np.clip(np.sin(np.pi * normalized), 0.0, 1.0) ** 0.75
        blue = np.clip(1.15 - 1.4 * normalized, 0.0, 1.0)
        rgb = (255.0 * np.stack([red, green, blue], axis=-1)).astype(np.uint8)
        image = Image.fromarray(np.flipud(rgb), mode="RGB").resize((panel, panel), Image.Resampling.BILINEAR)
        left = margin + t * (panel + gap)
        top = margin + label_h
        canvas.paste(image, (left, top))
        draw.text((left, margin), f"frame {t}, level index {level}", fill="black")

        step = max(1, speed.shape[-1] // 10)
        scale_x = panel / speed.shape[-1]
        scale_y = panel / speed.shape[-2]
        arrow = 0.35 * min(scale_x, scale_y) * step
        for iy in range(step // 2, speed.shape[-2], step):
            for ix in range(step // 2, speed.shape[-1], step):
                magnitude = max(float(speed[t, iy, ix]), 1e-6)
                dx = arrow * float(u[t, level, iy, ix]) / magnitude
                dy = -arrow * float(v[t, level, iy, ix]) / magnitude
                cx = left + (ix + 0.5) * scale_x
                cy = top + (speed.shape[-2] - iy - 0.5) * scale_y
                draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill="white", width=1)
    canvas.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/m2cond_infinite")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--outer-depth", type=int, choices=(1, 2), default=1)
    parser.add_argument("--split-step", type=int)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--time-stride", type=int, default=2)
    parser.add_argument("--query-size", type=int, default=32)
    parser.add_argument("--query-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lat-origin", type=float, default=25.0)
    parser.add_argument("--lon-origin", type=float, default=225.0)
    parser.add_argument("--time-origin", default="2023-01-15T00")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sampler = SpaceTimeSampler(
        args.checkpoint,
        num_steps=args.num_steps,
        device=args.device,
        use_ema=True,
    )
    grid = SpaceTimeGrid(
        lat_origin=args.lat_origin,
        lon_origin=args.lon_origin,
        time_origin=args.time_origin,
    )
    field = InfiniteSpaceTimeDiffusion(
        sampler,
        grid=grid,
        window=args.window,
        stride=args.stride,
        time_stride=args.time_stride,
        seed=args.seed,
        outer_depth=args.outer_depth,
        split_step=args.split_step,
    )

    t0 = args.time_stride
    y0 = args.stride
    x0 = args.stride
    bounds = (t0, t0 + args.query_frames, y0, y0 + args.query_size, x0, x0 + args.query_size)
    started = time.perf_counter()
    u, v = field.field_uv(*bounds)
    cold_seconds = time.perf_counter() - started
    cold_calls = field.model_window_calls
    cold_forward_evaluations = field.model_forward_evaluations

    started = time.perf_counter()
    u_cached, v_cached = field.field_uv(*bounds)
    warm_seconds = time.perf_counter() - started
    warm_extra_calls = field.model_window_calls - cold_calls
    warm_extra_forward_evaluations = (
        field.model_forward_evaluations - cold_forward_evaluations
    )
    repeat_max_abs = float(max(np.max(np.abs(u - u_cached)), np.max(np.abs(v - v_cached))))

    report = _metrics(
        u,
        v,
        t0=t0,
        y0=y0,
        x0=x0,
        time_stride=args.time_stride,
        stride=args.stride,
    )
    report.update(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_step": sampler.step,
            "edm_steps": args.num_steps,
            "outer_depth": field.outer_depth,
            "split_step": field.split_step if field.outer_depth == 2 else None,
            "split_sigma": (
                float(sampler.sigma_schedule(device="cpu")[field.split_step])
                if field.outer_depth == 2
                else None
            ),
            "window": args.window,
            "stride": args.stride,
            "time_window": sampler.tau,
            "time_stride": args.time_stride,
            "cold_seconds": cold_seconds,
            "cold_model_window_calls": cold_calls,
            "phase_window_calls": field.phase_window_calls,
            "cold_model_forward_evaluations": cold_forward_evaluations,
            "phase_forward_evaluations": field.phase_forward_evaluations,
            "evaluated_window_index_ranges": {
                "time": [min(c[1] for c in field.evaluated_contexts), max(c[1] for c in field.evaluated_contexts)],
                "y": [min(c[2] for c in field.evaluated_contexts), max(c[2] for c in field.evaluated_contexts)],
                "x": [min(c[3] for c in field.evaluated_contexts), max(c[3] for c in field.evaluated_contexts)],
            },
            "warm_seconds": warm_seconds,
            "warm_extra_model_window_calls": warm_extra_calls,
            "warm_extra_model_forward_evaluations": warm_extra_forward_evaluations,
            "repeat_max_abs_mps": repeat_max_abs,
            "exact_cached_repeat": repeat_max_abs == 0.0,
        }
    )
    np.savez_compressed(output / "wind.npz", u=u, v=v, levels=sampler.stats.levels)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    _plot(u, v, output / "wind.png", level=sampler.n_levels // 2)


if __name__ == "__main__":
    main()
