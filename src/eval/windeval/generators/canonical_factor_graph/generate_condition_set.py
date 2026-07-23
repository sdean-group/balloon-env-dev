"""Generate a resumable held-out condition set with CFGD and a frozen checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IDIFF_DIR = HERE.parent / "infinite_diffusion"
for directory in (HERE, IDIFF_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from core import CanonicalFactorGraphField, ChartConfig, SpaceTimeGrid  # noqa: E402
from spacetime import SpaceTimeSampler  # noqa: E402

MONTHS = (1, 4, 7, 10)
DAYS = tuple(range(8, 15))
HOURS = (0, 12)


def _name(month: int, day: int, hour: int, seed: int) -> str:
    return f"m{month:02d}_d{day:02d}_h{hour:02d}_s{seed:02d}.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=18)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--months", type=int, nargs="+", default=list(MONTHS))
    parser.add_argument("--days", type=int, nargs="+", default=list(DAYS))
    parser.add_argument("--hours", type=int, nargs="+", default=list(HOURS))
    parser.add_argument("--core-size", type=int, default=80)
    parser.add_argument("--halo-size", type=int, default=8)
    parser.add_argument("--core-time", type=int, default=2)
    parser.add_argument("--halo-time", type=int, default=1)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--time-stride", type=int, default=2)
    parser.add_argument("--query-size", type=int, default=16)
    parser.add_argument("--query-frames", type=int, default=4)
    parser.add_argument("--query-t0", type=int, default=1)
    parser.add_argument("--query-y0", type=int, default=32)
    parser.add_argument("--query-x0", type=int, default=32)
    parser.add_argument("--lat-origin", type=float, default=27.0)
    parser.add_argument("--lon-origin", type=float, default=227.0)
    parser.add_argument("--window-batch-size", type=int, default=1)
    parser.add_argument("--max-cached-charts", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    conditions = [
        (month, day, hour, seed)
        for month in args.months
        for hour in args.hours
        for day in args.days
        for seed in range(args.num_seeds)
    ]
    summary = {
        "architecture": "canonical_factor_graph_diffusion",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": sampler.step,
        "num_steps": args.num_steps,
        "num_seeds": args.num_seeds,
        "conditions": len(conditions),
        "chart_config": config.__dict__,
        "query": {
            "t0": args.query_t0,
            "y0": args.query_y0,
            "x0": args.query_x0,
            "frames": args.query_frames,
            "size": args.query_size,
        },
        "grid": {
            "lat_origin": args.lat_origin,
            "lon_origin": args.lon_origin,
            "dlat": 0.25,
            "dlon": 0.25,
        },
    }
    (args.output_dir / "config.json").write_text(json.dumps(summary, indent=2) + "\n")

    bounds = (
        args.query_t0,
        args.query_t0 + args.query_frames,
        args.query_y0,
        args.query_y0 + args.query_size,
        args.query_x0,
        args.query_x0 + args.query_size,
    )
    for index, (month, day, hour, seed) in enumerate(conditions, start=1):
        output = args.output_dir / _name(month, day, hour, seed)
        if output.exists():
            print(f"[{index}/{len(conditions)}] {output.name}: exists, skipping", flush=True)
            continue
        target = np.datetime64(f"2023-{month:02d}-{day:02d}T{hour:02d}", "h")
        origin = target - np.timedelta64(args.query_t0, "h")
        grid = SpaceTimeGrid(
            lat_origin=args.lat_origin,
            lon_origin=args.lon_origin,
            time_origin=str(origin),
        )
        field = CanonicalFactorGraphField(
            sampler,
            config=config,
            grid=grid,
            seed=seed,
            max_cached_charts=args.max_cached_charts,
        )
        started = time.perf_counter()
        u, v = field.field_uv(*bounds)
        elapsed = time.perf_counter() - started
        lat, lon, times = grid.coordinates(
            t0=args.query_t0,
            y0=args.query_y0,
            x0=args.query_x0,
            tau=args.query_frames,
            height=args.query_size,
            width=args.query_size,
        )
        temporary = output.with_suffix(".npz.part")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                u=u,
                v=v,
                levels=sampler.stats.levels,
                lat=lat,
                lon=lon,
                times=times,
                month=month,
                day=day,
                hour=hour,
                seed=seed,
                seconds=elapsed,
                charts_generated=field.charts_generated,
                model_forward_evaluations=field.model_window_evaluations,
                model_batch_calls=field.model_batch_calls,
            )
        temporary.rename(output)
        print(
            f"[{index}/{len(conditions)}] {output.name}: {elapsed:.2f}s, "
            f"{field.model_window_evaluations} forwards, "
            f"{field.charts_generated} charts",
            flush=True,
        )


if __name__ == "__main__":
    main()
