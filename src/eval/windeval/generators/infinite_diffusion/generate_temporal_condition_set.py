"""Generate matched 24-hour episodes for temporal generator comparisons."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from .generate_direct_base_condition_set import sample_direct_block
    from .infinite_coordinates import SpaceTimeGrid
    from .spacetime import SpaceTimeSampler
    from .spacetime_infinite import InfiniteSpaceTimeDiffusion
except ImportError:  # standalone cluster execution
    from generate_direct_base_condition_set import sample_direct_block
    from infinite_coordinates import SpaceTimeGrid
    from spacetime import SpaceTimeSampler
    from spacetime_infinite import InfiniteSpaceTimeDiffusion

MONTHS = (1, 4, 7, 10)


def _name(month: int, day: int, seed: int) -> str:
    return f"m{month:02d}_d{day:02d}_s{seed:02d}.npz"


def _resolve_splits(
    outer_depth: int,
    num_steps: int,
    split_steps: list[int] | None,
) -> list[int]:
    if outer_depth == 1:
        return []
    if split_steps is not None:
        if len(split_steps) != outer_depth - 1:
            raise ValueError(f"T={outer_depth} requires {outer_depth - 1} split steps")
        return split_steps
    return [
        round(index * num_steps / outer_depth)
        for index in range(1, outer_depth)
    ]


def _direct_episode(
    sampler: SpaceTimeSampler,
    grid: SpaceTimeGrid,
    *,
    frames: int,
    t0: int,
    y0: int,
    x0: int,
    window: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if frames % sampler.tau:
        raise ValueError("direct episodes must contain a whole number of base-model blocks")
    u_blocks, v_blocks = [], []
    for offset in range(0, frames, sampler.tau):
        block_t0 = t0 + offset
        lat, lon, times = grid.coordinates(
            t0=block_t0,
            y0=y0,
            x0=x0,
            tau=sampler.tau,
            height=window,
            width=window,
        )
        u, v = sample_direct_block(
            sampler,
            lat=lat,
            lon=lon,
            times=times,
            global_t0=block_t0,
            global_y0=y0,
            global_x0=x0,
            seed=seed,
        )
        u_blocks.append(u)
        v_blocks.append(v)
    return np.concatenate(u_blocks), np.concatenate(v_blocks), len(u_blocks)


def _cfgd_episode(
    sampler: SpaceTimeSampler,
    grid: SpaceTimeGrid,
    *,
    frames: int,
    t0: int,
    y0: int,
    x0: int,
    window: int,
    seed: int,
    max_cached_charts: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    try:
        from ..canonical_factor_graph.core import (
            CanonicalFactorGraphField,
            ChartConfig,
        )
    except ImportError:
        generators_dir = Path(__file__).resolve().parent.parent
        if str(generators_dir) not in sys.path:
            sys.path.insert(0, str(generators_dir))
        from canonical_factor_graph.core import CanonicalFactorGraphField, ChartConfig

    config = ChartConfig(
        core_time=2,
        core_size=80,
        halo_time=1,
        halo_size=8,
        window_size=window,
        window_stride=32,
        time_stride=2,
        window_batch_size=1,
    )
    field = CanonicalFactorGraphField(
        sampler,
        config=config,
        grid=grid,
        seed=seed,
        max_cached_charts=max_cached_charts,
    )
    u, v = field.field_uv(
        t0, t0 + frames,
        y0, y0 + window,
        x0, x0 + window,
    )
    counts = {
        "charts_generated": field.charts_generated,
        "model_window_calls": field.model_window_evaluations,
        "model_batch_calls": field.model_batch_calls,
    }
    return u, v, counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("direct", "infinite", "cfgd"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=18)
    parser.add_argument("--outer-depth", type=int, default=1)
    parser.add_argument("--split-steps", type=int, nargs="+")
    parser.add_argument("--months", type=int, nargs="+", default=list(MONTHS))
    parser.add_argument("--day", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query-frames", type=int, default=24)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--time-stride", type=int, default=2)
    parser.add_argument("--cache-gb", type=float, default=4.0)
    parser.add_argument("--max-cached-charts", type=int, default=64)
    args = parser.parse_args(argv)

    splits = _resolve_splits(args.outer_depth, args.num_steps, args.split_steps)
    if args.method != "infinite" and args.outer_depth != 1:
        parser.error("--outer-depth applies only to --method infinite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampler = SpaceTimeSampler(
        args.checkpoint,
        num_steps=args.num_steps,
        device=args.device,
        use_ema=True,
    )
    summary = {
        "method": args.method,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": sampler.step,
        "num_steps": args.num_steps,
        "outer_depth": args.outer_depth if args.method == "infinite" else None,
        "split_steps": splits if args.method == "infinite" else [],
        "months": args.months,
        "day": args.day,
        "seed": args.seed,
        "query_frames": args.query_frames,
        "window": args.window,
        "stride": args.stride,
        "time_stride": args.time_stride,
        "conditions": len(args.months),
        "noise": "coordinate_keyed",
    }
    (args.output_dir / "config.json").write_text(json.dumps(summary, indent=2) + "\n")

    t0 = args.time_stride
    y0 = args.stride
    x0 = args.stride
    for index, month in enumerate(args.months, start=1):
        output = args.output_dir / _name(month, args.day, args.seed)
        if output.exists():
            print(f"[{index}/{len(args.months)}] {output.name}: exists, skipping", flush=True)
            continue

        target = np.datetime64(f"2023-{month:02d}-{args.day:02d}T00", "h")
        origin = target - np.timedelta64(t0, "h")
        grid = SpaceTimeGrid(
            lat_origin=25.0,
            lon_origin=225.0,
            time_origin=str(origin),
        )
        lat, lon, times = grid.coordinates(
            t0=t0,
            y0=y0,
            x0=x0,
            tau=args.query_frames,
            height=args.window,
            width=args.window,
        )

        started = time.perf_counter()
        counts: dict[str, int]
        if args.method == "direct":
            u, v, blocks = _direct_episode(
                sampler,
                grid,
                frames=args.query_frames,
                t0=t0,
                y0=y0,
                x0=x0,
                window=args.window,
                seed=args.seed,
            )
            counts = {
                "model_window_calls": blocks,
                "model_forward_evaluations": blocks * (2 * args.num_steps - 1),
            }
        elif args.method == "infinite":
            field = InfiniteSpaceTimeDiffusion(
                sampler,
                grid=grid,
                window=args.window,
                stride=args.stride,
                time_stride=args.time_stride,
                seed=args.seed,
                outer_depth=args.outer_depth,
                split_steps=splits,
                cache_bytes=int(args.cache_gb * 1024**3),
            )
            u, v = field.field_uv(
                t0, t0 + args.query_frames,
                y0, y0 + args.window,
                x0, x0 + args.window,
            )
            counts = {
                "model_window_calls": field.model_window_calls,
                "model_forward_evaluations": field.model_forward_evaluations,
            }
        else:
            u, v, counts = _cfgd_episode(
                sampler,
                grid,
                frames=args.query_frames,
                t0=t0,
                y0=y0,
                x0=x0,
                window=args.window,
                seed=args.seed,
                max_cached_charts=args.max_cached_charts,
            )
        elapsed = time.perf_counter() - started

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
                day=args.day,
                seed=args.seed,
                seconds=elapsed,
                **counts,
            )
        temporary.rename(output)
        print(
            f"[{index}/{len(args.months)}] {output.name}: {elapsed:.2f}s, "
            f"{counts}",
            flush=True,
        )


if __name__ == "__main__":
    main()
