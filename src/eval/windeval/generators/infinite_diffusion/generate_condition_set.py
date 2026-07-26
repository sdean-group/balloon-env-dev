"""Generate the held-out condition set with lazy space-time InfiniteDiffusion.

Each (month, day, hour, seed) block is saved independently, making a cluster job
resumable. Every depth/split schedule must use a separate output directory.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    from .spacetime import SpaceTimeSampler
    from .spacetime_infinite import InfiniteSpaceTimeDiffusion, SpaceTimeGrid
except ImportError:  # standalone cluster execution
    from spacetime import SpaceTimeSampler
    from spacetime_infinite import InfiniteSpaceTimeDiffusion, SpaceTimeGrid

MONTHS = (1, 4, 7, 10)
DAYS = tuple(range(8, 15))
HOURS = (0, 12)


def _name(month: int, day: int, hour: int, seed: int) -> str:
    return f"m{month:02d}_d{day:02d}_h{hour:02d}_s{seed:02d}.npz"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=18)
    parser.add_argument("--outer-depth", type=int, default=1)
    parser.add_argument("--split-step", type=int)
    parser.add_argument("--split-steps", type=int, nargs="+")
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--months", type=int, nargs="+", default=list(MONTHS))
    parser.add_argument("--days", type=int, nargs="+", default=list(DAYS))
    parser.add_argument("--hours", type=int, nargs="+", default=list(HOURS))
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--time-stride", type=int, default=2)
    args = parser.parse_args(argv)
    if args.split_step is not None and args.split_steps is not None:
        parser.error("provide --split-step or --split-steps, not both")
    if args.outer_depth == 1:
        resolved_split_steps: list[int] = []
    elif args.split_steps is not None:
        resolved_split_steps = list(args.split_steps)
    elif args.split_step is not None:
        resolved_split_steps = [args.split_step]
    else:
        resolved_split_steps = [
            round(index * args.num_steps / args.outer_depth)
            for index in range(1, args.outer_depth)
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampler = SpaceTimeSampler(
        args.checkpoint, num_steps=args.num_steps, device=args.device, use_ema=True
    )
    conditions = [(m, d, h, s) for m in args.months for h in args.hours
                  for d in args.days for s in range(args.num_seeds)]
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": sampler.step,
        "num_steps": args.num_steps,
        "outer_depth": args.outer_depth,
        "split_step": resolved_split_steps[0] if args.outer_depth == 2 else None,
        "split_steps": resolved_split_steps,
        "num_seeds": args.num_seeds,
        "window": args.window,
        "stride": args.stride,
        "time_stride": args.time_stride,
        "conditions": len(conditions),
    }
    (args.output_dir / "config.json").write_text(json.dumps(summary, indent=2) + "\n")

    for index, (month, day, hour, seed) in enumerate(conditions, start=1):
        output = args.output_dir / _name(month, day, hour, seed)
        if output.exists():
            print(f"[{index}/{len(conditions)}] {output.name}: exists, skipping", flush=True)
            continue
        target = np.datetime64(f"2023-{month:02d}-{day:02d}T{hour:02d}", "h")
        # The standard query begins at integer t=time_stride, so shift the grid origin back.
        origin = target - np.timedelta64(args.time_stride, "h")
        grid = SpaceTimeGrid(
            lat_origin=25.0,
            lon_origin=225.0,
            time_origin=str(origin),
        )
        field = InfiniteSpaceTimeDiffusion(
            sampler,
            grid=grid,
            window=args.window,
            stride=args.stride,
            time_stride=args.time_stride,
            seed=seed,
            outer_depth=args.outer_depth,
            split_steps=resolved_split_steps,
        )
        t0 = args.time_stride
        y0 = args.stride
        x0 = args.stride
        started = time.perf_counter()
        u, v = field.field_uv(
            t0, t0 + sampler.tau,
            y0, y0 + args.window,
            x0, x0 + args.window,
        )
        elapsed = time.perf_counter() - started
        lat, lon, times = grid.coordinates(
            t0=t0, y0=y0, x0=x0, tau=sampler.tau,
            height=args.window, width=args.window,
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
                model_window_calls=field.model_window_calls,
                model_forward_evaluations=field.model_forward_evaluations,
            )
        temporary.rename(output)
        print(
            f"[{index}/{len(conditions)}] {output.name}: {elapsed:.2f}s, "
            f"{field.model_forward_evaluations} forwards",
            flush=True,
        )


if __name__ == "__main__":
    main()
