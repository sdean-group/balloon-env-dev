"""Generate one resumable 4/16/64-core-tile condition set."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

try:
    from ..infinite_coordinates import SpaceTimeGrid
    from ..spacetime import SpaceTimeSampler
    from ..spacetime_infinite import InfiniteSpaceTimeDiffusion
    from .protocol import QUERY_SIZE, profile_for
except ImportError:  # pragma: no cover - standalone cluster execution
    from infinite_coordinates import SpaceTimeGrid
    from spacetime import SpaceTimeSampler
    from spacetime_infinite import InfiniteSpaceTimeDiffusion
    from protocol import QUERY_SIZE, profile_for

MONTHS = (1, 4, 7, 10)
DAYS = tuple(range(8, 15))
HOURS = (0, 12)


def _name(month: int, day: int, hour: int, seed: int) -> str:
    return f"m{month:02d}_d{day:02d}_h{hour:02d}_s{seed:02d}.npz"


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--core-tiles", type=int, choices=(4, 16, 64), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=18)
    parser.add_argument("--outer-depth", type=int, default=1)
    parser.add_argument("--split-steps", type=int, nargs="+")
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--months", type=int, nargs="+", default=list(MONTHS))
    parser.add_argument("--days", type=int, nargs="+", default=list(DAYS))
    parser.add_argument("--hours", type=int, nargs="+", default=list(HOURS))
    parser.add_argument("--query-origin", type=int, default=32)
    parser.add_argument("--cache-gb", type=float, default=2.0)
    args = parser.parse_args(argv)
    profile = profile_for(args.core_tiles)
    if args.outer_depth == 1:
        split_steps: list[int] = []
    elif args.split_steps is not None:
        split_steps = list(args.split_steps)
    else:
        split_steps = [
            round(index * args.num_steps / args.outer_depth)
            for index in range(1, args.outer_depth)
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampler = SpaceTimeSampler(
        args.checkpoint, num_steps=args.num_steps, device=args.device, use_ema=True
    )
    conditions = [
        (month, day, hour, seed)
        for month in args.months
        for hour in args.hours
        for day in args.days
        for seed in range(args.num_seeds)
    ]
    config = {
        "protocol": "fixed_64x64_domain_50_percent_overlap",
        "core_tiles": profile.core_tiles,
        "tiles_per_axis": profile.tiles_per_axis,
        "window": profile.window,
        "stride": profile.stride,
        "query_size": QUERY_SIZE,
        "query_origin": args.query_origin,
        "expected_final_windows_including_halo": profile.expected_final_windows,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": sampler.step,
        "num_steps": args.num_steps,
        "outer_depth": args.outer_depth,
        "split_steps": split_steps,
        "num_seeds": args.num_seeds,
        "months": list(args.months),
        "days": list(args.days),
        "hours": list(args.hours),
        "conditions": len(conditions),
    }
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != config:
            raise ValueError(
                f"{args.output_dir} contains a different experiment configuration; "
                "choose a new output directory"
            )
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    for index, (month, day, hour, seed) in enumerate(conditions, start=1):
        output = args.output_dir / _name(month, day, hour, seed)
        if output.exists():
            print(f"[{index}/{len(conditions)}] {output.name}: exists, skipping", flush=True)
            continue
        target = np.datetime64(f"2023-{month:02d}-{day:02d}T{hour:02d}", "h")
        grid = SpaceTimeGrid(
            lat_origin=25.0,
            lon_origin=225.0,
            time_origin=str(target),
        )
        if sampler.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(sampler.device)
        field = InfiniteSpaceTimeDiffusion(
            sampler,
            grid=grid,
            window=profile.window,
            stride=profile.stride,
            time_stride=sampler.tau,
            seed=seed,
            outer_depth=args.outer_depth,
            split_steps=split_steps,
            cache_bytes=int(args.cache_gb * 1024**3),
        )
        q0 = args.query_origin
        _synchronize(sampler.device)
        started = time.perf_counter()
        u, v = field.field_uv(0, 1, q0, q0 + QUERY_SIZE, q0, q0 + QUERY_SIZE)
        _synchronize(sampler.device)
        elapsed = time.perf_counter() - started
        lat, lon, times = grid.coordinates(
            t0=0,
            y0=q0,
            x0=q0,
            tau=1,
            height=QUERY_SIZE,
            width=QUERY_SIZE,
        )
        final_phase = "initial" if args.outer_depth == 1 else (
            "continuation" if args.outer_depth == 2
            else f"continuation_{args.outer_depth - 1}"
        )
        final_windows = int(field.phase_window_calls[final_phase])
        if final_windows != profile.expected_final_windows:
            raise RuntimeError(
                f"profile expected {profile.expected_final_windows} final windows, "
                f"observed {final_windows}"
            )
        peak_memory_mb = (
            float(torch.cuda.max_memory_allocated(sampler.device) / 1024**2)
            if sampler.device.type == "cuda"
            else np.nan
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
                core_tiles=profile.core_tiles,
                final_windows=final_windows,
                seconds=elapsed,
                peak_memory_mb=peak_memory_mb,
                model_window_calls=field.model_window_calls,
                model_forward_evaluations=field.model_forward_evaluations,
            )
        temporary.rename(output)
        print(
            f"[{index}/{len(conditions)}] {output.name}: {elapsed:.2f}s, "
            f"{final_windows} final windows, {field.model_forward_evaluations} forwards",
            flush=True,
        )


if __name__ == "__main__":
    main()
