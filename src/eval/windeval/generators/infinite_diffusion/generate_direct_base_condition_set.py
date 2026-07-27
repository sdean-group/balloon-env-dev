"""Generate matched held-out samples directly from the finite base block model."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

try:
    from .infinite_coordinates import SpaceTimeGrid, coordinate_noise
    from .spacetime import SpaceTimeSampler
except ImportError:  # standalone cluster execution
    from infinite_coordinates import SpaceTimeGrid, coordinate_noise
    from spacetime import SpaceTimeSampler

MONTHS = (1, 4, 7, 10)
DAYS = tuple(range(8, 15))
HOURS = (0, 12)


def _name(month: int, day: int, hour: int, seed: int) -> str:
    return f"m{month:02d}_d{day:02d}_h{hour:02d}_s{seed:02d}.npz"


@torch.no_grad()
def sample_direct_block(
    sampler: SpaceTimeSampler,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    times: np.ndarray,
    global_t0: int,
    global_y0: int,
    global_x0: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one block using the wrapper's coordinate-keyed noise, without overlaps."""
    height = len(lat)
    width = len(lon)
    noise = coordinate_noise(
        sampler.n_channels,
        sampler.tau,
        height,
        width,
        t0=global_t0,
        y0=global_y0,
        x0=global_x0,
        seed=seed,
        device=sampler.device,
        dtype=torch.float32,
    )
    model_input = noise.permute(1, 0, 2, 3).unsqueeze(0)
    cond, tfeat = sampler._condition((height, width), lat, lon, times)
    block = sampler._heun_block(model_input, cond=cond, tfeat=tfeat)[0]
    block = sampler.stats.denormalize(block)
    block = block.reshape(
        sampler.tau,
        sampler.n_levels,
        2,
        height,
        width,
    )
    array = block.detach().cpu().numpy()
    return array[:, :, 0], array[:, :, 1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=18)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--months", type=int, nargs="+", default=list(MONTHS))
    parser.add_argument("--days", type=int, nargs="+", default=list(DAYS))
    parser.add_argument("--hours", type=int, nargs="+", default=list(HOURS))
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--time-stride", type=int, default=2)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampler = SpaceTimeSampler(
        args.checkpoint,
        num_steps=args.num_steps,
        device=args.device,
        use_ema=True,
    )
    conditions = [
        (month, day, hour, seed)
        for month in args.months
        for hour in args.hours
        for day in args.days
        for seed in range(args.num_seeds)
    ]
    summary = {
        "method": "direct_base_model",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": sampler.step,
        "num_steps": args.num_steps,
        "num_seeds": args.num_seeds,
        "window": args.window,
        "stride": args.stride,
        "time_stride": args.time_stride,
        "conditions": len(conditions),
        "noise": "coordinate_keyed",
    }
    (args.output_dir / "config.json").write_text(json.dumps(summary, indent=2) + "\n")

    for index, (month, day, hour, seed) in enumerate(conditions, start=1):
        output = args.output_dir / _name(month, day, hour, seed)
        if output.exists():
            print(f"[{index}/{len(conditions)}] {output.name}: exists, skipping", flush=True)
            continue

        target = np.datetime64(f"2023-{month:02d}-{day:02d}T{hour:02d}", "h")
        origin = target - np.timedelta64(args.time_stride, "h")
        grid = SpaceTimeGrid(
            lat_origin=25.0,
            lon_origin=225.0,
            time_origin=str(origin),
        )
        t0 = args.time_stride
        y0 = args.stride
        x0 = args.stride
        lat, lon, times = grid.coordinates(
            t0=t0,
            y0=y0,
            x0=x0,
            tau=sampler.tau,
            height=args.window,
            width=args.window,
        )

        started = time.perf_counter()
        u, v = sample_direct_block(
            sampler,
            lat=lat,
            lon=lon,
            times=times,
            global_t0=t0,
            global_y0=y0,
            global_x0=x0,
            seed=seed,
        )
        elapsed = time.perf_counter() - started
        forward_evaluations = 2 * args.num_steps - 1

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
                model_window_calls=1,
                model_forward_evaluations=forward_evaluations,
            )
        temporary.rename(output)
        print(
            f"[{index}/{len(conditions)}] {output.name}: {elapsed:.2f}s, "
            f"{forward_evaluations} forwards",
            flush=True,
        )


if __name__ == "__main__":
    main()
