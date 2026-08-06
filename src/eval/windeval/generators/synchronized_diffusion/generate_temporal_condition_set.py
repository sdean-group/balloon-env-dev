"""Generate four matched 24-hour episodes for one synchronization rule."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GENERATORS_DIR = HERE.parent
IDIFF_DIR = GENERATORS_DIR / "infinite_diffusion"
for directory in (GENERATORS_DIR, IDIFF_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from canonical_factor_graph.core import ChartConfig, SpaceTimeGrid  # noqa: E402
from spacetime import SpaceTimeSampler  # noqa: E402
from synchronized_diffusion.core import SynchronizedChartField  # noqa: E402

MONTHS = (1, 4, 7, 10)


def _name(month: int, day: int, seed: int) -> str:
    return f"m{month:02d}_d{day:02d}_s{seed:02d}.npz"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=("sync_tweedies", "overlap_guided", "consensus_equilibrium"),
        required=True,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=18)
    parser.add_argument("--months", type=int, nargs="+", default=list(MONTHS))
    parser.add_argument("--day", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query-frames", type=int, default=24)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--time-stride", type=int, default=2)
    parser.add_argument("--max-cached-charts", type=int, default=64)
    parser.add_argument("--guidance-strength", type=float, default=0.15)
    parser.add_argument("--consensus-rounds", type=int, default=2)
    parser.add_argument("--consensus-relaxation", type=float, default=0.5)
    parser.add_argument("--dual-scale", type=float, default=0.25)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampler = SpaceTimeSampler(
        args.checkpoint,
        num_steps=args.num_steps,
        device=args.device,
        use_ema=True,
    )
    chart_config = ChartConfig(
        core_time=2,
        core_size=80,
        halo_time=1,
        halo_size=8,
        window_size=args.window,
        window_stride=args.stride,
        time_stride=args.time_stride,
        window_batch_size=1,
    )
    summary = {
        "architecture": "synchronized_chart_diffusion",
        "strategy": args.strategy,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": sampler.step,
        "num_steps": args.num_steps,
        "months": args.months,
        "day": args.day,
        "seed": args.seed,
        "query_frames": args.query_frames,
        "window": args.window,
        "stride": args.stride,
        "time_stride": args.time_stride,
        "conditions": len(args.months),
        "chart_config": chart_config.__dict__,
        "strategy_parameters": {
            "guidance_strength": args.guidance_strength,
            "consensus_rounds": args.consensus_rounds,
            "consensus_relaxation": args.consensus_relaxation,
            "dual_scale": args.dual_scale,
        },
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
        field = SynchronizedChartField(
            sampler,
            strategy=args.strategy,
            config=chart_config,
            grid=grid,
            seed=args.seed,
            max_cached_charts=args.max_cached_charts,
            guidance_strength=args.guidance_strength,
            consensus_rounds=args.consensus_rounds,
            consensus_relaxation=args.consensus_relaxation,
            dual_scale=args.dual_scale,
        )
        started = time.perf_counter()
        u, v = field.field_uv(
            t0,
            t0 + args.query_frames,
            y0,
            y0 + args.window,
            x0,
            x0 + args.window,
        )
        elapsed = time.perf_counter() - started
        lat, lon, times = grid.coordinates(
            t0=t0,
            y0=y0,
            x0=x0,
            tau=args.query_frames,
            height=args.window,
            width=args.window,
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
                day=args.day,
                seed=args.seed,
                seconds=elapsed,
                charts_generated=field.charts_generated,
                model_forward_evaluations=field.model_window_evaluations,
                model_batch_calls=field.model_batch_calls,
                overlap_objective_evaluations=field.overlap_objective_evaluations,
                consensus_iterations=field.consensus_iterations,
            )
        temporary.rename(output)
        print(
            f"[{index}/{len(args.months)}] {output.name}: {elapsed:.2f}s, "
            f"{field.model_window_evaluations} forwards, "
            f"{field.charts_generated} charts",
            flush=True,
        )


if __name__ == "__main__":
    main()
