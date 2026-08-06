"""Generate long CONTIGUOUS hourly sequences from the conditional model by tiling in TIME.

The base model emits tau=4-frame blocks, which is why the temporal rows of the
benchmark are N/A: the metrics need >= MIN_FRAMES contiguous frames. InfiniteDiffusion
tiles time as well as space (spacetime_infinite: "physical time and both horizontal axes
are lazily tiled"), so a genuinely contiguous 24-hour run is a real capability of the
method, not padding.

Writes one zarr per run; run_temporal_metrics.py scores them.
"""
from __future__ import annotations

import argparse
import time as _time
from pathlib import Path

import numpy as np

from src.eval.windeval import artifact
from src.eval.windeval.generators.infinite_diffusion.spacetime import SpaceTimeSampler
from src.eval.windeval.generators.infinite_diffusion.spacetime_infinite import (
    InfiniteSpaceTimeDiffusion,
)
from src.eval.windeval.generators.infinite_diffusion.infinite_coordinates import SpaceTimeGrid

# Held-out starts (days 8-14 of Jan/Apr/Jul/Oct 2023 are the held-out window).
STARTS = ("2023-01-11T00", "2023-07-11T00")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--frames", type=int, default=24, help="contiguous hourly frames")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    dev = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={dev} frames={args.frames} size={args.size}", flush=True)

    sampler = SpaceTimeSampler(args.ckpt, num_steps=args.steps, device=dev)
    print(f"  ckpt step={sampler.step} tau={sampler.tau} levels={sampler.n_levels}",
          flush=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for run, start in enumerate(STARTS):
        # Geographic box matched to the reference's held-out window; ascending lat.
        grid = SpaceTimeGrid(lat_origin=32.25, lon_origin=232.0,
                             dlat=0.25, dlon=0.25,
                             time_origin=start, dt_hours=1)
        field = InfiniteSpaceTimeDiffusion(sampler, grid=grid, window=64, stride=32,
                                           seed=args.seed + run)
        t0 = _time.time()
        u, v = field.field_uv(0, args.frames, 0, args.size, 0, args.size)
        dur = _time.time() - t0

        lat, lon, times = grid.coordinates(t0=0, y0=0, x0=0, tau=args.frames,
                                           height=args.size, width=args.size)
        ds = artifact.make_field(u, v, level=np.asarray(sampler.stats.levels),
                                 lat=lat, lon=lon, time=times)
        out = outdir / f"idiff_temporal_{run}.zarr"
        ds.to_zarr(out, mode="w", consolidated=False, zarr_format=2)
        print(f"  run {run} start={start}: u{u.shape} in {dur / 60:.1f} min -> {out.name}",
              flush=True)
        print(f"    speed range {np.hypot(u, v).min():.2f}..{np.hypot(u, v).max():.2f} m/s",
              flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
