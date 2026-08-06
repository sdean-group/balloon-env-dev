"""Regenerate the poster's zoom / seed-consistency montage with the CONDITIONAL model.

The original figure came from runs/idiff_m1/step_84000.pt, an unconditional M1
checkpoint tiled by the old `sampler.InfiniteDiffusion`. The conditional space-time
model needs `spacetime_infinite.InfiniteSpaceTimeDiffusion`, which ties every pixel
to a real (lat, lon, time) via SpaceTimeGrid -- so the panels below are nested
random-access queries into ONE seeded field, at real geographic scale.

Captions report physical extent, not pixel counts. At 0.25 deg per pixel:
  north-south  0.25 * 111.32                = 27.83 km / px  (constant)
  east-west    0.25 * 111.32 * cos(lat)     = 21.32 km / px  (at 40 N)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.eval.windeval.generators.infinite_diffusion.spacetime import SpaceTimeSampler
from src.eval.windeval.generators.infinite_diffusion.spacetime_infinite import (
    InfiniteSpaceTimeDiffusion,
)
from src.eval.windeval.generators.infinite_diffusion.infinite_coordinates import SpaceTimeGrid

INK = "#173B6C"
BOX = "#D74A4A"
CMAP = "viridis"

DEG_KM = 111.32


def _fonts() -> str:
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("Lato", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"):
        if n in have:
            return n
    return "DejaVu Sans"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="zoom_montage_new.png")
    ap.add_argument("--sizes", type=int, nargs="+", default=[192, 96, 48])
    ap.add_argument("--level-idx", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--center-lat", type=float, default=40.0)
    ap.add_argument("--center-lon", type=float, default=240.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={dev}  ckpt={args.ckpt}")

    sampler = SpaceTimeSampler(args.ckpt, num_steps=args.steps, device=dev)
    print(f"  step={sampler.step} conditional={sampler.conditional} tau={sampler.tau}")

    grid = SpaceTimeGrid()
    field = InfiniteSpaceTimeDiffusion(sampler, grid=grid, window=args.window,
                                       stride=args.stride, seed=args.seed)

    # Pixel coords of the requested geographic centre.
    cy = int(round((args.center_lat - grid.lat_origin) / grid.dlat))
    cx = int(round((args.center_lon - grid.lon_origin) / grid.dlon))
    print(f"  centre pixel (y={cy}, x={cx}) = {args.center_lat}N {args.center_lon}E")

    ns_km = grid.dlat * DEG_KM
    ew_km = grid.dlon * DEG_KM * np.cos(np.deg2rad(args.center_lat))

    panels, labels = [], []
    for sz in args.sizes:
        h = sz // 2
        t = time.time()
        u, v = field.field_uv(0, 1, cy - h, cy + h, cx - h, cx + h)
        speed = np.hypot(u[0, args.level_idx], v[0, args.level_idx])
        panels.append(speed)
        labels.append(f"{sz * ns_km:,.0f} × {sz * ew_km:,.0f} km")
        print(f"  {sz}x{sz}px -> {labels[-1]}   "
              f"[{speed.min():.1f}..{speed.max():.1f} m/s]  {time.time() - t:.1f}s")

    plt.rcParams["font.family"] = _fonts()
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.7), facecolor="white")
    if n == 1:
        axes = [axes]
    for i, (ax, sp, lab) in enumerate(zip(axes, panels, labels)):
        ax.imshow(sp, origin="lower", cmap=CMAP)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(lab, fontsize=30, color=INK, labelpad=16)
        if i < n - 1:
            sz, nxt = args.sizes[i], args.sizes[i + 1]
            lo = sz / 2 - nxt / 2
            ax.add_patch(plt.Rectangle((lo, lo), nxt, nxt, fill=False, ec=BOX, lw=4.0))

    fig.subplots_adjust(left=0.012, right=0.988, top=0.99, bottom=0.13, wspace=0.045)
    fig.savefig(args.out, dpi=300, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
