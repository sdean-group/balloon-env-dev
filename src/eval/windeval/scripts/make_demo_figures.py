"""Regenerate the demo figures for docs/wind-eval-demo.md.

Three views of the *trained* InfiniteDiffusion wind field:
  1. fig_trained_field.png  — speed+quiver | vorticity+seam grid (realism + seamlessness)
  2. fig_zoom_montage.png   — same seed, zooming in (infinite extent + multi-scale coherence)
  3. fig_compare.png        — ERA5 (truth) | trained (ours) | toy (machinery baseline) speed maps

Run (full pixi env + viz feature):
    PYTHONPATH=. .pixi/envs/default/bin/python -m src.eval.windeval.scripts.make_demo_figures \
        --ckpt runs/idiff_m1/step_84000.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .. import artifact
from ..generators.infinite_diffusion import build_sampler, InfiniteDiffusion, ToyDivFreeDenoiser
from ..generators.infinite_diffusion import viz

REPO = Path(__file__).resolve().parents[4]   # .../balloon-env-dev
FIGS = REPO / "docs" / "figures"
DATA = Path(__file__).resolve().parents[1] / "data"
LEVEL = 9  # mid-band (model level ~58 of the 49-66 stratospheric band)


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/idiff_m1/step_84000.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=18)
    args = ap.parse_args(argv)
    dev = args.device or _device()
    FIGS.mkdir(parents=True, exist_ok=True)
    print(f"[figs] trained={args.ckpt} device={dev} -> {FIGS}")

    trained = build_sampler(args.ckpt, num_steps=args.steps, window=64, seed=0, device=dev,
                            cache_bytes=None)

    # 1. hero: speed+quiver | vorticity + seam grid
    viz.plot_field(trained, 0, 192, 0, 192, level=LEVEL, out=FIGS / "fig_trained_field.png",
                   title="Trained InfiniteDiffusion wind (192×192 px ≈ 5400 km) — "
                         f"level {LEVEL} (~58 hPa band)")
    print("  wrote fig_trained_field.png")

    # 2. infinite extent + multi-scale coherence (concentric -> cache makes inner panels cheap)
    viz.plot_zoom_montage(trained, center=(0, 0), sizes=(192, 96, 48), level=LEVEL,
                          out=FIGS / "fig_zoom_montage.png",
                          title="Same seed, zooming in — random-access into an unbounded field")
    print("  wrote fig_zoom_montage.png")

    # 3. comparison: ERA5 truth | trained | toy (speed maps, shared colour scale)
    era5 = artifact.read(DATA / "era5_real.zarr")
    e_u = era5["u"].values[0, LEVEL]
    e_v = era5["v"].values[0, LEVEL]
    e_speed = np.hypot(e_u, e_v)

    tu, tv = trained.field_uv(0, 64, 0, 64)
    t_speed = np.hypot(tu[LEVEL], tv[LEVEL])

    toy = InfiniteDiffusion(ToyDivFreeDenoiser(era5.sizes["level"]), window=64, stride=32,
                            T=2, seed=0)
    yu, yv = toy.field_uv(0, 64, 0, 64)
    y_speed = np.hypot(yu[LEVEL], yv[LEVEL])

    # per-panel colour scale: the point is spatial STRUCTURE (what the Axis-1 metrics measure,
    # all scale-invariant); absolute amplitude differs (the trained field is smoother/calmer —
    # noted as a limitation in the writeup) and a shared scale would just wash it out.
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
    for a, (sp, ttl) in zip(ax, [
        (e_speed, "ERA5 reanalysis (truth)\nAxis-1 COMPOSITE 0.93"),
        (t_speed, "Trained InfiniteDiffusion (ours)\nAxis-1 0.66 · Axis-2 0.88"),
        (y_speed, "Analytic toy (machinery baseline)\nAxis-1 0.48 · Axis-2 0.98"),
    ]):
        im = a.imshow(sp, origin="lower", cmap="viridis")
        a.set_title(ttl, fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, shrink=0.7, label="m/s")
    fig.suptitle("Wind-speed structure, mid-stratosphere (each panel scaled to its own range)",
                 fontsize=12)
    fig.savefig(FIGS / "fig_compare.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_compare.png")


if __name__ == "__main__":
    main()
