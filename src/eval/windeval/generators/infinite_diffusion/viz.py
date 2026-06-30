"""Tier-1 static visualization for the InfiniteDiffusion wind field (Phase 1c).

Renders directly from an :class:`InfiniteDiffusion` sampler (no artifact needed). Three
views, each of which doubles as intuition for an Axis-2 metric:

- ``plot_field``  : speed+quiver and vorticity-with-seam-overlay. A working generator shows
  *nothing* at the seam lines; a broken tiler shows them as streaks in vorticity (which is
  far more sensitive than u,v) -> this is the visual twin of the seam-discontinuity metric.
- ``plot_zoom_montage`` : same seed, panels zooming in -> multi-scale coherence + the fact
  that random-access zoom is free (extent-drift intuition).

matplotlib lives in the pixi ``viz`` feature. Uses the Agg backend (file output).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _vorticity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    dudy, _ = np.gradient(u, axis=(0, 1))
    _, dvdx = np.gradient(v, axis=(0, 1))
    return dvdx - dudy


def plot_field(sampler, y0, y1, x0, x1, *, level=0, out=None, title=None):
    """Speed+quiver (left) and vorticity with the seam grid overlaid (right)."""
    u, v = sampler.field_uv(y0, y1, x0, x1)
    u, v = u[level], v[level]
    speed = np.hypot(u, v)
    vort = _vorticity(u, v)
    seams = sampler.seam_lines(y0, y1, x0, x1)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    ext = [0, x1 - x0, 0, y1 - y0]

    im0 = ax[0].imshow(speed, origin="lower", cmap="viridis", extent=ext)
    s = max(1, min(u.shape) // 24)
    yy, xx = np.mgrid[0:u.shape[0]:s, 0:u.shape[1]:s]
    ax[0].quiver(xx, yy, u[::s, ::s], v[::s, ::s], color="white", alpha=0.8,
                 scale_units="xy", angles="xy")
    ax[0].set_title("wind speed + direction")
    fig.colorbar(im0, ax=ax[0], shrink=0.8, label="m/s")

    vmax = np.percentile(np.abs(vort), 99) or 1.0
    im1 = ax[1].imshow(vort, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=ext)
    for sx in seams["x"]:
        ax[1].axvline(sx - x0, color="k", lw=0.6, ls="--", alpha=0.5)
    for sy in seams["y"]:
        ax[1].axhline(sy - y0, color="k", lw=0.6, ls="--", alpha=0.5)
    ax[1].set_title(f"vorticity + seam grid ({len(seams['x'])}×{len(seams['y'])} stitches)")
    fig.colorbar(im1, ax=ax[1], shrink=0.8, label="1/px")

    fig.suptitle(title or f"InfiniteDiffusion wind field — level {level}, "
                          f"region y[{y0}:{y1}] x[{x0}:{x1}]")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return Path(out)
    return fig


def plot_zoom_montage(sampler, *, center=(0, 0), sizes=(512, 128, 32), level=0,
                      out=None, title=None):
    """Same seed, zooming in by region size -> multi-scale coherence + free random access."""
    cy, cx = center
    n = len(sizes)
    fig, ax = plt.subplots(1, n, figsize=(4.2 * n, 4.4))
    if n == 1:
        ax = [ax]
    for i, sz in enumerate(sizes):
        h = sz // 2
        y0, y1, x0, x1 = cy - h, cy + h, cx - h, cx + h
        u, v = sampler.field_uv(y0, y1, x0, x1)
        speed = np.hypot(u[level], v[level])
        ax[i].imshow(speed, origin="lower", cmap="viridis")
        ax[i].set_title(f"{sz}×{sz} px")
        ax[i].set_xticks([]); ax[i].set_yticks([])
        if i < n - 1:  # red box marking the next (inner) zoom
            nxt = sizes[i + 1]
            lo, hi = sz / 2 - nxt / 2, sz / 2 + nxt / 2
            ax[i].add_patch(plt.Rectangle((lo, lo), nxt, nxt, fill=False, ec="red", lw=1.5))
    fig.suptitle(title or "Multi-scale coherence (same seed, zooming in)")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return Path(out)
    return fig
