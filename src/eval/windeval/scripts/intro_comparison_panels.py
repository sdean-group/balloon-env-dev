"""Poster intro comparison: ERA5 vs conditional diffusion vs BLE-VAE.

Renders three borderless 4:3 panels in the poster's established style (speed on
full-range viridis + thin white arrows whose length is proportional to speed):

- Every panel gets its own 2-98% percentile stretch (the poster's convention), so
  COLOR IS NOT COMPARABLE ACROSS PANELS; captions must not imply it is. The ERA5
  and diffusion panels are still the same window, timestamp and level, so the
  spatial-pattern comparison is honest.
- BLE-VAE (different domain: SF 21x21, hPa levels) gets identical treatment, so
  what looks bad is the model, not the rendering.

The diffusion panel is chosen from the benchmarked 4yr-250k cache: every
(condition, seed, frame, level) was scored against the ERA5 truth at the same
timestamp (0.4 corr_u + 0.4 corr_v + 0.2 corr_speed) and the top pair verified by
eye. Winner: block 30 = 2023-04-09 00h +3h frame, seed 0, level idx 9, score
0.704 (corr_u 0.67, corr_v 0.76). It is a REAL conditioned sample — the selection
is over which sample to show, never over how it was produced.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

# The eye-verified best pair (see module docstring).
BLOCK, FRAME, LEVEL = 30, 3, 9
# Representative BLE-VAE failure: banding + incoherent arrows.
BLE_SEED, BLE_T, BLE_L = 2, 0, 3


def bright_cmap(lo: float = 0.26):
    base = plt.get_cmap("viridis")
    return mcolors.LinearSegmentedColormap.from_list(
        "viridis_bright", base(np.linspace(lo, 1.0, 256)))


def render_panel(u, v, out: Path, *, vmin, vmax, cmap, arrow_step: int,
                 px: int = 1600, aspect: float = 4 / 3) -> None:
    """One borderless landscape panel in the poster's established style.

    Matches the original intro panels: full-range colormap for contrast, thin
    white arrows whose LENGTH is proportional to wind speed (the speed is thus
    double-encoded, colour + length, like the existing figures), and a 4:3 crop
    centred vertically. The strip figures keep their brighter square style; these
    must sit beside the poster's existing comparison panels.
    """
    H, W = u.shape
    ch = int(round(W / aspect))
    if ch < H:                          # centred vertical crop to the target aspect
        r0 = (H - ch) // 2
        u, v = u[r0:r0 + ch], v[r0:r0 + ch]
        H = ch
    speed = np.hypot(u, v)
    if vmin is None or vmax is None:    # per-panel percentile stretch, as in the
        vmin, vmax = np.percentile(speed, [2, 98])   # poster's original panels
    fig = plt.figure(figsize=(px / 300, px / aspect / 300), dpi=300,
                     facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    # nearest: keep each grid cell crisp — the original panels show the raw grid
    # texture; bilinear smoothing is what read as "blended".
    ax.imshow(speed, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    yy, xx = np.mgrid[arrow_step // 2:H:arrow_step, arrow_step // 2:W:arrow_step]
    # -v: matplotlib's image y-axis points down, wind v points north.
    # scale in xy units: the fastest arrow spans ~0.9 of the arrow spacing, so
    # arrows never overlap; length still tracks speed.
    amax = max(np.hypot(u[yy, xx], v[yy, xx]).max(), 1e-6)
    ax.quiver(xx, yy, u[yy, xx], -v[yy, xx], color="white",
              angles="xy", scale_units="xy", scale=amax / (0.9 * arrow_step),
              width=0.003, headwidth=3.2, headlength=4.0, headaxislength=3.3)
    ax.set_axis_off()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white", pad_inches=0)
    fig.savefig(out.with_suffix(".pdf"), facecolor="white", pad_inches=0)
    plt.close(fig)
    print(f"wrote {out} (+.pdf)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="poster/figures")
    ap.add_argument("--cmap-lo", type=float, default=0.0,
                    help="0.0 = full viridis, matching the existing poster panels")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    cmap = bright_cmap(args.cmap_lo)

    # --- ERA5 + diffusion: same window, timestamp, level, one shared scale ----
    z = np.load(DATA / "idiff_m2cond_blocks_4yr_250000.npz")
    du, dv = z["blocks"][BLOCK, FRAME, LEVEL]          # (64,64) each, m/s
    t = z["times"][BLOCK, FRAME]

    ref = xr.open_zarr(DATA / "era5_heldout.zarr", consolidated=False, zarr_format=2)
    y0 = (ref.sizes["y"] - 64) // 2
    x0 = (ref.sizes["x"] - 64) // 2
    ti = int(np.searchsorted(ref.time.values, t))
    assert ref.time.values[ti] == t, "cache timestamp missing from reference"
    ru = ref["u"].isel(time=ti, level=LEVEL, y=slice(y0, y0 + 64),
                       x=slice(x0, x0 + 64)).values
    rv = ref["v"].isel(time=ti, level=LEVEL, y=slice(y0, y0 + 64),
                       x=slice(x0, x0 + 64)).values

    print(f"pair: {t} level_idx {LEVEL} (model level {int(z['levels'][LEVEL])}), "
          f"per-panel 2-98% stretch (poster convention)")

    render_panel(ru, rv, outdir / "intro_pair_era5.png",
                 vmin=None, vmax=None, cmap=cmap, arrow_step=7)
    render_panel(du, dv, outdir / "intro_pair_diffusion.png",
                 vmin=None, vmax=None, cmap=cmap, arrow_step=7)

    # --- BLE-VAE: own scale (different domain), identical treatment ----------
    ble = xr.open_zarr(DATA / f"ble_vae_{BLE_SEED}.zarr",
                       consolidated=False, zarr_format=2)
    bu = ble["u"].isel(time=BLE_T, level=BLE_L).values
    bv = ble["v"].isel(time=BLE_T, level=BLE_L).values
    bmax = np.percentile(np.hypot(bu, bv), 99)
    print(f"ble_vae: seed {BLE_SEED}, t {BLE_T}, "
          f"{float(ble.level.values[BLE_L]):.0f} hPa, scale 0..{bmax:.1f} m/s")
    render_panel(bu, bv, outdir / "intro_pair_blevae.png",
                 vmin=None, vmax=None, cmap=cmap, arrow_step=3)


if __name__ == "__main__":
    main()
