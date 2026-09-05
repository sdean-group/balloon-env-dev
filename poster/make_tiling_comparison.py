"""Build the poster's naive-tiling vs InfiniteDiffusion comparison.

The left panel uses nine independently generated 64x64 conditional-model samples.
The right panel uses the saved trained 192x192 InfiniteDiffusion field. Both show
mid-stratospheric wind speed on one shared color scale.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "eval" / "windeval" / "data"
OUT = ROOT / "poster" / "figures" / "tiling_comparison.png"


def main() -> None:
    blocks = np.load(DATA / "idiff_m2cond_blocks_4yr_250000.npz")["blocks"]
    # Nine real base-model samples. Each is an independent diffusion path.
    chosen = blocks[np.arange(9), 0, 9]  # sample, first hour, mid-level, (u,v), y, x
    speeds = np.hypot(chosen[:, 0], chosen[:, 1])
    naive = np.block(
        [[speeds[3 * row + col] for col in range(3)] for row in range(3)]
    )

    field = xr.open_zarr(DATA / "infinite_diffusion_trained.zarr", consolidated=False)
    u = field["u"].values[0, 9]
    v = field["v"].values[0, 9]
    infinite = np.hypot(u, v)

    combined = np.concatenate([naive.ravel(), infinite.ravel()])
    vmin, vmax = np.percentile(combined, [1, 99])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.8), facecolor="white")
    for index, (ax, image) in enumerate(zip(axes, [naive, infinite])):
        ax.imshow(image, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                  interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        for boundary in (64, 128):
            ax.axvline(boundary - 0.5, color="white", lw=2.0 if index == 0 else 1.35,
                       ls="-" if index == 0 else (0, (5, 5)), alpha=0.95)
            ax.axhline(boundary - 0.5, color="white", lw=2.0 if index == 0 else 1.35,
                       ls="-" if index == 0 else (0, (5, 5)), alpha=0.95)

    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.99, wspace=0.055)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
