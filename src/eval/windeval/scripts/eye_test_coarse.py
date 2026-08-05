"""Eye test for the coarse-conditioning experiment: ERA5 vs plain diff vs coarse diff.

Two figures, chosen for what the numbers CANNOT show:

  1. `coarse_eye_spatial.png` — wind-speed maps at one level, one column per source and
     one row per condition. This is the "does it look like weather" test: the coarse
     input is visibly blocky, plain upsampling is visibly smooth, and the question is
     whether the diffusion rows put believable fine structure back without inventing
     nonsense.
  2. `coarse_eye_vertical.png` — the vertical structure the board says is broken.
     Left: wind bearing vs level for a sample of columns (a coherent column is a
     vertical line; real weather twists and reverses). Right: the opposing-wind column
     fraction per source against the ERA5 value, i.e. the same statistic the board
     scores, drawn so the failure mode is visible rather than tabulated.

Conventions: the diffusion model is "diff" in figures (Shaurya); wind speed uses ONE
sequential hue light→dark (never a rainbow); series colours come from benchmark.COLORS so
an entity keeps its colour across every figure; text stays in ink, never a series colour.

Run:  PYTHONPATH=. .pixi/envs/default/bin/python -m src.eval.windeval.scripts.eye_test_coarse
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import artifact
from ..reference import build_heldout
from ..benchmark import (COLORS, DATA, _cond_window, _coarse_upsample_artifacts,
                         _ref_block_at, _pick_device)
from ..metrics.structure import opposing_wind_fraction, _center

FIGDIR = Path(__file__).resolve().parents[4] / "docs" / "figures" / "benchmark_v2"
SPEED_CMAP = "Blues"                  # single hue, light→dark, CVD-safe
# the two "hardest day" conditions used elsewhere in the project
CONDITIONS = [(1, 8, 12), (7, 12, 0)]
LEVEL_IDX = 9                         # mid of the 18-level stack


def _speed(u, v):
    return np.hypot(u, v)


def _load_blocks(tag):
    """(blocks, month, day, hour) from a cached conditional-sampling npz, or None.

    A miss is REPORTED, not silent. The cache is tagged per snapshot (the benchmark is
    invoked with e.g. ``--coarse-tag m2coarse2_10k``), so a stale default tag would
    otherwise render a polite "not yet trained" panel for a model that is trained — a
    figure that is wrong rather than absent. Print what was looked for and what exists.
    """
    p = DATA / f"idiff_m2cond_blocks_{tag}.npz" if tag else None
    if p is None or not p.exists():
        have = sorted(q.name for q in DATA.glob("idiff_m2cond_blocks_*.npz"))
        print(f"[eye] MISS {p.name if p else '(no tag)'} — panel will read 'not yet "
              f"trained'. Caches present: {have or 'none'}")
        return None
    z = np.load(p)
    return z["blocks"], z["month"], z["day"], z["hour"], z["seed_idx"]


def _pick(cache, m, d, h):
    blocks, mo, dy, hr, sd = cache
    sel = np.nonzero((mo == m) & (dy == d) & (hr == h) & (sd == 0))[0]
    return blocks[sel[0]] if len(sel) else None      # (τ, L, 2, H, W)


def build(cond_tag="300k", coarse_tag="m2coarse2", factor=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(parents=True, exist_ok=True)
    ref = artifact.read(build_heldout())
    y0, x0, lat, lon = _cond_window(ref)

    plain = _load_blocks(cond_tag)
    coarse = _load_blocks(coarse_tag)

    # ---------------- figure 1: spatial ----------------
    cols = [("ERA5 (truth)", "era5"), (f"coarse input ({factor}×)", "coarsein"),
            ("coarse upsampled", "upsamp"), ("diff (no coarse)", "plain")]
    if coarse is not None:
        cols.append(("diff + coarse", "coarse"))

    fig, axes = plt.subplots(len(CONDITIONS), len(cols),
                             figsize=(2.6 * len(cols), 2.9 * len(CONDITIONS)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for r, (m, d, h) in enumerate(CONDITIONS):
        ts = (np.datetime64(f"2023-{m:02d}-{d:02d}T{h:02d}", "h")
              + np.arange(4).astype("timedelta64[h]"))
        ru, rv = _ref_block_at(ref, y0, x0, ts)
        truth = _speed(ru[0, LEVEL_IDX], rv[0, LEVEL_IDX])
        vmin, vmax = 0.0, float(np.percentile(truth, 99.5))

        panels = {}
        panels["era5"] = truth
        # the model's actual input: block-mean, shown at its native blockiness
        cu = ru[0, LEVEL_IDX].reshape(64 // factor, factor, 64 // factor, factor).mean((1, 3))
        cv = rv[0, LEVEL_IDX].reshape(64 // factor, factor, 64 // factor, factor).mean((1, 3))
        panels["coarsein"] = np.kron(_speed(cu, cv), np.ones((factor, factor)))
        import torch
        x = torch.from_numpy(np.stack([ru[0], rv[0]])[None].astype("float32"))
        lo = torch.nn.functional.avg_pool2d(x.reshape(1, -1, 64, 64), factor)
        up = torch.nn.functional.interpolate(lo, size=(64, 64), mode="bilinear",
                                             align_corners=False).numpy().reshape(2, -1, 64, 64)
        panels["upsamp"] = _speed(up[0, LEVEL_IDX], up[1, LEVEL_IDX])
        for key, cache in (("plain", plain), ("coarse", coarse)):
            b = _pick(cache, m, d, h) if cache is not None else None
            panels[key] = (_speed(b[0, LEVEL_IDX, 0], b[0, LEVEL_IDX, 1])
                           if b is not None else None)

        for c, (title, key) in enumerate(cols):
            ax = axes[r, c]
            img = panels.get(key)
            if img is None:
                ax.text(.5, .5, "not yet\ntrained", ha="center", va="center",
                        fontsize=8, color="#8a8a85", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                im = ax.imshow(img, cmap=SPEED_CMAP, vmin=vmin, vmax=vmax, origin="upper")
                ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"2023-{m:02d}-{d:02d} {h:02d}Z", fontsize=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01)
    cb.set_label("wind speed (m/s)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"Eye test — wind speed, model level index {LEVEL_IDX} "
                 f"(64² window, 0.25°)", fontsize=10)
    p1 = FIGDIR / "coarse_eye_spatial.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # ---------------- figure 2: vertical structure ----------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    m, d, h = CONDITIONS[0]
    ts = (np.datetime64(f"2023-{m:02d}-{d:02d}T{h:02d}", "h")
          + np.arange(4).astype("timedelta64[h]"))
    ru, rv = _ref_block_at(ref, y0, x0, ts)

    def bearings(u, v, n=6, seed=0):
        rng = np.random.default_rng(seed)
        ys = rng.integers(0, u.shape[-2], n); xs = rng.integers(0, u.shape[-1], n)
        return [np.degrees(np.arctan2(v[:, y, x], u[:, y, x])) for y, x in zip(ys, xs)]

    series = [("ERA5 (truth)", ru[0], rv[0], COLORS["era5 (ref)"])]
    if plain is not None:
        b = _pick(plain, m, d, h)
        if b is not None:
            series.append(("diff (no coarse)", b[0, :, 0], b[0, :, 1], COLORS["idiff m2cond"]))
    if coarse is not None:
        b = _pick(coarse, m, d, h)
        if b is not None:
            series.append(("diff + coarse", b[0, :, 0], b[0, :, 1], COLORS["idiff m2coarse"]))
    lev = np.arange(ru.shape[1])
    for name, u, v, col in series:
        for j, br in enumerate(bearings(u, v)):
            axL.plot(np.unwrap(np.radians(br)) * 180 / np.pi, lev, lw=2, color=col,
                     alpha=.75, label=name if j == 0 else None)
    axL.set_xlabel("wind bearing (deg, unwrapped)")
    axL.set_ylabel("model level index (low = high altitude)")
    axL.set_title("Vertical structure of 6 sample columns\n"
                  "(a straight line = no directional shear)", fontsize=9)
    axL.grid(True, alpha=.25, lw=.5)
    axL.spines[["top", "right"]].set_visible(False)
    axL.legend(fontsize=7, frameon=False)

    # right: the scored statistic, drawn
    ref_win = _center(ref.isel(time=slice(0, None, 4)).compute(), 64)
    era5_frac = opposing_wind_fraction(ref_win, seed=1)
    names, vals, colr = ["ERA5 (truth)"], [era5_frac], [COLORS["era5 (ref)"]]
    up_pooled, _ = _coarse_upsample_artifacts(ref, factor)
    names.append("coarse upsampled"); colr.append(COLORS["coarse upsampled"])
    vals.append(opposing_wind_fraction(up_pooled))
    for key, cache, label in (("plain", plain, "diff (no coarse)"),
                              ("coarse", coarse, "diff + coarse")):
        if cache is None:
            continue
        blocks, mo, dy, hr, sd = cache
        s0 = sd == 0
        u = blocks[s0][:, :, :, 0].reshape(-1, blocks.shape[2], 64, 64)
        v = blocks[s0][:, :, :, 1].reshape(-1, blocks.shape[2], 64, 64)
        ds = artifact.make_field(u, v, level=ref["level"].values, lat=lat, lon=lon)
        names.append(label); vals.append(opposing_wind_fraction(ds))
        colr.append(COLORS["idiff m2cond" if key == "plain" else "idiff m2coarse"])
    ypos = np.arange(len(names))
    axR.barh(ypos, vals, color=colr, height=.55)
    axR.axvline(era5_frac, color=COLORS["era5 (ref)"], lw=2, ls="--", zorder=3)
    axR.set_ylim(len(names) - .4, -.9)          # headroom for the reference label
    axR.text(era5_frac * .98, -.62, f"ERA5 = {era5_frac:.2f} ", ha="right", va="center",
             fontsize=8, color="#3d3d3c")
    for y, val in zip(ypos, vals):
        axR.text(val + .01, y, f"{val:.2f}", va="center", fontsize=8, color="#3d3d3c")
    axR.set_yticks(ypos); axR.set_yticklabels(names, fontsize=8)
    axR.set_xlabel("fraction of columns with opposing winds (>90°, both ≥5 m/s)")
    axR.set_title("Vertical decoupling — the property balloon\nstation-keeping depends on",
                  fontsize=9)
    axR.grid(True, axis="x", alpha=.25, lw=.5)
    axR.spines[["top", "right"]].set_visible(False)
    p2 = FIGDIR / "coarse_eye_vertical.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"wrote:\n  {p1}\n  {p2}")
    return p1, p2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cond-tag", default="300k")
    ap.add_argument("--coarse-tag", default="m2coarse")
    ap.add_argument("--factor", type=int, default=8)
    a = ap.parse_args(argv)
    build(a.cond_tag, a.coarse_tag, a.factor)


if __name__ == "__main__":
    main()
