"""Render the poster's diffusion explainer strip from captured trajectory states.

Lay-audience figure: random static on the left becomes a generated wind field on
the right. Wind is encoded by COLOR ONLY (speed on a perceptually-uniform ramp) --
no direction arrows, so nothing implies colour encodes direction.

Panels come from diffusion_strip_sample.py, i.e. the real reverse trajectory of
the trained model, not a re-noised final sample.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow

HERE = Path(__file__).resolve().parent

INK = "#173B6C"        # navy   -- primary text
MUTED = "#5F6E82"      # gray   -- secondary text, process arrows
CMAP = "viridis"       # sequential magnitude, monotonic lightness, CVD-safe

# Chosen from the 19-state trajectory for an even visual progression. 13 -> 14 for
# the fourth slot: 13 is still visibly grainy and made the jump to the final panel
# read as a discontinuity.
PANELS = (0, 11, 12, 14, 18)
CAPTIONS = ("pure noise", "", "", "", "generated wind field")


def _pick_font() -> str:
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Lato", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"):
        if name in have:
            return name
    return "DejaVu Sans"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=str(HERE / "diffusion_strip_states.npz"))
    ap.add_argument("--out", default="poster/figures/diffusion_explainer.png")
    ap.add_argument("--pdf", action="store_true", help="also write a .pdf alongside")
    ap.add_argument("--no-process-arrows", action="store_true",
                    help="drop the forward/backward process arrows entirely")
    args = ap.parse_args()

    z = np.load(args.states)
    speeds = z["speeds"]
    panels = [speeds[i] for i in PANELS]

    plt.rcParams["font.family"] = _pick_font()

    n = len(panels)
    fig_w, fig_h = 14.0, 6.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Geometry in figure coords: a row of square panels with even gutters. Vertical
    # bands are laid out explicitly (title / arrow / panels / arrow / caption /
    # legend) so nothing collides or runs off the canvas.
    left, right = 0.035, 0.965
    gutter = 0.011
    span = right - left
    pw = (span - gutter * (n - 1)) / n
    ph = pw * fig_w / fig_h
    py = 0.375

    for k, (idx, field) in enumerate(zip(PANELS, panels)):
        ax = fig.add_axes([left + k * (pw + gutter), py, pw, ph])
        lo, hi = np.percentile(field, [2, 98])
        ax.imshow(field, cmap=CMAP, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if CAPTIONS[k]:
            ax.set_xlabel(CAPTIONS[k], fontsize=13, color=INK, labelpad=9)

    x0, x1 = left, left + n * pw + (n - 1) * gutter
    xmid = (x0 + x1) / 2
    top_y = py + ph + 0.055
    bot_y = 0.275

    if not args.no_process_arrows:
        akw = dict(width=0.0013, head_width=0.016, head_length=0.008,
                   length_includes_head=True, color=MUTED, transform=fig.transFigure)
        fig.add_artist(FancyArrow(x0, top_y, x1 - x0, 0, **akw))
        fig.add_artist(FancyArrow(x1, bot_y, x0 - x1, 0, **akw))
        fig.text(xmid, top_y + 0.030,
                 "denoising  —  the model turns random static into a wind field",
                 ha="center", va="bottom", fontsize=15, color=INK)
        fig.text(xmid, bot_y - 0.042,
                 "noising  —  training adds static to real wind data, step by step",
                 ha="center", va="top", fontsize=15, color=MUTED)

    # Speed legend: colour-only encoding, labelled at the ends without arrows.
    cb_w, cb_h = 0.14, 0.020
    cb_x = xmid - cb_w / 2
    cb_y = 0.085 if not args.no_process_arrows else 0.13
    fig.text(xmid, cb_y + cb_h + 0.020, "wind speed", ha="center", va="bottom",
             fontsize=12, color=MUTED)
    cax = fig.add_axes([cb_x, cb_y, cb_w, cb_h])
    cax.imshow(np.linspace(0, 1, 256)[None], aspect="auto", cmap=CMAP)
    cax.set_xticks([]); cax.set_yticks([])
    for s in cax.spines.values():
        s.set_visible(False)
    fig.text(cb_x - 0.010, cb_y + cb_h / 2, "slower", ha="right", va="center",
             fontsize=12, color=MUTED)
    fig.text(cb_x + cb_w + 0.010, cb_y + cb_h / 2, "faster", ha="left", va="center",
             fontsize=12, color=MUTED)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"wrote {out}  ({fig_w * 300:.0f}x{fig_h * 300:.0f} px)")
    if args.pdf:
        fig.savefig(out.with_suffix(".pdf"), facecolor="white")
        print(f"wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
