"""Draw D05-style zoom connectors onto the finished montage PNG.

Post-processing only -- it reads the exported figure and writes it back at the
SAME pixel dimensions and dpi. That matters because re-rendering the montage
would mean re-sampling the 192x192 field (hours on MPS), and nothing about
drawing a line needs the model.

Panel and red-box geometry are detected from the pixels rather than hard-coded,
so this keeps working if the montage is re-exported at another size.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BOX_RGB = (215, 74, 74)          # #D74A4A, the existing zoom-box stroke
WHITE_CUT = 246                  # below this on any channel => "not background"


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Contiguous True runs in a 1-D boolean mask, as inclusive (start, end)."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1)
    groups = np.split(idx, splits + 1)
    return [(int(g[0]), int(g[-1])) for g in groups if g.size >= min_len]


def detect_panels(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Locate the image panels (colourful blocks) as (x0, y0, x1, y1)."""
    ink = (rgb[:, :, :3] < WHITE_CUT).any(axis=2)
    # Panels are the tall solid blocks; captions are short and sparse. Use the
    # column profile over the upper region, where only panels live.
    upper = ink[: int(ink.shape[0] * 0.75)]
    cols = upper.sum(axis=0) > upper.shape[0] * 0.5
    panels = []
    for x0, x1 in _runs(cols, min_len=50):
        rows = ink[:, x0:x1 + 1].sum(axis=1) > (x1 - x0) * 0.5
        yr = _runs(rows, min_len=50)
        y0, y1 = yr[0]
        panels.append((x0, y0, x1, y1))
    return panels


def detect_boxes(rgb: np.ndarray, tol: int = 42) -> list[tuple[int, int, int, int]]:
    """Locate the red zoom boxes as (x0, y0, x1, y1)."""
    d = np.abs(rgb[:, :, :3].astype(int) - np.array(BOX_RGB)).sum(axis=2)
    red = d < tol
    cols = red.any(axis=0)
    boxes = []
    for x0, x1 in _runs(cols, min_len=20):
        sub = red[:, x0:x1 + 1]
        rows = np.flatnonzero(sub.any(axis=1))
        boxes.append((x0, int(rows[0]), x1, int(rows[-1])))
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="poster/figures/zoom_montage.png")
    ap.add_argument("--out", default=None, help="default: overwrite --image")
    ap.add_argument("--width", type=int, default=9, help="line width in px")
    ap.add_argument("--dry-run", action="store_true", help="report geometry only")
    args = ap.parse_args()

    src = Path(args.image)
    im = Image.open(src).convert("RGBA")
    rgb = np.asarray(im)
    print(f"{src.name}: {im.size[0]}x{im.size[1]} dpi={im.info.get('dpi')}")

    panels = detect_panels(rgb)
    boxes = detect_boxes(rgb)
    print(f"detected {len(panels)} panels, {len(boxes)} red boxes")
    for i, p in enumerate(panels):
        print(f"  panel {i}: x {p[0]}..{p[2]}  y {p[1]}..{p[3]}")
    for i, b in enumerate(boxes):
        print(f"  box   {i}: x {b[0]}..{b[2]}  y {b[1]}..{b[3]}")

    if len(panels) < 2 or len(boxes) < len(panels) - 1:
        raise SystemExit("geometry detection failed; inspect with --dry-run")
    if args.dry_run:
        return

    d = ImageDraw.Draw(im)
    for i, box in enumerate(boxes[: len(panels) - 1]):
        nxt = panels[i + 1]
        bx1, by0, by1 = box[2], box[1], box[3]        # box right edge, top, bottom
        nx0, ny0, ny1 = nxt[0], nxt[1], nxt[3]        # next panel left edge, top, bottom
        d.line([(bx1, by0), (nx0, ny0)], fill=BOX_RGB, width=args.width)
        d.line([(bx1, by1), (nx0, ny1)], fill=BOX_RGB, width=args.width)
        print(f"  connector {i}: ({bx1},{by0})->({nx0},{ny0}) and "
              f"({bx1},{by1})->({nx0},{ny1})")

    out = Path(args.out) if args.out else src
    dpi = im.info.get("dpi")
    im.save(out, dpi=dpi) if dpi else im.save(out)
    chk = Image.open(out)
    print(f"wrote {out}: {chk.size[0]}x{chk.size[1]} dpi={chk.info.get('dpi')}")


if __name__ == "__main__":
    main()
