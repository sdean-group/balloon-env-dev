"""Training loss curves + the benchmark metric they fail to predict.

Two panels, and the pairing IS the point: the loss falls monotonically on every run
while the benchmark's effective-resolution metric gets WORSE. A loss curve alone would
read as "the run succeeded"; panel B is what the loss cannot see, because the EDM
objective is dominated by the large scales the model already solves.

Data: parsed from SLURM `[train] step N/M loss X` lines (runs/logs/*.out), which is
the same series W&B logged for the 4-yr run. Benchmark points come from
docs/benchmark-reports/benchmark_v2_*.md.

    PYTHONPATH=. python -m src.eval.windeval.scripts.plot_loss_curves
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
LOGS = ROOT / "runs" / "logs"
OUT = ROOT / "docs" / "figures" / "training"

# Colorblind-safe, distinct in both hue and lightness (Okabe-Ito subset).
C_4YR = "#0072B2"   # blue
C_1YR = "#D55E00"   # vermillion
C_CRS = "#009E73"   # bluish green
INK = "#222222"
MUTED = "#8A8A8A"

STEP_RE = re.compile(r"^\[train\] step\s+(\d+)/(\d+)\s+loss\s+([0-9.]+)")


def parse_logs(files: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Merge legs into one series, keyed by global step so restarts do not duplicate."""
    d: dict[int, float] = {}
    for f in files:
        for line in open(f):
            m = STEP_RE.match(line)
            if m:
                d[int(m.group(1))] = float(m.group(3))
    s = sorted(d)
    return np.array(s), np.array([d[k] for k in s])


def ema(y: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    out = np.empty_like(y)
    acc = y[0]
    for i, v in enumerate(y):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def main() -> None:
    runs = {
        "m2cond_4yr": sorted(glob.glob(str(LOGS / "idiff-m2cond-4yr-1041.out"))),
        "m2cond_1yr": sorted(glob.glob(str(LOGS / "idiff-m2cond-9*.out")))
                      + [str(LOGS / "idiff-m2cond-300k-1014.out")],
        "m2coarse2": [str(LOGS / "idiff-m2coarse2-1022.out"),
                      str(LOGS / "idiff-m2coarse2-1027.out")],
    }
    data = {k: parse_logs(v) for k, v in runs.items()}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # ---- Panel A: loss ----------------------------------------------------
    for key, colour, label in (
        ("m2cond_4yr", C_4YR, "diff, 4 yr ERA5 (500k)"),
        ("m2cond_1yr", C_1YR, "diff, 1 yr ERA5 (300k)"),
    ):
        s, y = data[key]
        ax.plot(s / 1e3, y, color=colour, lw=0.6, alpha=0.22)
        ax.plot(s / 1e3, ema(y), color=colour, lw=2.0, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("training step (thousands)")
    ax.set_ylabel("EDM loss")
    ax.set_title("A. Training loss falls monotonically", fontsize=11, loc="left",
                 color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(True, which="major", color=MUTED, alpha=0.18, lw=0.6)

    # Direct labels at the final value — no number on every point.
    # Offsets are opposite-signed so the two end labels cannot collide where the
    # curves converge (they finish 0.0004 apart).
    for key, colour, dy in (("m2cond_4yr", C_4YR, -11), ("m2cond_1yr", C_1YR, 10)):
        s, y = data[key]
        ax.annotate(f"{y[-1]:.4f}", xy=(s[-1] / 1e3, ema(y)[-1]),
                    xytext=(-2, dy), textcoords="offset points", ha="right",
                    fontsize=8.5, color=colour, va="center")
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # ---- Panel B: what the loss cannot see --------------------------------
    # ALL points measured under IDENTICAL current sampler settings (the 1-yr 100k/200k
    # points were re-scored 2026-08-05; an older untagged cache gave 281 at 1-yr 100k
    # under superseded settings and must not be mixed in).
    steps_4yr = np.array([100, 250, 500])
    leff_4yr = np.array([280.62, 336.74, 374.16])
    steps_1yr = np.array([100, 200, 300])
    leff_1yr = np.array([336.74, 420.93, 481.06])

    bx.plot(steps_1yr, leff_1yr, "s-", color=C_1YR, lw=2.0, ms=7,
            label="diff, 1 yr ERA5")
    bx.plot(steps_4yr, leff_4yr, "o-", color=C_4YR, lw=2.0, ms=7,
            label="diff, 4 yr ERA5")
    bx.axhline(198.08, color=C_CRS, lw=1.6, ls="--",
               label="diff+coarse (300k)")
    bx.axhline(56.12, color=MUTED, lw=1.4, ls=":", label="reference floor (56 km)")

    for x, y in zip(steps_4yr, leff_4yr):
        bx.annotate(f"{y:.0f}", xy=(x, y), xytext=(0, -15), textcoords="offset points",
                    ha="center", fontsize=8.5, color=C_4YR)
    for x, y in zip(steps_1yr, leff_1yr):
        bx.annotate(f"{y:.0f}", xy=(x, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color=C_1YR)

    bx.set_xlabel("training step (thousands)")
    bx.set_ylabel("$L_{eff}$  (km, lower is better)")
    bx.set_title("B. Effective resolution gets WORSE — in both runs", fontsize=11,
                 loc="left", color=INK, pad=10)
    bx.set_ylim(0, 560)
    bx.legend(frameon=False, fontsize=8.5, loc="lower right")
    bx.grid(True, which="major", color=MUTED, alpha=0.18, lw=0.6)

    for a in (ax, bx):
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(MUTED)
        a.tick_params(colors=INK, labelsize=9)

    fig.suptitle("The training loss does not predict the benchmark",
                 fontsize=12.5, y=0.99, x=0.008, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"loss_curves.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT/'loss_curves.png'}")

    # ---- m2coarse2 gets its own axes: DIFFERENT TARGET, not comparable ----
    fig2, cx = plt.subplots(figsize=(5.6, 4.0))
    s, y = data["m2coarse2"]
    cx.plot(s / 1e3, y, color=C_CRS, lw=0.6, alpha=0.22)
    cx.plot(s / 1e3, ema(y), color=C_CRS, lw=2.0, label="diff+coarse (residual target)")
    cx.set_yscale("log")
    cx.set_xlabel("training step (thousands)")
    cx.set_ylabel("EDM loss (residual space)")
    cx.set_title("diff+coarse trains a DIFFERENT target", fontsize=11, loc="left",
                 color=INK, pad=10)
    cx.annotate(f"{y[-1]:.4f}", xy=(s[-1] / 1e3, ema(y)[-1]), xytext=(4, 0),
                textcoords="offset points", fontsize=8.5, color=C_CRS, va="center")
    cx.text(0.97, 0.90,
            "Diffuses $(x-U(c))/\\sigma_r$, not $x$.\n"
            "Its loss is NOT comparable to panel A.",
            transform=cx.transAxes, fontsize=8.5, color=MUTED,
            va="top", ha="right")
    cx.grid(True, which="major", color=MUTED, alpha=0.18, lw=0.6)
    for side in ("top", "right"):
        cx.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        cx.spines[side].set_color(MUTED)
    cx.tick_params(colors=INK, labelsize=9)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(OUT / f"loss_coarse.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT/'loss_coarse.png'}")


if __name__ == "__main__":
    main()
