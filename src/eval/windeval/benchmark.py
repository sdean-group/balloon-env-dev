"""Benchmark v2 — the one runner: every generator × the reference-based metric suite.

Replaces the five per-phase benchmark scripts (m0/m1/stage2/ble/temporal — git history
has them). Design + calibration findings: docs/benchmark-v2-changes.md; metric
definitions: metrics/suite.py METRIC_INFO. All values are RAW (no 0–1 scores).

Reference: the held-out ERA5 slice (days 8–14, zero overlap with training dates) —
subsampled to 4-hourly for the spatial/distribution metrics, full hourly (segment-aware)
for the temporal ones. The `self-split floor` row scores one disjoint half of the held-out
set against the other: the same-distribution sampling-noise level every other row should
be read against.

Rows (distilled set, Shaurya 2026-07-10 — each earns its place; see changes doc):
  self-split floor      the scale for every raw metric (what "perfect" reads as, given
                        sampling noise); days 8–10 vs 11–14, the only row not vs full ref
  white noise           trivial lower anchor for the spectral metrics (NOTE: at-floor on
                        marginal W1 *by construction* — its documented blind spot)
  phase shuffle         white noise's complement: right stats, zero structure — fires the
                        distribution metrics and brackets the diffusion failure mode
  ble_vae               prior state of the art; the row to beat (SF-box climate caveat)
  idiff trained         the model (--ckpt)
  idiff m2cond          the conditional space-time model (--cond-ckpt), sampled at
                        held-out (month, day, hour) conditions over the center window;
                        the only row with the real (location, month, hour) W1_cond
  --temporal adds:      kinematic toy (mean-wind advection = the no-learned-dynamics
                        floor) + M2/M3 rows when their checkpoints exist
Dropped from the board (calibration lives on in test_metrics_v2.py regardless):
  idiff toy (machinery validation is done; tiling penalty checks the real model now),
  time shuffled (SR_time anchor — asserted in the test suite).

Tiling penalty (the procedural check): the suite runs on a single-tile (64²) and a
multi-tile (192²) generation of the same generator; penalty = multi − single per metric.

Run (full pixi env):
    PYTHONPATH=. .pixi/envs/default/bin/python -m src.eval.windeval.benchmark \
        --ckpt runs/idiff_m1/step_84000.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import artifact
from .reference import build_heldout, split
from .anchors import _phase_randomize
from .metrics import run_suite, tiling_penalty, climatological_dz
from .metrics.distributions import conditional_w1_grouped
from .metrics.suite import METRIC_INFO

DATA = Path(__file__).resolve().parent / "data"
FIGDIR = Path(__file__).resolve().parents[3] / "docs" / "figures" / "benchmark_v2"
SPATIAL_STRIDE_H = 4          # subsample the hourly reference for spatial/dist metrics

# fixed entity -> color, used across every figure (color follows the entity)
COLORS = {
    "era5 (ref)": "#3d3d3c", "self-split floor": "#8a8a85",
    "idiff trained": "#2a78d6", "idiff toy": "#1baf7a", "ble_vae": "#eda100",
    "phase shuffle": "#4a3aa7", "white noise": "#e34948",
    "kinematic toy": "#e87ba4", "time shuffled": "#eb6834",
    "idiff m2cond": "#0ea5b7",
}

# W1_cond protocol (Shaurya 2026-07-14): a condition = (the fixed 64² center window,
# month, hour-of-day). Reference pool = that hour on each held-out day 8–14 (the time
# harmonics are bandlimited below day-scale, so day-to-day variability IS the
# within-condition distribution); model pool = COND_SEEDS draws at those same timestamps.
COND_MONTHS = (1, 4, 7, 10)
COND_HOURS = (0, 12)
COND_DAYS = tuple(range(8, 15))
COND_SEEDS = 2


def _like(ds, u, v):
    return artifact.make_field(u.astype("float32"), v.astype("float32"),
                               level=ds["level"].values, lat=ds["lat"].values,
                               lon=ds["lon"].values, time=ds["time"].values)


def _anchor_rows(half_a) -> dict:
    """Known-broken fields derived from held-out half A (spatially subsampled)."""
    rng = np.random.default_rng(0)
    u, v = half_a["u"].values.copy(), half_a["v"].values.copy()
    for t in range(u.shape[0]):
        for l in range(u.shape[1]):
            u[t, l] = _phase_randomize(u[t, l], rng)
            v[t, l] = _phase_randomize(v[t, l], rng)
    ps = _like(half_a, u, v)

    au, av = half_a["u"].values, half_a["v"].values
    nu = (au.mean(axis=(2, 3), keepdims=True)
          + au.std(axis=(2, 3), keepdims=True) * rng.standard_normal(au.shape))
    nv = (av.mean(axis=(2, 3), keepdims=True)
          + av.std(axis=(2, 3), keepdims=True) * rng.standard_normal(av.shape))
    noise = _like(half_a, nu, nv)
    return {"phase shuffle": ps, "white noise": noise}


def _pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _trained_artifacts(ckpt: str, *, regen: bool, num_steps: int = 18):
    """(multi-tile 192² ds, single-tile 64² ds, seed-crop datasets) for the trained model."""
    from .generators.infinite_diffusion import InfiniteDiffusionGenerator
    from .generators.infinite_diffusion.trained import TrainedWindowDenoiser
    from .generators.infinite_diffusion.gate import crops_to_dataset

    device = _pick_device()
    phi = TrainedWindowDenoiser(ckpt, num_steps=num_steps, device=device)
    gen = InfiniteDiffusionGenerator(
        denoiser=phi, levels=phi.stats.levels.astype(float), window=64, stride=32,
        T=1, seed=0, name="infinite_diffusion_trained", device=device)

    multi = DATA / "infinite_diffusion_trained.zarr"
    if regen or not multi.exists():
        gen.to_artifact(multi, height=192, width=192, n_queries=8)
    single = DATA / "infinite_diffusion_trained_64.zarr"
    if regen or not single.exists():
        gen.to_artifact(single, height=64, width=64, n_queries=8)

    # N standalone seed-crops for the conditional metric (one condition; see changes doc)
    crops = phi.sample_crops(8, 64, seed=1).cpu().numpy()
    seeds = [crops_to_dataset(crops[i:i + 1], phi.stats.levels) for i in range(len(crops))]
    return artifact.read(multi), artifact.read(single), seeds


def _cond_window(ref):
    """The fixed 64² window centered on the reference grid (single-location protocol)."""
    y0 = (ref.sizes["y"] - 64) // 2
    x0 = (ref.sizes["x"] - 64) // 2
    return y0, x0, ref["lat"].values[y0:y0 + 64], ref["lon"].values[x0:x0 + 64]


def _cond_ref(ref, y0, x0, month, hour, days):
    """Reference fields at one (window, month, hour) condition, for the given days."""
    t = ref["time"]
    sel = ((t.dt.month == month) & (t.dt.hour == hour) & t.dt.day.isin(list(days)))
    return ref.sel(time=sel).isel(y=slice(y0, y0 + 64), x=slice(x0, x0 + 64)).compute()


def _cond_floor_groups(ref):
    """Condition-matched W1_cond floor: days 8–10 vs 11–14 at each (month, hour)."""
    from .reference import SPLIT_DAY
    y0, x0, _, _ = _cond_window(ref)
    lo = [d for d in COND_DAYS if d < SPLIT_DAY]
    hi = [d for d in COND_DAYS if d >= SPLIT_DAY]
    return [( [_cond_ref(ref, y0, x0, m, h, lo)], _cond_ref(ref, y0, x0, m, h, hi) )
            for m in COND_MONTHS for h in COND_HOURS]


def _conditional_artifacts(ckpt: str, ref, *, regen: bool, num_steps: int = 18):
    """(pooled seed-0 block Dataset, W1_cond condition groups) for the conditional model.

    Samples one τ-frame block per (month, day, hour, seed) over the fixed center window.
    The pooled dataset keeps real timestamps, so the temporal metrics run on τ-frame
    segments — 3 lead hours is all a τ=4 block supports (see report caveat).
    """
    from .generators.infinite_diffusion.spacetime import SpaceTimeSampler

    y0, x0, lat, lon = _cond_window(ref)
    conds = [(m, d, h) for m in COND_MONTHS for h in COND_HOURS for d in COND_DAYS]
    cache = DATA / "idiff_m2cond_blocks.npz"
    if regen or not cache.exists():
        sampler = SpaceTimeSampler(ckpt, num_steps=num_steps, device=_pick_device())
        blocks, times = [], []
        for i, (m, d, h) in enumerate(conds):
            ts = (np.datetime64(f"2023-{m:02d}-{d:02d}T{h:02d}", "h")
                  + np.arange(sampler.tau).astype("timedelta64[h]"))
            for s in range(COND_SEEDS):
                print(f"[bench] m2cond block {i * COND_SEEDS + s + 1}/"
                      f"{len(conds) * COND_SEEDS} (2023-{m:02d}-{d:02d} {h:02d}h "
                      f"seed {s}) …", flush=True)
                us, vs = sampler.sample_block((64, 64), seed=i * COND_SEEDS + s,
                                              lat=lat, lon=lon, times=ts)
                blocks.append(np.stack([us, vs], axis=2))     # (τ, L, 2, H, W)
                times.append(ts)
        np.savez(cache, blocks=np.asarray(blocks, dtype="float32"),
                 times=np.asarray(times),
                 month=np.repeat([c[0] for c in conds], COND_SEEDS),
                 day=np.repeat([c[1] for c in conds], COND_SEEDS),
                 hour=np.repeat([c[2] for c in conds], COND_SEEDS),
                 seed_idx=np.tile(np.arange(COND_SEEDS), len(conds)),
                 levels=sampler.stats.levels)
    z = np.load(cache)
    blocks, times, levels = z["blocks"], z["times"], z["levels"]

    # main row: seed-0 blocks pooled with their real (sorted) timestamps
    s0 = z["seed_idx"] == 0
    u = blocks[s0][:, :, :, 0].reshape(-1, len(levels), 64, 64)
    v = blocks[s0][:, :, :, 1].reshape(-1, len(levels), 64, 64)
    t = times[s0].ravel()
    order = np.argsort(t)
    pooled = artifact.make_field(u[order], v[order], level=levels,
                                 lat=lat, lon=lon, time=t[order])

    # W1_cond groups: frame 0 (the exact condition hour) of every block at (month, hour)
    groups = []
    for m in COND_MONTHS:
        for h in COND_HOURS:
            at = (z["month"] == m) & (z["hour"] == h)
            seeds = [artifact.make_field(b[0, :, 0], b[0, :, 1], level=levels,
                                         lat=lat, lon=lon) for b in blocks[at]]
            groups.append((seeds, _cond_ref(ref, y0, x0, m, h, COND_DAYS)))
    return pooled, groups


def _kinematic_toy(ckpt: str, *, regen: bool, n_times: int = 48):
    """Trained static field + mean-wind advection: the deliberately-naive temporal floor."""
    path = DATA / "temporal_kinematic_toy.zarr"
    if not regen and path.exists():
        ds = artifact.read(path)
        if ds.sizes["time"] >= 16:
            return ds
    from .generators.infinite_diffusion import InfiniteDiffusionGenerator
    from .generators.infinite_diffusion.trained import TrainedWindowDenoiser
    from .generators.infinite_diffusion.advected import velocity_from_stats
    device = _pick_device()
    phi = TrainedWindowDenoiser(ckpt, num_steps=18, device=device)
    gen = InfiniteDiffusionGenerator(
        denoiser=phi, levels=phi.stats.levels.astype(float), window=64, stride=32,
        T=1, seed=0, name="kinematic_toy", device=device)
    gen.to_artifact(path, height=64, width=64, n_queries=8, n_times=n_times,
                    dt_seconds=3600.0, advect_vel=velocity_from_stats(phi.stats))
    return artifact.read(path)


# ---------- report ----------

def _fmt(v, spec="{:.2f}"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    return spec.format(v)


def _table(rows: dict[str, dict], metrics: list[str]) -> list[str]:
    cols = list(rows)
    L = ["| Metric (see METRIC_INFO) | " + " | ".join(cols) + " |",
         "|" + "---|" * (len(cols) + 1)]
    for m in metrics:
        better, _ = METRIC_INFO[m]
        cells = []
        for c in cols:
            s = _fmt(rows[c].get(m))
            if m == "L_eff (km)" and rows[c].get("resolved_to_grid"):
                s += "†"
            cells.append(s)
        L.append(f"| {m} ({better}) | " + " | ".join(cells) + " |")
    return L


def _figures(details: dict[str, dict], datasets: dict[str, object], ref_sp) -> list[str]:
    """Report figures. Fixed entity colors; 2px lines; recessive grid; log axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGDIR.mkdir(parents=True, exist_ok=True)
    made = []
    style = dict(lw=2)

    def ax_setup(ax):
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    # 1. spatial PSD triptych (E, div, vort)
    with_spec = {n: d for n, d in details.items() if "spectra" in d}
    if with_spec:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        any_name = next(iter(with_spec))
        ref_spec = with_spec[any_name]["spectra"]["ref"]
        for ax, comp, title in zip(axes, ("E", "div", "vort"),
                                   ("kinetic energy", "divergence", "vorticity")):
            ax.loglog(ref_spec["k"] * 1e3, ref_spec[comp], color=COLORS["era5 (ref)"],
                      label="era5 (ref)", **style)
            for name, d in with_spec.items():
                sp = d["spectra"]["pred"]
                ax.loglog(sp["k"] * 1e3, sp[comp], color=COLORS.get(name, "#999"),
                          label=name, **style)
            ax.set_xlabel("k (cycles/km)")
            ax.set_title(f"PSD of {title}")
            ax_setup(ax)
        axes[0].set_ylabel("PSD (density)")
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        p = FIGDIR / "psd_triptych.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p.name)

    # 2. marginal log-density of u at a mid level
    fig, ax = plt.subplots(figsize=(6, 4))
    lv = ref_sp.sizes["level"] // 2
    bins = np.linspace(-60, 80, 141)
    ax.semilogy(*_hist(ref_sp["u"].values[:, lv].ravel(), bins),
                color=COLORS["era5 (ref)"], label="era5 (ref)", **style)
    for name, ds in datasets.items():
        if name in ("time shuffled",):
            continue                       # identical marginals to the real frames
        l = min(lv, ds.sizes["level"] - 1)
        ax.semilogy(*_hist(ds["u"].values[:, l].ravel(), bins),
                    color=COLORS.get(name, "#999"), label=name, **style)
    ax.set_xlabel("u (m/s)")
    ax.set_ylabel("density (log)")
    ax.set_title(f"marginal of u, mid level (log tails)")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    p = FIGDIR / "marginal_u.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p.name)

    # 3. temporal PSD + 4. MSD curves (rows that have them)
    with_t = {n: d for n, d in details.items() if "temporal_psd" in d}
    if with_t:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        any_name = next(iter(with_t))
        rr = with_t[any_name]["temporal_psd"]["ref"]
        ax1.loglog(rr["f"], rr["P"], color=COLORS["era5 (ref)"], label="era5 (ref)", **style)
        dr = with_t[any_name]["dispersion"]["ref"]
        ax2.loglog(dr["hours"][1:], dr["msd"][1:].mean(axis=1),
                   color=COLORS["era5 (ref)"], label="era5 (ref)", **style)
        for name, d in with_t.items():
            pp = d["temporal_psd"]["pred"]
            ax1.loglog(pp["f"], pp["P"], color=COLORS.get(name, "#999"), label=name, **style)
            dp = d["dispersion"]["pred"]
            ax2.loglog(dp["hours"][1:], dp["msd"][1:].mean(axis=1),
                       color=COLORS.get(name, "#999"), label=name, **style)
        ax1.set_xlabel("f (cycles/hour)"); ax1.set_ylabel("PSD (density)")
        ax1.set_title("temporal power spectrum")
        ax2.set_xlabel("lead time (h)"); ax2.set_ylabel("MSD (m²)")
        ax2.set_title("tracer dispersion (levels avg)")
        for ax in (ax1, ax2):
            ax.grid(True, alpha=0.25, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        p = FIGDIR / "temporal.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p.name)
    return made


def _hist(x, bins):
    h, e = np.histogram(x, bins=bins, density=True)
    return 0.5 * (e[:-1] + e[1:]), np.where(h > 0, h, np.nan)


SPATIAL_METRICS = ["SR_E", "SR_div", "SR_vort", "L_eff (km)",
                   "W1 shear u ((m/s)/km)", "W1 shear v ((m/s)/km)"]
DIST_METRICS = ["W1 u (m/s)", "W1 v (m/s)", "tail err 1% (m/s)", "tail err 0.1% (m/s)",
                "W1 cond (m/s)"]
TEMPORAL_METRICS = ["SR_time", "disp log-MSD RMSE", "final spread ratio"]


def run(ckpt: str | None, *, generate: bool = True, regen: bool = False,
        temporal: bool = False, cond_ckpt: str | None = None) -> str:
    DATA.mkdir(exist_ok=True)
    ref = artifact.read(build_heldout())
    ref_sp = ref.isel(time=slice(0, None, SPATIAL_STRIDE_H)).compute()
    half_a, half_b = split(ref)
    a_sp = half_a.isel(time=slice(0, None, SPATIAL_STRIDE_H)).compute()
    b_sp = half_b.isel(time=slice(0, None, SPATIAL_STRIDE_H)).compute()
    dz = climatological_dz(DATA / "era5_real_stage2.zarr")

    scores: dict[str, dict] = {}
    details: dict[str, dict] = {}
    datasets: dict[str, object] = {}

    def score(name, pred, *, ref_s=None, ref_t=None, seeds=None):
        print(f"[bench] scoring {name} …", flush=True)
        s, d = run_suite(pred, ref_sp if ref_s is None else ref_s, dz=dz,
                         seed_datasets=seeds,
                         ref_temporal=ref if ref_t is None else ref_t)
        scores[name], details[name] = s, d
        datasets[name] = pred

    # floor + anchors (floor is half-vs-half; anchors score vs the full ref)
    score("self-split floor", a_sp, ref_s=b_sp, ref_t=half_b)
    # condition-matched W1_cond floor: days 8–10 vs 11–14 at each (month, hour)
    scores["self-split floor"].update(conditional_w1_grouped(_cond_floor_groups(ref)))
    for name, ds in _anchor_rows(a_sp).items():
        score(name, ds)

    # bounded baseline (prior state of the art)
    ble = DATA / "ble_vae_0.zarr"
    if ble.exists():
        score("ble_vae", artifact.read(ble))

    # the model
    tiling: dict[str, dict] = {}
    if generate and ckpt and Path(ckpt).exists():
        multi, single, seeds = _trained_artifacts(ckpt, regen=regen)
        score("idiff trained", multi, seeds=seeds)
        s_single, _ = run_suite(single, ref_sp, dz=dz)
        tiling["idiff trained"] = tiling_penalty(s_single, scores["idiff trained"])
        if temporal:
            score("kinematic toy", _kinematic_toy(ckpt, regen=regen))
    elif generate:
        print(f"[bench] no checkpoint at {ckpt} — skipping trained rows")

    # the conditional model (location+time), sampled at held-out conditions.
    # No tiling-penalty row yet: multi-tile needs per-tile coords in the
    # InfiniteDiffusion wrapper — deferred until single-block results are read.
    if generate and cond_ckpt and Path(cond_ckpt).exists():
        pooled, groups = _conditional_artifacts(cond_ckpt, ref, regen=regen)
        score("idiff m2cond", pooled)
        scores["idiff m2cond"].update(conditional_w1_grouped(groups))
    elif generate and cond_ckpt:
        print(f"[bench] no conditional checkpoint at {cond_ckpt} — skipping m2cond row")

    figs = _figures(details, datasets, ref_sp)

    # ---- report ----
    L = [
        "# Benchmark v2 — reference-based metric suite", "",
        "Raw values only (units per metric; direction in parentheses). Reference = held-out "
        "ERA5, days 8–14 of Jan/Apr/Jul/Oct 2023 (zero overlap with training dates); "
        "`self-split floor` = one disjoint half of the held-out set vs the other — read every "
        "row against it. N/A = metric not applicable (missing capability/levels), not a "
        "failure. Design + calibration: `docs/benchmark-v2-changes.md`.", "",
        "## Physical consistency — spatial", "",
        *_table(scores, SPATIAL_METRICS), "",
        "† = never dropped below the 0.5 energy-ratio threshold: resolved over the whole "
        "compared range (the value shown is the finest compared wavelength).", "",
        "## Data distribution", "",
        *_table(scores, DIST_METRICS), "",
        "`W1 cond` protocol: a condition = (fixed 64² center window, month, hour-of-day); "
        "reference pool = that hour on each held-out day 8–14, model pool = seeds at those "
        "same timestamps, averaged over the 8 conditions. The floor row uses the "
        "condition-matched split (days 8–10 vs 11–14). Unconditional rows (`idiff trained`) "
        "still report the degenerate one-condition version — not comparable across those "
        "two protocols. `idiff m2cond` rows: temporal metrics are N/A (τ=4-frame blocks "
        "are below the suite's 16-frame segment minimum — temporal tiling unlocks them) "
        "and its spatial/dist rows are the 64² center window vs the full reference; no "
        "tiling penalty yet (per-tile coords in the wrapper are deferred).", "",
        "## Physical consistency — temporal", "",
        *_table(scores, TEMPORAL_METRICS), "",
        "Caveats: `ble_vae` is the SF box at 0.45° with 10 arbitrary levels — its "
        "distribution rows partly measure climate mismatch, and level-indexed comparisons "
        "pair the first 10 reference levels. `white noise` is at-floor on marginal W1 by "
        "construction (moment-matched) — read it on the spectral rows; `phase shuffle` "
        "covers the distribution rows.", "",
    ]
    if tiling:
        L += ["## Tiling penalty (multi-tile − single-tile; 0 = seamless)", ""]
        tmetrics = sorted({m for t in tiling.values() for m in t})
        cols = list(tiling)
        L += ["| Metric | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
        for m in tmetrics:
            L.append(f"| {m} | " + " | ".join(_fmt(tiling[c].get(m), "{:+.3f}")
                                              for c in cols) + " |")
        L.append("")
    if figs:
        L += ["## Figures", ""] + [f"![{f}](../../../docs/figures/benchmark_v2/{f})"
                                   for f in figs] + [""]

    report = "\n".join(L)
    out = DATA.parent / "benchmark_v2_report.md"
    out.write_text(report)
    print(report)
    print(f"\nreport: {out}\nfigures: {FIGDIR}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark v2 — unified leaderboard.")
    ap.add_argument("--ckpt", default="runs/idiff_m1/step_84000.pt")
    ap.add_argument("--cond-ckpt", default="runs/idiff_m2cond/latest.pt",
                    help="conditional spacetime checkpoint (adds the idiff m2cond row; "
                         "skipped with a message if the file is missing)")
    ap.add_argument("--no-generate", action="store_true",
                    help="skip rows that need torch generation (anchors/ble only)")
    ap.add_argument("--regen", action="store_true",
                    help="regenerate cached generator artifacts")
    ap.add_argument("--temporal", action="store_true",
                    help="add the temporal rows (kinematic toy; M2/M3 when their ckpts land)")
    args = ap.parse_args(argv)
    run(args.ckpt, generate=not args.no_generate, regen=args.regen,
        temporal=args.temporal, cond_ckpt=args.cond_ckpt)


if __name__ == "__main__":
    main()
