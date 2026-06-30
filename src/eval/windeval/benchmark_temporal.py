"""Temporal leaderboard — does the wind field EVOLVE like real weather? (Phase 4d)

The spatial board (`benchmark_m1`) scores a single frame; this one scores *time*. It places
the temporal generators side by side and lets the numbers settle the M2-vs-M3 head-to-head:

  - **ERA5 (peer)**         — the ceiling: real contiguous reanalysis.
  - **kinematic-toy (M1)**  — the deliberately-naive floor (advect by the mean wind).
  - **M3 autoregressive**   — roll p(frame_{t+1}|frame_t) forward from an ERA5 seed.
  - **M2 joint spacetime**  — denoise an H×W×τ block jointly.
  - **shuffled (anchor)**   — ERA5 frames in random time order: the incoherent lower bound.

Scored on **`score: temporal realism`** (peer-matched: persistence-match + tendency-match — real
evolution lives in a band, so too-frozen AND too-chaotic are both penalised) and **drift** (spatial
COMPOSITE vs lead time; on a generator it should stay ~flat, its expected failure is decay). The
raw persistence / tendency / structure-advection diagnostics are reported too.

Cadence: all rows are compared at a common timestep (``--dt-hours``, default = the peer's), since
tendency (m/s per step) is cadence-dependent. M2 is scored on its native τ-frame block length.

Run (full pixi env):
    PYTHONPATH=. .pixi/envs/default/bin/python -m src.eval.windeval.benchmark_temporal \
        [--m3-ckpt runs/idiff_m3/latest.pt] [--m2-ckpt runs/idiff_m2/latest.pt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from . import artifact, anchors
from .metrics import field_scores
from .metrics.temporal import temporal_scores, temporal_persistence, tendency_mag, drift
from .benchmark_m0 import _fmt

DATA = Path(__file__).resolve().parent / "data"
SECONDS_PER_HOUR = 3600.0


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _datetime_axis(n: int, dt_hours: float) -> np.ndarray:
    return np.datetime64("2023-01-01T00") + np.arange(n) * np.timedelta64(
        int(dt_hours * SECONDS_PER_HOUR), "s")


def _shuffled_anchor(peer: xr.Dataset, *, seed: int = 0) -> xr.Dataset:
    """ERA5 frames in random time order — destroys temporal coherence, keeps each frame real."""
    rng = np.random.default_rng(seed)
    u, v = peer["u"].values, peer["v"].values
    perm = rng.permutation(u.shape[0])
    return artifact.make_field(u[perm], v[perm], level=peer["level"].values,
                               lat=peer["lat"].values, lon=peer["lon"].values,
                               time=peer["time"].values)


def _toy_temporal(*, ckpt: str, device: str, n_times: int, dt_hours: float,
                  crop: int) -> xr.Dataset:
    """The kinematic advective floor: static field carried by the per-level mean wind."""
    from .generators.infinite_diffusion import InfiniteDiffusionGenerator
    from .generators.infinite_diffusion.trained import TrainedWindowDenoiser
    from .generators.infinite_diffusion.advected import velocity_from_stats

    phi = TrainedWindowDenoiser(ckpt, num_steps=18, device=device)
    gen = InfiniteDiffusionGenerator(denoiser=phi, levels=phi.stats.levels.astype(float),
                                     window=64, stride=32, T=1, seed=0,
                                     name="kinematic_toy", device=device)
    out = DATA / "temporal_kinematic_toy.zarr"
    gen.to_artifact(out, height=crop, width=crop, n_queries=8, n_times=n_times,
                    dt_seconds=dt_hours * SECONDS_PER_HOUR,
                    advect_vel=velocity_from_stats(phi.stats))
    return artifact.read(out)


def _seed_frame(seed_src: xr.Dataset, levels: np.ndarray, crop: int):
    """A real (u0,v0) seed crop from ``seed_src`` on the model's levels — drives M3's t=0."""
    lv = seed_src["level"].values
    idx = np.array([int(np.where(lv == L)[0][0]) for L in levels])
    u = seed_src["u"].values[0, idx]            # (L, Y, X)
    v = seed_src["v"].values[0, idx]
    Y, X = u.shape[1:]
    c = min(crop, Y, X)
    return u[:, :c, :c].astype(np.float32), v[:, :c, :c].astype(np.float32)


def _m3_temporal(*, ckpt: str, device: str, seed_src: xr.Dataset, n_times: int,
                 dt_hours: float, crop: int) -> xr.Dataset:
    """M3 rollout from an ERA5 seed frame (spatial realism from the seed; M3 owns the transition)."""
    from .generators.infinite_diffusion.autoregressive import ConditionedDenoiser
    cd = ConditionedDenoiser(ckpt, num_steps=18, device=device)
    u0, v0 = _seed_frame(seed_src, cd.stats.levels, crop)
    us, vs = cd.rollout((u0, v0), n_times=n_times, seed=0)
    lat = seed_src["lat"].values[:us.shape[2]]
    lon = seed_src["lon"].values[:us.shape[3]]
    return artifact.make_field(us, vs, level=cd.stats.levels, lat=lat, lon=lon,
                               time=_datetime_axis(n_times, dt_hours))


def _m2_temporal(*, ckpt: str, device: str, dt_hours: float, crop: int) -> xr.Dataset:
    """M2 joint-spacetime: one denoised τ-frame block (its native temporal extent)."""
    from .generators.infinite_diffusion.spacetime import SpaceTimeSampler
    samp = SpaceTimeSampler(ckpt, num_steps=18, device=device)
    us, vs = samp.sample_block((crop, crop), seed=0)          # (τ, L, H, W)
    lat = np.arange(crop) * 0.25 + 37.0
    lon = np.arange(crop) * 0.25 + 237.0
    return artifact.make_field(us, vs, level=samp.stats.levels.astype(float),
                               lat=lat, lon=lon, time=_datetime_axis(samp.tau, dt_hours))


def _score(ds, *, ref_persistence, ref_tendency, with_drift=True) -> dict:
    s = temporal_scores(ds, ref_persistence=ref_persistence, ref_tendency=ref_tendency)
    if with_drift and ds.sizes.get("time", 1) > 1:
        s.update(drift(ds, field_scores))
    return s


def run(*, m3_ckpt: str | None = None, m2_ckpt: str | None = None,
        static_ckpt: str = "runs/idiff_m1/step_84000.pt",
        peer_zarr: str | None = None, device: str | None = None,
        n_times: int = 24, dt_hours: float | None = None, crop: int = 32) -> str:
    DATA.mkdir(exist_ok=True)
    device = device or _pick_device()
    peer_path = Path(peer_zarr) if peer_zarr else (DATA / "era5_real.zarr")
    peer = artifact.read(peer_path)
    if dt_hours is None:
        t = peer["time"].values
        dt_hours = float((t[1] - t[0]) / np.timedelta64(1, "h")) if len(t) > 1 else 1.0
    print(f"[temporal] peer={peer_path.name} dt={dt_hours}h n_times={n_times} crop={crop} "
          f"device={device}")

    ref_p = temporal_persistence(peer)
    ref_t = tendency_mag(peer)

    scores: dict[str, dict] = {}
    scores["era5 (peer)"] = _score(peer, ref_persistence=ref_p, ref_tendency=ref_t)
    scores["shuffled (anchor)"] = _score(_shuffled_anchor(peer), ref_persistence=ref_p,
                                         ref_tendency=ref_t)
    if Path(static_ckpt).exists():
        scores["kinematic-toy (M1)"] = _score(
            _toy_temporal(ckpt=static_ckpt, device=device, n_times=n_times, dt_hours=dt_hours,
                          crop=crop), ref_persistence=ref_p, ref_tendency=ref_t)
    if m3_ckpt and Path(m3_ckpt).exists():
        scores["M3 autoregressive"] = _score(
            _m3_temporal(ckpt=m3_ckpt, device=device, seed_src=peer, n_times=n_times,
                         dt_hours=dt_hours, crop=crop), ref_persistence=ref_p, ref_tendency=ref_t)
    if m2_ckpt and Path(m2_ckpt).exists():
        scores["M2 spacetime"] = _score(
            _m2_temporal(ckpt=m2_ckpt, device=device, dt_hours=dt_hours, crop=crop),
            ref_persistence=ref_p, ref_tendency=ref_t)

    cols = [c for c in ["era5 (peer)", "M3 autoregressive", "M2 spacetime",
                        "kinematic-toy (M1)", "shuffled (anchor)"] if c in scores]

    def row(label, key, fmt="{:.3f}", bold=False):
        def cell(c):
            s = _fmt(scores[c].get(key, np.nan), fmt)
            return f"**{s}**" if (bold and s != "N/A") else s
        return f"| {label} | " + " | ".join(cell(c) for c in cols) + " |"

    L = [
        "# Temporal Leaderboard — does the wind field evolve like weather? (Phase 4d)", "",
        "Peer-matched realism (real evolution lives in a band — too-frozen AND too-chaotic are "
        "both wrong): `temporal realism` = mean(persistence-match, tendency-match). `drift` = "
        "spatial COMPOSITE vs lead time (a generator should stay ~flat). Higher = better; "
        "**N/A** = needs >1 frame / a reference.", "",
        f"Peer `{peer_path.name}` · dt {dt_hours}h · {n_times} frames · crop {crop} · device "
        f"`{device}`. Peer refs: persistence {ref_p:.3f}, tendency {ref_t:.3f} m/s/step.", "",
        "## Headline", "",
        "| Metric | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
        row("score: temporal realism", "score: temporal realism", bold=True),
        row("  └ persistence match", "score: persistence match"),
        row("  └ tendency match", "score: tendency match"),
        row("drift slope/step (→0 = flat)", "drift slope/step", fmt="{:+.4f}"),
        row("drift mean (spatial COMPOSITE)", "drift mean"),
        "",
        "## Diagnostics (not realism scores)", "",
        "| Metric | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
        row("temporal persistence (consec. corr)", "temporal persistence"),
        row("tendency (m/s/step)", "tendency (m/s/h)", fmt="{:.3f}"),
        row("structure advection (diag)", "structure advection (diag)"),
        "",
        "## Verdict (automated checks)",
    ]
    for ok, msg in _verdict(scores):
        L.append(f"- {'✅' if ok else '❌'} {msg}")

    report = "\n".join(L)
    out = DATA.parent / "benchmark_temporal_report.md"
    out.write_text(report)
    print(report)
    print(f"\nreport: {out}")
    return report


def _verdict(s):
    """Sanity checks that hold regardless of which learned rows are present."""
    c = []
    peer = s["era5 (peer)"]
    sh = s["shuffled (anchor)"]
    c.append((peer["score: temporal realism"] > sh["score: temporal realism"],
              f"peer realism ({peer['score: temporal realism']:.2f}) > shuffled "
              f"({sh['score: temporal realism']:.2f}) — the metric ranks coherent over incoherent"))
    c.append((peer["temporal persistence"] > 0.9,
              f"peer is temporally coherent (persistence {peer['temporal persistence']:.2f})"))
    c.append((sh["temporal persistence"] < 0.5,
              f"shuffled anchor is incoherent (persistence {sh['temporal persistence']:.2f})"))
    if "kinematic-toy (M1)" in s:
        toy = s["kinematic-toy (M1)"]
        c.append((toy["score: tendency match"] < 0.6,
                  f"kinematic toy flagged TOO FROZEN by tendency-match "
                  f"({toy['score: tendency match']:.2f}) — the naive floor, as designed"))
    for route in ("M3 autoregressive", "M2 spacetime"):
        if route in s:
            r = s[route]
            c.append((r["score: temporal realism"] > s["kinematic-toy (M1)"]["score: temporal realism"]
                      if "kinematic-toy (M1)" in s else r["score: temporal realism"] > 0,
                      f"{route} realism {r['score: temporal realism']:.2f} beats the kinematic toy "
                      "(learned dynamics > naive advection)"))
    return c


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Temporal leaderboard — M2 vs M3 vs toy vs ERA5.")
    ap.add_argument("--m3-ckpt", default=None, help="paired (autoregressive) checkpoint")
    ap.add_argument("--m2-ckpt", default=None, help="spacetime checkpoint")
    ap.add_argument("--static-ckpt", default="runs/idiff_m1/step_84000.pt",
                    help="static checkpoint driving the kinematic toy")
    ap.add_argument("--peer", default=None, help="ERA5 peer zarr (default: era5_real.zarr)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-times", type=int, default=24)
    ap.add_argument("--dt-hours", type=float, default=None, help="common cadence (default: peer's)")
    ap.add_argument("--crop", type=int, default=32)
    args = ap.parse_args(argv)
    run(m3_ckpt=args.m3_ckpt, m2_ckpt=args.m2_ckpt, static_ckpt=args.static_ckpt,
        peer_zarr=args.peer, device=args.device, n_times=args.n_times,
        dt_hours=args.dt_hours, crop=args.crop)


if __name__ == "__main__":
    main()
