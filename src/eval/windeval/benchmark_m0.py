"""M0 leaderboard — the toy InfiniteDiffusion generator on BOTH axes.

Closes Phase 1 ("M0"): an *infinite* generator running through the benchmark. Scores it on
Axis-1 (field quality, reference-free, grid-independent) and Axis-2 (the procedure claims)
on one board, alongside the existing peers/anchors:

  era5_real (ceiling) · ble_vae (bar to beat) · infinite_diffusion_toy (new) ·
  phase_shuffle · white_noise (lower bounds)

The story M0 must show: the toy passes **Axis-2** (seamless / seed-consistent / O(1) /
no extent drift — the *machinery* claims), while sitting mid-pack on **Axis-1** — exactly
right, because field quality is the *denoiser's* job, which is Phase 2. The two halves
(machinery vs model) are scored independently, as designed. Bounded peers are Axis-2 N/A.

Run:  PYTHONPATH=. .pixi/envs/default/bin/python -m src.eval.windeval.benchmark_m0
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import artifact, anchors
from .ingest_era5 import ingest
from .metrics import field_scores, procedure_scores

DATA = Path(__file__).resolve().parent / "data"


def _fmt(x, spec="{:.3f}"):
    return "N/A" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else spec.format(x)


def _toy_artifact_and_scores():
    """Materialize the toy InfiniteDiffusion artifact + score both axes."""
    from .generators.infinite_diffusion import InfiniteDiffusionGenerator

    gen = InfiniteDiffusionGenerator(n_levels=10, window=64, stride=32, T=2, seed=0)
    path = DATA / "infinite_diffusion_toy.zarr"
    gen.to_artifact(path, height=192, width=192, n_queries=48)
    ds = artifact.read(path)
    q = artifact.read_querylog(path)

    # extent family: growing crops from the same seed (Axis-2 extent drift)
    family = []
    for sz in (64, 96, 128, 192):
        u, v = gen.sampler.field_uv(0, sz, 0, sz)
        lat, lon = gen._coords(0, sz, 0, sz)
        family.append(artifact.make_field(u, v, level=gen.levels, lat=lat, lon=lon,
                                           time=np.array([0])))

    a1 = field_scores(ds)
    a2 = procedure_scores(ds, querylog=q, extent_family=family)
    return {**a1, **a2}, path


def run() -> str:
    DATA.mkdir(exist_ok=True)

    # peers + anchors (Axis-1 only; no procedure capabilities -> Axis-2 N/A)
    real_z = ingest(DATA / "era5_sf_uvtq.grib", DATA / "era5_real_stage2.zarr",
                    lnsp_path=DATA / "era5_sf_lnsp.grib")
    ps_z = anchors.phase_shuffle(real_z, DATA / "anchor_ps_stage2.zarr", seed=0)
    no_z = anchors.white_noise(real_z, DATA / "anchor_noise_stage2.zarr", seed=0)

    scores: dict[str, dict] = {}
    for name, z in [("era5_real (peer)", real_z), ("phase_shuffle", ps_z),
                    ("white_noise", no_z)]:
        ds = artifact.read(z)
        scores[name] = {**field_scores(ds), **procedure_scores(ds)}

    # BLE-VAE (bounded; Axis-2 N/A — null control for the procedure axis)
    ble = artifact.read(DATA / "ble_vae_0.zarr")
    scores["ble_vae"] = {**field_scores(ble), **procedure_scores(ble)}

    # the new generator
    scores["infinite_diffusion_toy"], toy_path = _toy_artifact_and_scores()

    cols = ["era5_real (peer)", "ble_vae", "infinite_diffusion_toy",
            "phase_shuffle", "white_noise"]

    def row(label, key, bold=False, fmt="{:.3f}"):
        def cell(c):
            v = scores[c].get(key, np.nan)
            s = _fmt(v, fmt)
            return f"**{s}**" if (bold and s != "N/A") else s
        return f"| {label} | " + " | ".join(cell(c) for c in cols) + " |"

    L = [
        "# M0 Leaderboard — InfiniteDiffusion on both axes", "",
        "An *infinite* generator running through the benchmark. Axis-1 = field quality "
        "(reference-free, grid-independent); Axis-2 = the procedure claims only the "
        "unbounded class makes. Scores ∈ [0,1], higher = better; **N/A** = capability not "
        "declared (not a failure).", "",
        "## Headline — two axes", "",
        "| Axis | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
        row("Axis-1: spatial COMPOSITE", "COMPOSITE", bold=True),
        row("Axis-2: PROC COMPOSITE", "PROC COMPOSITE", bold=True),
        "",
        "## Axis-2 detail — the procedure (only `unbounded`/`tiled`/`random_access` gens)", "",
        "| Procedure metric | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
        row("seam discontinuity (score; 1=seamless)", "score: seam"),
        row("  └ seam |div| excess (1=ideal)", "seam div excess", fmt="{:.2f}"),
        row("revisit determinism (score; 1=exact)", "score: revisit"),
        row("  └ revisit max|Δ|", "revisit max|Δ|", fmt="{:.1e}"),
        row("budget / O(1) (score; far≈near)", "score: budget"),
        row("  └ latency p50 (ms)", "latency p50 (ms)", fmt="{:.1f}"),
        row("  └ budget far/near ratio", "budget far/near", fmt="{:.2f}"),
        row("extent drift (score; 1=flat)", "score: extent"),
        row("  └ drift slope / octave", "extent drift slope/oct", fmt="{:+.3f}"),
        "",
        "## Axis-1 detail", "",
        "| Metric | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
        row("spectrum slope (≈ −3)", "spectrum slope", fmt="{:.2f}"),
        row("increment kurtosis (> 3)", "increment kurtosis", fmt="{:.2f}"),
        row("vort/div ratio (> 1)", "vort/div ratio", fmt="{:.2f}"),
        "",
        "## Verdict (automated checks)",
    ]
    for ok, msg in _verdict(scores):
        L.append(f"- {'✅' if ok else '❌'} {msg}")

    report = "\n".join(L)
    out = DATA.parent / "benchmark_m0_report.md"
    out.write_text(report)
    print(report)
    print(f"\ntoy artifact: {toy_path}\nreport: {out}")
    return report


def _verdict(s):
    toy = s["infinite_diffusion_toy"]
    real, no = s["era5_real (peer)"], s["white_noise"]
    c = []
    c.append((toy["PROC COMPOSITE"] > 0.8,
              f"MACHINERY: infinite gen passes Axis-2 (PROC COMPOSITE {toy['PROC COMPOSITE']:.2f}) "
              "— seamless / seed-consistent / O(1) / no extent drift"))
    c.append((toy["score: seam"] > 0.6 and toy["score: revisit"] == 1.0
              and toy["score: extent"] > 0.5,
              f"  seam {toy['score: seam']:.2f}, revisit {toy['score: revisit']:.2f}, "
              f"budget {toy['score: budget']:.2f}, extent {toy['score: extent']:.2f}"))
    c.append((not np.isfinite(real["PROC COMPOSITE"]) and not np.isfinite(s["ble_vae"]["PROC COMPOSITE"]),
              "bounded/peer generators are Axis-2 N/A (not scored as losses) — "
              "InfiniteDiffusion is the first to exercise the procedure axis"))
    c.append((real["COMPOSITE"] > no["COMPOSITE"],
              f"Axis-1 sanity: ERA5 ceiling ({real['COMPOSITE']:.2f}) > white noise "
              f"({no['COMPOSITE']:.2f})"))
    c.append((0.2 < toy["COMPOSITE"] < real["COMPOSITE"],
              f"EXPECTED: toy is mid-pack on Axis-1 ({toy['COMPOSITE']:.2f}) — field quality is "
              "the *denoiser's* job (Phase 2); the toy only proves the machinery"))
    return c


if __name__ == "__main__":
    run()
