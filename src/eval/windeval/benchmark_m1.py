"""M1 leaderboard — the TRAINED InfiniteDiffusion denoiser on both axes (Phase 3 / M2).

Phase 1 (M0) proved the *machinery* with a toy denoiser. Phase 2 (M1) trained the real
window denoiser; its single-crop Axis-1 gate passed (COMPOSITE 0.60 > toy 0.48 at step
84k). This board does the **swap** — drops the trained denoiser into the *unmodified*
InfiniteDiffusion machinery via the same generator the toy used — and scores it on both
axes next to ERA5 / BLE-VAE / the toy / the lower-bound anchors.

The story M1 must show:
  - **Axis-1**: trained > toy (the denoiser learned real wind structure), climbing toward
    the ERA5 ceiling — field quality is the model's job and the model now does it.
  - **Axis-2**: trained keeps the machinery's claims (seamless / seed-consistent / O(1) /
    no extent drift). Swapping a *learned* Phi for the analytic toy must NOT break the
    blending wrapper — that's the whole point of the two-halves design.

Run (full pixi env — jax + infinite-tensor):
    PYTHONPATH=. .pixi/envs/default/bin/python -m src.eval.windeval.benchmark_m1 \
        --ckpt runs/idiff_m1/step_84000.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from . import artifact, anchors
from .ingest_era5 import ingest
from .metrics import field_scores, procedure_scores
from .metrics.realism import _amplitude_score
from .benchmark_m0 import _fmt, _toy_artifact_and_scores

DATA = Path(__file__).resolve().parent / "data"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _trained_artifact_and_scores(ckpt: str, *, device: str, num_steps: int, s_churn: float = 0.0,
                                 height: int = 192, width: int = 192, n_queries: int = 48):
    """Swap the trained denoiser into the machinery, materialize an artifact, score both axes."""
    from .generators.infinite_diffusion import InfiniteDiffusionGenerator
    from .generators.infinite_diffusion.trained import TrainedWindowDenoiser

    phi = TrainedWindowDenoiser(ckpt, num_steps=num_steps, device=device, s_churn=s_churn)
    # outer T=1: one MultiDiffusion blend of complete internal samples (see trained.py).
    gen = InfiniteDiffusionGenerator(
        denoiser=phi, levels=phi.stats.levels.astype(float),
        window=64, stride=32, T=1, seed=0,
        name="infinite_diffusion_trained", device=device,
    )
    path = DATA / "infinite_diffusion_trained.zarr"
    gen.to_artifact(path, height=height, width=width, n_queries=n_queries)
    ds = artifact.read(path)
    q = artifact.read_querylog(path)

    family = []
    for sz in (64, 96, 128, 192):
        u, v = gen.sampler.field_uv(0, sz, 0, sz)
        lat, lon = gen._coords(0, sz, 0, sz)
        family.append(artifact.make_field(u, v, level=gen.levels, lat=lat, lon=lon,
                                           time=np.array([0])))

    a1 = field_scores(ds)
    a2 = procedure_scores(ds, querylog=q, extent_family=family)
    return {**a1, **a2}, path


def run(ckpt: str = "runs/idiff_m1/step_84000.pt", *, device: str | None = None,
        num_steps: int = 18, s_churn: float = 0.0) -> str:
    DATA.mkdir(exist_ok=True)
    device = device or _pick_device()
    print(f"[m1] scoring trained denoiser {ckpt} on device={device} "
          f"(num_steps={num_steps}, s_churn={s_churn})")

    # peers + anchors (Axis-1 only; no procedure capabilities -> Axis-2 N/A)
    real_z = ingest(DATA / "era5_sf_uvtq.grib", DATA / "era5_real_stage2.zarr",
                    lnsp_path=DATA / "era5_sf_lnsp.grib")
    ps_z = anchors.phase_shuffle(real_z, DATA / "anchor_ps_stage2.zarr", seed=0)
    no_z = anchors.white_noise(real_z, DATA / "anchor_noise_stage2.zarr", seed=0)

    scores: dict[str, dict] = {}
    for name, z in [("era5_real (peer)", real_z), ("phase_shuffle", ps_z), ("white_noise", no_z)]:
        ds = artifact.read(z)
        scores[name] = {**field_scores(ds), **procedure_scores(ds)}
    ble = artifact.read(DATA / "ble_vae_0.zarr")
    scores["ble_vae"] = {**field_scores(ble), **procedure_scores(ble)}

    # the two infinite generators: trained (new) + toy (machinery baseline)
    scores["infinite_diffusion_trained"], trained_path = _trained_artifact_and_scores(
        ckpt, device=device, num_steps=num_steps, s_churn=s_churn)
    scores["infinite_diffusion_toy"], _ = _toy_artifact_and_scores()

    # amplitude (scale) — score every generator against the ERA5 peer's RMS wind speed.
    # All field_scores calls already emit the raw "amplitude rms"; we add the relative score
    # here so the reference is consistent across the board. NOT folded into COMPOSITE.
    ref_rms = scores["era5_real (peer)"].get("amplitude rms")
    if ref_rms:
        for c in scores:
            rms = scores[c].get("amplitude rms")
            if rms is not None:
                scores[c]["score: amplitude"] = _amplitude_score(rms, ref_rms)

    # Decompose the trained model's amplitude deficit: climatology offset (its training box is
    # genuinely calmer than the era5_real peer) vs the real generation defect (under-dispersion
    # vs its OWN target). Computed from the training zarr when present.
    amp_note = ""
    train_zarr = DATA / "era5_train.zarr"
    tr_rms = scores["infinite_diffusion_trained"].get("amplitude rms")
    if train_zarr.exists() and tr_rms and ref_rms:
        import xarray as xr
        tz = xr.open_zarr(train_zarr, consolidated=False, zarr_format=2)
        train_rms = float(np.sqrt(np.mean(tz["u"].values ** 2 + tz["v"].values ** 2)))
        offset = _amplitude_score(train_rms, ref_rms)
        defect = _amplitude_score(tr_rms, train_rms)
        amp_note = (
            f"Decomposition for the trained model: peer rms {ref_rms:.1f}, its training-climatology "
            f"rms {train_rms:.1f} (a perfect model would still score only **{offset:.2f}** vs the "
            f"peer — the unavoidable climatology offset), trained rms {tr_rms:.1f} → **{defect:.2f}** "
            f"vs its own climatology = the real generation defect (anomaly-variance under-dispersion)."
        )

    cols = ["era5_real (peer)", "ble_vae", "infinite_diffusion_trained",
            "infinite_diffusion_toy", "phase_shuffle", "white_noise"]

    def row(label, key, bold=False, fmt="{:.3f}"):
        def cell(c):
            v = scores[c].get(key, np.nan)
            s = _fmt(v, fmt)
            return f"**{s}**" if (bold and s != "N/A") else s
        return f"| {label} | " + " | ".join(cell(c) for c in cols) + " |"

    L = [
        "# M1 Leaderboard — TRAINED InfiniteDiffusion on both axes", "",
        "Phase 3 swap: the trained window denoiser dropped into the *unmodified* machinery. "
        "Axis-1 = field quality (reference-free); Axis-2 = the procedure claims only the "
        "unbounded class makes. Scores ∈ [0,1], higher = better; **N/A** = capability not "
        "declared (not a failure).", "",
        f"Trained checkpoint: `{ckpt}` · device `{device}` · internal EDM steps {num_steps}.", "",
        "## Headline — two axes", "",
        "| Axis | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
        row("Axis-1: spatial COMPOSITE", "COMPOSITE", bold=True),
        row("Axis-2: PROC COMPOSITE", "PROC COMPOSITE", bold=True),
        "",
        "## Axis-1 detail (field quality — the model's job)", "",
        "| Metric | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
        row("spectrum slope (≈ −3)", "spectrum slope", fmt="{:.2f}"),
        row("increment kurtosis (> 3)", "increment kurtosis", fmt="{:.2f}"),
        row("vort/div ratio (> 1)", "vort/div ratio", fmt="{:.2f}"),
        row("score: spectrum", "score: spectrum"),
        row("score: intermittency", "score: intermittency"),
        "",
        "### Amplitude (scale — NOT in COMPOSITE; scored vs the ERA5 peer RMS)", "",
        "The four metrics above are scale-invariant, so a too-calm field with the right "
        "structure slips past them. `amplitude rms` (RMS wind speed) catches that; "
        "`score: amplitude` = min(r, 1/r) vs the peer (1 = parity). The trained model targets its "
        "*training* climatology (lat 25–55°N, 4 seasonal weeks), genuinely calmer than this "
        "narrower era5_real peer (lat 33–42°N), so part of the deficit is climatology, not "
        "generation error:", "",
        f"> {amp_note}" if amp_note else "", "",
        "| Metric | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
        row("amplitude rms (m/s)", "amplitude rms", fmt="{:.2f}"),
        row("score: amplitude (vs peer)", "score: amplitude"),
        "",
        "## Axis-2 detail — the procedure (machinery preserved under the swap?)", "",
        "| Procedure metric | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
        row("seam discontinuity (score; 1=seamless)", "score: seam"),
        row("  └ seam |div| excess (1=ideal)", "seam div excess", fmt="{:.2f}"),
        row("revisit determinism (score; 1=exact)", "score: revisit"),
        row("  └ revisit max|Δ|", "revisit max|Δ|", fmt="{:.1e}"),
        row("budget / O(1) (score; far≈near)", "score: budget"),
        row("  └ budget far/near ratio", "budget far/near", fmt="{:.2f}"),
        row("extent drift (score; 1=flat)", "score: extent"),
        row("  └ drift slope / octave", "extent drift slope/oct", fmt="{:+.3f}"),
        "",
        "## Verdict (automated checks)",
    ]
    for ok, msg in _verdict(scores):
        L.append(f"- {'✅' if ok else '❌'} {msg}")

    report = "\n".join(L)
    out = DATA.parent / "benchmark_m1_report.md"
    out.write_text(report)
    print(report)
    print(f"\ntrained artifact: {trained_path}\nreport: {out}")
    return report


def _verdict(s):
    tr = s["infinite_diffusion_trained"]
    toy = s["infinite_diffusion_toy"]
    real, no = s["era5_real (peer)"], s["white_noise"]
    c = []
    c.append((tr["COMPOSITE"] > toy["COMPOSITE"],
              f"AXIS-1: trained ({tr['COMPOSITE']:.2f}) beats the toy ({toy['COMPOSITE']:.2f}) "
              "— the learned denoiser produces better wind structure than the analytic stand-in"))
    c.append((tr["score: spectrum"] >= 0.9,
              f"  spectrum solved: score {tr['score: spectrum']:.2f} (slope {tr['spectrum slope']:.2f})"))
    c.append((no["COMPOSITE"] < tr["COMPOSITE"] <= real["COMPOSITE"] + 1e-9,
              f"  trained sits between noise ({no['COMPOSITE']:.2f}) and the ERA5 ceiling "
              f"({real['COMPOSITE']:.2f}); residual gap is mostly intermittency "
              f"(score {tr['score: intermittency']:.2f})"))
    c.append((tr["PROC COMPOSITE"] > 0.8,
              f"AXIS-2 PRESERVED: swapping a *learned* denoiser keeps the machinery claims "
              f"(PROC COMPOSITE {tr['PROC COMPOSITE']:.2f}) — seamless / seed-consistent / O(1)"))
    c.append((tr["score: seam"] > 0.6 and tr["score: revisit"] == 1.0 and tr["score: extent"] > 0.5,
              f"  seam {tr['score: seam']:.2f}, revisit {tr['score: revisit']:.2f}, "
              f"budget {tr['score: budget']:.2f}, extent {tr['score: extent']:.2f}"))
    c.append((not np.isfinite(real["PROC COMPOSITE"]) and not np.isfinite(s["ble_vae"]["PROC COMPOSITE"]),
              "bounded/peer generators remain Axis-2 N/A (the procedure axis is only for "
              "unbounded generators)"))
    return c


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="M1 leaderboard — trained InfiniteDiffusion, both axes.")
    ap.add_argument("--ckpt", default="runs/idiff_m1/step_84000.pt")
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    ap.add_argument("--steps", type=int, default=18, help="internal EDM sampler steps")
    ap.add_argument("--s-churn", type=float, default=0.0,
                    help="EDM stochasticity (0=deterministic ODE; >0 = window-seeded churn)")
    args = ap.parse_args(argv)
    run(args.ckpt, device=args.device, num_steps=args.steps, s_churn=args.s_churn)


if __name__ == "__main__":
    main()
