"""Validate the Axis-2 (procedure) metric suite + calibrate the seam ruler (Phase 1d).

Two parts:
  A. Calibration anchors for the SEAM metric — build fields with KNOWN seam content and
     confirm the metric reads them: a smooth (non-tiled) field scores ≈ ideal; a deliberately
     tile-mismatched field (independent random tiles, hard edges) spikes. This is the
     'calibrate the ruler before trusting it' discipline applied to Axis-2.
  B. End-to-end procedure_scores on the toy InfiniteDiffusion artifact: seam (seamless),
     revisit (deterministic), budget (bounded), extent drift (flat over growing crops), and
     the N/A handling for a bounded generator.

Needs torch + infinite-tensor (for part B) + the harness (jax). Skips cleanly if absent.
Run:  PYTHONPATH=. .pixi/envs/default/bin/python tests/test_windeval/test_procedure.py
"""
import tempfile
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
    from src.eval.windeval import artifact
    from src.eval.windeval.metrics import procedure
    HAVE = True
except ImportError as e:  # pragma: no cover
    HAVE = False
    _ERR = e


def _smooth_divfree(ny, nx, rng, scale=6.0):
    """Smooth divergence-free (u,v) from a low-pass random streamfunction."""
    psi = rng.standard_normal((ny, nx))
    ky = np.fft.fftfreq(ny); kx = np.fft.fftfreq(nx)
    KX, KY = np.meshgrid(kx, ky)
    F = np.fft.fft2(psi) * np.exp(-((np.sqrt(KX ** 2 + KY ** 2) * scale) ** 2))
    psi = np.fft.ifft2(F).real
    u = -np.gradient(psi, axis=0)
    v = np.gradient(psi, axis=1)
    s = 15.0 / (np.hypot(u, v).std() + 1e-9)
    return u * s, v * s


def _field_ds(u, v, seams):
    """(ny,nx) u,v -> a field/ Dataset with declared seam_boundaries + tiled capability."""
    ny, nx = u.shape
    lat = 37.77 + np.arange(ny) * 0.25
    lon = 237.58 + np.arange(nx) * 0.25
    ds = artifact.make_field(u[None], v[None], level=np.array([50.0]), lat=lat, lon=lon,
                             time=np.array([0]))
    ds.attrs["capabilities"] = {"extent": "unbounded", "tiled": True, "random_access": True}
    ds.attrs["seam_boundaries"] = seams
    ds.attrs["coord_to_meters"] = "tangent_plane"
    return ds


def run():
    if not HAVE:
        print(f"SKIP test_procedure (deps not installed: {_ERR})")
        return True

    checks = []

    def chk(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        checks.append(bool(ok))

    rng = np.random.default_rng(0)
    NY = NX = 128
    TILE = 32
    seams = {"y": list(range(TILE, NY, TILE)), "x": list(range(TILE, NX, TILE))}

    # --- A. calibrate the seam metric ---
    us, vs = _smooth_divfree(NY, NX, rng)                       # seamless field, seams DECLARED anyway
    seamless = procedure.seam_discontinuity(_field_ds(us, vs, seams))

    um, vm = np.zeros((NY, NX)), np.zeros((NY, NX))             # tile-mismatched: independent tiles
    for iy in range(0, NY, TILE):
        for ix in range(0, NX, TILE):
            tu, tv = _smooth_divfree(TILE, TILE, rng, scale=3.0)
            um[iy:iy + TILE, ix:ix + TILE] = tu
            vm[iy:iy + TILE, ix:ix + TILE] = tv
    mismatched = procedure.seam_discontinuity(_field_ds(um, vm, seams))

    chk("seam ruler: seamless ≈ ideal (excess≈1, score high)",
        seamless["seam div excess"] < 1.5 and seamless["score: seam"] > 0.6,
        f"excess={seamless['seam div excess']:.2f}, score={seamless['score: seam']:.2f}")
    chk("seam ruler: mismatched spikes (excess>>1, score low)",
        mismatched["seam div excess"] > 2.0 and mismatched["score: seam"] < 0.2,
        f"excess={mismatched['seam div excess']:.2f}, score={mismatched['score: seam']:.2f}")
    chk("seam ruler: discriminates (mismatched >> seamless)",
        mismatched["seam div excess"] > 2 * seamless["seam div excess"],
        f"{mismatched['seam div excess']:.2f} vs {seamless['seam div excess']:.2f}")

    # --- B. end-to-end on the toy InfiniteDiffusion artifact ---
    from src.eval.windeval.generators.infinite_diffusion import InfiniteDiffusionGenerator
    gen = InfiniteDiffusionGenerator(n_levels=8, window=64, stride=32, T=2, seed=0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "idiff.zarr"
        gen.to_artifact(path, height=192, width=192, n_queries=48)
        ds = artifact.read(path)
        q = artifact.read_querylog(path)

        # extent family: growing crops from the same seed
        family = []
        for sz in (64, 96, 128, 192):
            u, v = gen.sampler.field_uv(0, sz, 0, sz)
            lat, lon = gen._coords(0, sz, 0, sz)
            family.append(artifact.make_field(u, v, level=gen.levels, lat=lat, lon=lon,
                                               time=np.array([0])))

        sc = procedure.procedure_scores(ds, querylog=q, extent_family=family)

    chk("toy: seam seamless (score high)", sc["score: seam"] > 0.6,
        f"excess={sc['seam div excess']:.2f}, score={sc['score: seam']:.2f}")
    chk("toy: revisit deterministic (score=1)", sc["score: revisit"] == 1.0,
        f"max|Δ|={sc['revisit max|Δ|']:.1e}")
    chk("toy: budget bounded & ~flat", np.isfinite(sc["score: budget"]),
        f"p50={sc['latency p50 (ms)']:.1f}ms, far/near={sc['budget far/near']:.2f}, "
        f"score={sc['score: budget']:.2f}")
    chk("toy: extent drift ~flat (score high)", sc["score: extent"] > 0.5,
        f"slope/oct={sc['extent drift slope/oct']:+.3f}, score={sc['score: extent']:.2f}")
    chk("toy: PROC COMPOSITE finite", np.isfinite(sc["PROC COMPOSITE"]),
        f"{sc['PROC COMPOSITE']:.2f}")

    # --- N/A handling: bounded generator gets no Axis-2 scores ---
    bnd = _field_ds(us, vs, {"y": [], "x": []})
    bnd.attrs["capabilities"] = {"extent": "bounded", "tiled": False, "random_access": False}
    na = procedure.procedure_scores(bnd)
    chk("bounded gen -> Axis-2 N/A (PROC COMPOSITE NaN)", not np.isfinite(na["PROC COMPOSITE"]))

    ok = all(checks)
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
