"""Validate the InfiniteDiffusion machinery + toy denoiser (Phase 1a/1b).

Confirms the Axis-2 signature claims the method makes — seed-consistency, region/order
invariance, constant-time random access — plus basic Axis-1 sanity of the toy denoiser
(divergence-free / rotational-dominated, target spectrum, no seam spike).

Needs torch + infinite-tensor (the generator-only deps). Skips cleanly if absent, so the
harness test suite still runs in an env without them.

Run:  PYTHONPATH=. .pixi/envs/default/bin/python tests/test_windeval/test_infinite_diffusion.py
"""
import numpy as np

try:
    import torch  # noqa: F401
    from src.eval.windeval.generators.infinite_diffusion import (
        InfiniteDiffusion, ToyDivFreeDenoiser,
    )
    HAVE_TORCH = True
except ImportError as e:  # pragma: no cover
    HAVE_TORCH = False
    _IMPORT_ERR = e

NLEV = 4


def _divergence(u, v):
    dudy, dudx = np.gradient(u, axis=(0, 1))
    dvdy, dvdx = np.gradient(v, axis=(0, 1))
    return dudx + dvdy, dvdx - dudy  # div, vort


def _gen(seed=0, denoiser=None):
    toy = denoiser or ToyDivFreeDenoiser(NLEV, slope=-3.0, rms=15.0, vertical_corr=1.0)
    return InfiniteDiffusion(toy, window=32, stride=16, T=2, seed=seed,
                             cache_bytes=64 * 1024 * 1024)


def _radial_slope(f2d):
    F = np.fft.fft2(f2d - f2d.mean())
    p = np.abs(F).ravel() ** 2
    ky = np.fft.fftfreq(f2d.shape[0]); kx = np.fft.fftfreq(f2d.shape[1])
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX ** 2 + KY ** 2).ravel()
    m = kr > 0
    edges = np.linspace(0, kr[m].max(), 17)
    idx = np.digitize(kr, edges) - 1
    kc = 0.5 * (edges[:-1] + edges[1:])
    pw = np.array([p[idx == b].mean() if np.any(idx == b) else np.nan for b in range(16)])
    ok = np.isfinite(pw) & (kc > 0) & (pw > 0)
    return float(np.polyfit(np.log10(kc[ok]), np.log10(pw[ok]), 1)[0])


class _CountingDenoiser:
    """Wraps a denoiser to count Phi evaluations (for the O(1) access check)."""
    def __init__(self, inner):
        self.inner, self.n_channels, self.calls = inner, inner.n_channels, 0

    def __call__(self, x, t=0):
        self.calls += 1
        return self.inner(x, t)


def run():
    if not HAVE_TORCH:
        print(f"SKIP test_infinite_diffusion (torch/infinite-tensor not installed: {_IMPORT_ERR})")
        return True

    checks = []

    def chk(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        checks.append(bool(ok))

    # determinism + region/order invariance
    g = _gen(0)
    chk("determinism", torch.equal(g.materialize(0, 64, 0, 64), g.materialize(0, 64, 0, 64)))
    g2 = _gen(0)
    _ = g2.materialize(100, 180, -40, 40)            # different query order populates cache differently
    big, sub = g2.materialize(0, 96, 0, 96), g2.materialize(32, 64, 32, 64)
    chk("region/order invariance", torch.allclose(big[:, 32:64, 32:64], sub, atol=1e-5))

    # seed-consistency across fresh instances + sensitivity
    a = _gen(7).materialize(0, 64, 0, 64)
    chk("seed-consistency (fresh instance identical)", torch.equal(a, _gen(7).materialize(0, 64, 0, 64)))
    chk("seed sensitivity", not torch.allclose(a, _gen(8).materialize(0, 64, 0, 64), atol=1e-3))

    # constant-time random access: Phi-call count independent of location magnitude
    # (window-aligned offsets isolate the location claim from sub-stride phase variation)
    def n_calls(y0, x0):
        cd = _CountingDenoiser(ToyDivFreeDenoiser(NLEV))
        _gen(0, cd).materialize(y0, y0 + 64, x0, x0 + 64)
        return cd.calls
    c0, c1, c2 = n_calls(0, 0), n_calls(32_000, -160_000), n_calls(32_000_000, 64_000_000)
    chk("O(1) random access", c0 == c1 == c2, f"calls={c0},{c1},{c2}")

    # toy field sanity
    u, v = g.field_uv(0, 96, 0, 96)
    div, vort = _divergence(u[0], v[0])
    vd = np.abs(vort).std() / (np.abs(div).std() + 1e-12)
    chk("rotational-dominated (vort/div >> 1)", vd > 2.0, f"vort/div={vd:.2f}")

    field = g.materialize(0, 128, 0, 128).cpu().numpy()[0]
    step = np.abs(np.diff(field, axis=1))
    seams = [s - 1 for s in g.seam_lines(0, 128, 0, 128)["x"] if 0 < s - 1 < step.shape[1]]
    interior = [i for i in range(step.shape[1]) if i not in seams]
    chk("seam continuity (no seam spike)", step[:, seams].mean() < 2.0 * step[:, interior].mean(),
        f"seam={step[:, seams].mean():.2f} vs interior={step[:, interior].mean():.2f}")

    slope = np.mean([_radial_slope(u[0]), _radial_slope(v[0])])
    chk("spectrum slope near target", -4.0 < slope < -2.0, f"slope={slope:.2f}")

    ok = all(checks)
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
