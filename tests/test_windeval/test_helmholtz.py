"""Analytic validation of the Helmholtz rotational-fraction metric.

Build fields with KNOWN rotational/divergent content and confirm the decomposition
recovers them. This is the 'calibrate the ruler' discipline applied to the metric:
  - pure vortex (from a streamfunction)      -> rotational fraction ~ 1
  - pure source  (from a velocity potential) -> rotational fraction ~ 0
  - equal-energy mix                         -> ~ 0.5

Run:  ../.venv/bin/python -m tests.test_helmholtz
"""
import numpy as np

from src.eval.windeval.metrics.realism import _helmholtz_rot_frac


def _gaussian(ny, nx, sigma):
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    y -= ny / 2.0
    x -= nx / 2.0
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


def make_vortex(ny=64, nx=64, sigma=8.0):
    psi = _gaussian(ny, nx, sigma)           # streamfunction
    u = -np.gradient(psi, axis=0)            # u = -dψ/dy   (divergence-free)
    v = np.gradient(psi, axis=1)             # v =  dψ/dx
    return u, v


def make_source(ny=64, nx=64, sigma=8.0):
    phi = _gaussian(ny, nx, sigma)           # velocity potential
    u = np.gradient(phi, axis=1)             # u = dφ/dx    (curl-free)
    v = np.gradient(phi, axis=0)             # v = dφ/dy
    return u, v


def _unit_energy(u, v):
    e = np.sqrt(np.sum(u ** 2 + v ** 2))
    return u / e, v / e


def run():
    uv, vv = _unit_energy(*make_vortex())
    us, vs = _unit_energy(*make_source())
    um, vm = uv + us, vv + vs                 # equal-energy mix (components orthogonal)

    cases = [
        ("pure vortex", uv, vv, lambda f: f > 0.90, "~1"),
        ("pure source", us, vs, lambda f: f < 0.10, "~0"),
        ("equal mix", um, vm, lambda f: 0.40 < f < 0.60, "~0.5"),
    ]

    print(f"{'case':14} {'taper=0':>9} {'taper=0.5':>10}  expect  result")
    ok_all = True
    for name, u, v, check, exp in cases:
        f0 = _helmholtz_rot_frac(u, v, 1.0, 1.0, taper=0.0)
        ft = _helmholtz_rot_frac(u, v, 1.0, 1.0, taper=0.5)
        ok = check(f0) and check(ft)          # both with and without taper
        ok_all &= ok
        print(f"{name:14} {f0:9.3f} {ft:10.3f}  {exp:6}  {'PASS' if ok else 'FAIL'}")

    # taper must not corrupt a clean (already-decaying) field
    f_notaper = _helmholtz_rot_frac(uv, vv, 1.0, 1.0, taper=0.0)
    f_taper = _helmholtz_rot_frac(uv, vv, 1.0, 1.0, taper=0.5)
    ok_taper = abs(f_notaper - f_taper) < 0.05
    ok_all &= ok_taper
    print(f"\ntaper leaves clean vortex ~unchanged (|Δ|={abs(f_notaper-f_taper):.3f}): "
          f"{'PASS' if ok_taper else 'FAIL'}")

    print("\n" + ("ALL PASS" if ok_all else "SOME FAILED"))
    return ok_all


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
