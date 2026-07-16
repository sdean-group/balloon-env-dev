"""Calibration of the benchmark-v2 metric suite (see docs/benchmark-v2-changes.md).

A metric earns its place by separating known-good from known-broken fields:

  A. Self-comparison: every residual/W1 must be ~0 when pred == ref.
  B. Grid-size invariance: a sub-crop of the same real data must score near the
     same-distribution floor (the density-normalized, physical-k spectra are what make
     different crop sizes comparable at all).
  C. Spatial anchors: everything must sit far above the floor (real half vs half).
     NOTE the windowing finding (docs/benchmark-v2-changes.md): phase-shuffle preserves
     the *rectangular-window* periodogram exactly (the classic trap), but the Kaiser
     window strips boundary leakage from the real field and can't from the shuffled one
     (there the leakage became genuine interior signal) — so under our estimator even
     SR_E catches phase-shuffle, alongside SR_div/SR_vort (u–v cross-phase) and shear W1
     (vertical decorrelation).
  D. Temporal anchors: time-shuffled real data must blow up SR_time. Trajectory
     dispersion is deliberately NOT expected to catch time-shuffle (the jet is
     quasi-stationary, so the shuffled frames still transport tracers about right — the
     project's standing physical finding); it must instead catch wrong TRANSPORT, tested
     with an amplitude-halved field (MSD ~4× low → log-RMSE ≈ ln 4).

Uses the held-out ERA5 (subsampled for speed); SKIPs if the zarr isn't present.

Run:  PYTHONPATH=. .pixi/envs/default/bin/python tests/test_windeval/test_metrics_v2.py
"""
from pathlib import Path

import numpy as np

from src.eval.windeval import artifact
from src.eval.windeval.reference import build_heldout, split
from src.eval.windeval.metrics import run_suite, climatological_dz
from src.eval.windeval.anchors import _phase_randomize

HELDOUT = Path("src/eval/windeval/data/era5_heldout.zarr")
STAGE2 = Path("src/eval/windeval/data/era5_real_stage2.zarr")


def _like(ds, u, v, time=None):
    return artifact.make_field(u.astype("float32"), v.astype("float32"),
                               level=ds["level"].values, lat=ds["lat"].values,
                               lon=ds["lon"].values,
                               time=ds["time"].values if time is None else time)


def run():
    if not HELDOUT.parent.exists() or not (HELDOUT.exists() or
                                           (HELDOUT.parent / "era5_temporal.zarr").exists()):
        print("SKIP test_metrics_v2 (no held-out ERA5 data)")
        return True

    checks = []

    def chk(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        checks.append(bool(ok))

    ds = artifact.read(build_heldout())
    dz = climatological_dz(STAGE2 if STAGE2.exists() else None)
    rng = np.random.default_rng(0)

    # small spatial subsets for speed: every 12th hour of each half
    half_a, half_b = split(ds)
    A = half_a.isel(time=slice(0, None, 12)).compute()
    B = half_b.isel(time=slice(0, None, 12)).compute()

    # ---- A. self-comparison ----
    s_self, _ = run_suite(A, A, dz=dz)
    chk("self: SR_E ≈ 0", s_self["SR_E"] < 1e-9)
    chk("self: W1 u ≈ 0", s_self["W1 u (m/s)"] < 1e-9)
    chk("self: L_eff resolved to grid", s_self["resolved_to_grid"])

    # ---- floor: real half vs real half ----
    s_floor, _ = run_suite(A, B, dz=dz)
    chk("floor is small but nonzero", 0 < s_floor["SR_E"] < 1.0,
        f"SR_E floor {s_floor['SR_E']:.3f}")

    # ---- B. grid-size invariance: 81² sub-crop of the same data vs the 121² ref ----
    sub = A.isel(y=slice(20, 101), x=slice(20, 101))
    s_sub, _ = run_suite(sub, B, dz=dz)
    chk("sub-crop scores near the floor (density norm works)",
        s_sub["SR_E"] < s_floor["SR_E"] + 0.5,
        f"sub {s_sub['SR_E']:.3f} vs floor {s_floor['SR_E']:.3f}")

    # ---- C. spatial anchors ----
    u, v = A["u"].values.copy(), A["v"].values.copy()
    for t in range(u.shape[0]):
        for l in range(u.shape[1]):
            u[t, l] = _phase_randomize(u[t, l], rng)
            v[t, l] = _phase_randomize(v[t, l], rng)
    ps = _like(A, u, v)
    # white noise matched to per-(t,level) mean/std, like anchors.white_noise
    au, av = A["u"].values, A["v"].values
    nu = (au.mean(axis=(2, 3), keepdims=True)
          + au.std(axis=(2, 3), keepdims=True) * rng.standard_normal(au.shape))
    nv = (av.mean(axis=(2, 3), keepdims=True)
          + av.std(axis=(2, 3), keepdims=True) * rng.standard_normal(av.shape))
    noise = _like(A, nu, nv)
    s_ps, _ = run_suite(ps, B, dz=dz)
    s_no, _ = run_suite(noise, B, dz=dz)

    chk("SR_E catches phase-shuffle (windowed estimator; see module doc)",
        s_ps["SR_E"] > 2 * s_floor["SR_E"],
        f"ps {s_ps['SR_E']:.3f} vs floor {s_floor['SR_E']:.3f}")
    chk("SR_div/SR_vort catch it too",
        s_ps["SR_div"] > 2 * s_floor["SR_div"] and s_ps["SR_vort"] > 2 * s_floor["SR_vort"],
        f"div {s_floor['SR_div']:.2f}->{s_ps['SR_div']:.2f}, "
        f"vort {s_floor['SR_vort']:.2f}->{s_ps['SR_vort']:.2f}")
    chk("shear W1 catches vertical decorrelation",
        s_ps["W1 shear u ((m/s)/km)"] > 2 * s_floor["W1 shear u ((m/s)/km)"],
        f"{s_floor['W1 shear u ((m/s)/km)']:.2f} -> {s_ps['W1 shear u ((m/s)/km)']:.2f}")
    chk("white noise: SR_E blows up", s_no["SR_E"] > 5 * max(s_floor["SR_E"], 0.1),
        f"{s_no['SR_E']:.2f}")
    chk("marginal W1 catches phase-shuffle (wrecked means)",
        s_ps["W1 u (m/s)"] > 3 * s_floor["W1 u (m/s)"],
        f"floor {s_floor['W1 u (m/s)']:.2f} -> ps {s_ps['W1 u (m/s)']:.2f}")
    # designed blind spot: moment-matched noise has correct marginals BY CONSTRUCTION —
    # marginal W1 measures placement, not structure; SR_E is what catches this anchor.
    chk("moment-matched noise sits at the marginal-W1 floor (blind spot is understood)",
        s_no["W1 u (m/s)"] < 2 * s_floor["W1 u (m/s)"],
        f"noise {s_no['W1 u (m/s)']:.2f} vs floor {s_floor['W1 u (m/s)']:.2f} "
        f"(SR_E catches it: {s_no['SR_E']:.1f})")

    # ---- D. temporal anchors (January block, hourly; within-season floor) ----
    T = ds.isel(time=slice(0, 168)).compute()              # Jan 8–14, contiguous
    Ta = T.isel(time=slice(0, 72))                         # Jan 8–10
    Tb = T.isel(time=slice(72, 168))                       # Jan 11–14
    perm = rng.permutation(Ta.sizes["time"])
    tsh = _like(Ta, Ta["u"].values[perm], Ta["v"].values[perm])
    half = _like(Ta, 0.5 * Ta["u"].values, 0.5 * Ta["v"].values)
    s_tfloor, _ = run_suite(Ta, Tb, dz=dz)
    s_tsh, _ = run_suite(tsh, Tb, dz=dz)
    s_half, _ = run_suite(half, Tb, dz=dz)
    chk("temporal metrics engaged (SR_time finite)", np.isfinite(s_tfloor["SR_time"]))
    chk("time-shuffle blows up SR_time",
        s_tsh["SR_time"] > 2 * s_tfloor["SR_time"],
        f"floor {s_tfloor['SR_time']:.2f} -> shuffled {s_tsh['SR_time']:.2f}")
    chk("dispersion catches wrong transport (amplitude-halved winds)",
        s_half["disp log-MSD RMSE"] > s_tfloor["disp log-MSD RMSE"] + 0.5
        and s_half["final spread ratio"] < 0.8,
        f"floor {s_tfloor['disp log-MSD RMSE']:.2f} -> halved {s_half['disp log-MSD RMSE']:.2f}, "
        f"spread ratio {s_half['final spread ratio']:.2f}")

    ok = all(checks)
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
