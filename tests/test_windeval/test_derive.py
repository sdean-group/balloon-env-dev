"""Validate pressure/altitude/shear derivation on real Stage-2 ERA5.

Run:  ../.venv/bin/python -m tests.test_derive
"""
import numpy as np

from src.eval.windeval import artifact
from src.eval.windeval.ingest_era5 import ingest
from src.eval.windeval import derive
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "src" / "eval" / "windeval" / "data"


def run():
    z_path = ingest(DATA / "era5_sf_uvtq.grib", DATA / "era5_real_stage2.zarr",
                    lnsp_path=DATA / "era5_sf_lnsp.grib")
    ds = artifact.read(z_path)

    p_hpa = derive.full_pressure(ds) / 100.0      # (t,L,y,x)
    z = derive.altitude(ds)                       # (t,L,y,x) metres
    shear, z_mid = derive.vector_shear(ds)        # (t,L-1,y,x) 1/s

    # per-level means (over time, space)
    p_mean = p_hpa.mean(axis=(0, 2, 3))
    z_mean = z.mean(axis=(0, 2, 3))

    print(f"{'lvl':>4} {'p[hPa]':>8} {'alt[km]':>8}")
    for k, p, a in zip(ds["level"].values, p_mean, z_mean):
        print(f"{int(k):4d} {p:8.2f} {a/1000:8.2f}")
    print(f"\nshear |dV/dz|: mean {shear.mean()*1000:.3f} (m/s)/km, "
          f"max {shear.max()*1000:.2f} (m/s)/km")

    checks = [
        (50 <= p_mean.min() and p_mean.max() <= 140, "pressures within 50-140 hPa band"),
        (np.all(np.diff(p_mean) > 0), "pressure increases with level index"),
        (np.all(np.diff(z_mean) < 0), "altitude decreases with level index (up = lower idx)"),
        (4000 < (z_mean[0] - z_mean[-1]) < 9000, f"band spans ~5-7 km (got {(z_mean[0]-z_mean[-1])/1000:.2f} km)"),
        (np.all(np.isfinite(shear)) and shear.min() >= 0, "shear finite & non-negative"),
        (0.1 < shear.mean() * 1000 < 20, "mean shear physically plausible (0.1-20 (m/s)/km)"),
    ]
    ok = all(c for c, _ in checks)
    print()
    for c, msg in checks:
        print(f"{'PASS' if c else 'FAIL'}  {msg}")
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
