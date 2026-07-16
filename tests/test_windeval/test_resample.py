"""Validate the common-grid resampler.

Run:  ../.venv/bin/python -m tests.test_resample
"""
import numpy as np

from src.eval.windeval import artifact, resample
from src.eval.windeval.artifact import grid_spacing_m
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "src" / "eval" / "windeval" / "data"


def run():
    lat, lon = resample.common_grid(37.77, 237.58, 50.0, 16)
    dlat_km = abs(lat[1] - lat[0]) * 111.32

    ds = artifact.read(DATA / "era5_real_stage2.zarr")

    # identity: regrid onto the source grid itself -> ~unchanged
    same = resample.regrid(ds, ds["lat"].values, ds["lon"].values)
    rel_err = (np.abs(same["u"].values - ds["u"].values).mean()
               / (np.abs(ds["u"].values).mean() + 1e-9))

    # anti-aliased downsample to 50 km -> reduces variance, spacing ~50 km
    coarse = resample.regrid(ds, lat, lon, src_spacing_km=27.83, target_spacing_km=50.0)
    dx, dy = grid_spacing_m(coarse)
    var_ratio = coarse["u"].values.var() / ds["u"].values.var()

    print(f"common-grid lat spacing: {dlat_km:.1f} km (target 50)")
    print(f"identity regrid rel error: {rel_err:.4f}")
    print(f"coarse grid dx,dy: {dx/1000:.1f}, {dy/1000:.1f} km")
    print(f"variance ratio after downsample: {var_ratio:.3f}")

    checks = [
        (abs(dlat_km - 50) < 1, "common grid spacing ≈ 50 km"),
        (rel_err < 0.02, "identity regrid preserves the field"),
        (abs(dx / 1000 - 50) < 5 and abs(dy / 1000 - 50) < 5, "coarse grid ≈ 50 km dx,dy"),
        (0.5 < var_ratio <= 1.05, "anti-alias downsample reduces (not inflates) variance"),
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
