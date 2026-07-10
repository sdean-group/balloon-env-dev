"""Vertical-shear distribution — Physical Consistency §Spatial 2 of the benchmark spec.

S_u = ∂u/∂z ≈ Δu/Δz (and S_v) is computed between every adjacent model-level pair at
every (time, y, x) point; the pooled samples form a distribution that is compared to
ERA5's with Wasserstein-1.

Δz problem: generated artifacts store only u,v — no T/q/sp to integrate real altitudes.
We therefore use a *climatological* layer thickness per level pair, precomputed once from
the ERA5 stage-2 data (hypsometric integration, then averaged over time and space) and
applied identically to both sides of the W1 — so the comparison is fair even though the
absolute shear values carry the climatological approximation. Shear is reported in
(m/s)/km for readability.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .distributions import wasserstein1

_DZ_CACHE = Path(__file__).resolve().parent.parent / "data" / "climatological_dz_m.npy"


def climatological_dz(stage2_path=None, cache: Path = _DZ_CACHE) -> np.ndarray:
    """Mean layer thickness Δz (m) per adjacent level pair, shape (L−1,).

    Computed from a stage-2 artifact (has T, q, sp) via the hypsometric equation and
    cached; later calls just load the cache.
    """
    if cache.exists():
        return np.load(cache)
    if stage2_path is None:
        raise FileNotFoundError(
            f"no Δz cache at {cache} — pass a stage-2 artifact path once to build it")
    from .. import artifact, derive
    ds = artifact.read(stage2_path)
    z = derive.altitude(ds)                        # (t, L, y, x), decreasing with index
    dz = (z[:, :-1] - z[:, 1:]).mean(axis=(0, 2, 3))
    np.save(cache, dz.astype("float64"))
    return dz


def shear_samples(ds, dz: np.ndarray, var: str) -> np.ndarray:
    """Pooled Δvar/Δz samples in (m/s)/km over all times, level pairs and points."""
    a = ds[var].values                             # (t, L, y, x)
    s = (a[:, :-1] - a[:, 1:]) / dz[None, :, None, None]   # 1/s
    return s.ravel() * 1e3                          # (m/s)/km


def shear_w1(pred_ds, ref_ds, dz: np.ndarray) -> dict:
    """W1 between pred and ref shear distributions, per component, in (m/s)/km."""
    return {
        f"W1 shear {var} ((m/s)/km)": wasserstein1(
            shear_samples(pred_ds, dz, var), shear_samples(ref_ds, dz, var))
        for var in ("u", "v")
    }
