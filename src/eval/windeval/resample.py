"""Horizontal regridding onto a common grid — for FAIR cross-generator spectral metrics.

Spectra are functions of spatial scale, so comparing them across generators on
different grids is invalid (different Nyquist limits, scale ranges, bin counts). The
fix: regrid both fields onto one common grid at the COARSER resolution, over an extent
inside both domains (no extrapolation), then recompute spectral metrics there.

Downsampling needs an anti-alias low-pass first (else fine-scale energy aliases into
low wavenumbers); upsampling does not (can't invent scales). Coherence metrics are
dimensionless and need none of this.
"""
from __future__ import annotations

import numpy as np

from . import artifact


def common_grid(center_lat, center_lon, spacing_km, n):
    """1D (lat, lon) degree coords for an n×n grid at `spacing_km`, centred on a point."""
    off_km = (np.arange(n) - (n - 1) / 2.0) * spacing_km
    lat = center_lat + off_km / 111.32
    lon = center_lon + off_km / (111.32 * np.cos(np.deg2rad(center_lat)))
    return lat, lon


def _gaussian_kernel(sigma):
    r = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-r, r + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _smooth_yx(a, sigma):
    """Separable Gaussian blur over the last two axes (y, x)."""
    k = _gaussian_kernel(sigma)
    a = np.apply_along_axis(lambda m: np.convolve(m, k, "same"), -2, a)
    a = np.apply_along_axis(lambda m: np.convolve(m, k, "same"), -1, a)
    return a


def _interp_axis(data, src, tgt, axis):
    """Linear interpolation of `data` along `axis` from src coords to tgt coords."""
    src = np.asarray(src, dtype=float)
    if src[0] > src[-1]:                      # ensure increasing for np.interp
        src = src[::-1]
        data = np.flip(data, axis)
    idx = np.interp(tgt, src, np.arange(len(src)))
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, len(src) - 1)
    w = (idx - lo).reshape([len(tgt) if i == axis else 1 for i in range(data.ndim)])
    return np.take(data, lo, axis) * (1 - w) + np.take(data, hi, axis) * w


def regrid(ds, lat_t, lon_t, *, src_spacing_km=None, target_spacing_km=None):
    """Regrid an artifact's u,v onto (lat_t, lon_t). Anti-alias if downsampling.

    Returns a field-only Dataset (time, level, y, x) on the common grid — enough for
    spatial/spectral metrics.
    """
    u, v = ds["u"].values, ds["v"].values    # (t, L, y, x)
    if (src_spacing_km is not None and target_spacing_km is not None
            and target_spacing_km > src_spacing_km):
        sigma = 0.5 * target_spacing_km / src_spacing_km   # px, ~half-target low-pass
        u, v = _smooth_yx(u, sigma), _smooth_yx(v, sigma)

    lat_src, lon_src = ds["lat"].values, ds["lon"].values
    out = {}
    for name, a in (("u", u), ("v", v)):
        a = _interp_axis(a, lat_src, lat_t, axis=2)        # y
        a = _interp_axis(a, lon_src, lon_t, axis=3)        # x
        out[name] = a
    return artifact.make_field(out["u"], out["v"], level=ds["level"].values,
                               lat=np.asarray(lat_t), lon=np.asarray(lon_t),
                               time=ds["time"].values)
