"""HEALPix face layout: storage (NEST) <-> the 12-face ``(F, nside, nside)`` layout convolutions use.

The stores written by :mod:`ingest_arco` are NEST ordered (healpy-native; the nested hierarchy
is the coarse/fine contract). Convolutions and cross-face halo padding want each of the 12
base faces as a square image in one fixed orientation: earth2grid's ``HEALPIX_PAD_XY``
(origin north, clockwise), which is the convention ``earth2grid.healpix.pad`` expects.
Converting is a fixed permutation of the pixel axis; we compute it once with earth2grid,
verify it geometrically with healpy, and cache it as ``.npy`` so training code needs
neither library at load time.

Conventions
-----------
- ``perm[k]`` is the NEST index of the k-th pixel in XY order, so ``xy = nest[..., perm]``.
- Faces: ``xy.reshape(..., 12, nside, nside)``; index ``[f, y, x]``.
- ``inv`` is the inverse permutation: ``nest = xy[..., inv]``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def nest_to_xy_perm(nside: int) -> np.ndarray:
    """Permutation NEST -> HEALPIX_PAD_XY for ``12*nside**2`` pixels (needs earth2grid)."""
    import torch
    from earth2grid import healpix
    # reorder(x, src, dest): data given in src order returned in dest order. With identity
    # data in NEST order, out[k] is the NEST index of the k-th XY pixel. Values are exact in
    # float64 up to 2**53, far beyond any nside we use.
    idx = torch.arange(12 * nside * nside, dtype=torch.float64)
    out = healpix.reorder(idx, healpix.PixelOrder.NEST, healpix.HEALPIX_PAD_XY)
    perm = out.round().to(torch.int64).numpy()
    if sorted(perm.tolist()) != list(range(12 * nside * nside)):
        raise RuntimeError("reorder did not return a permutation")
    return perm


def face_latlon(nside: int, perm: np.ndarray) -> np.ndarray:
    """Per-pixel (lat, lon) in degrees in face layout: ``(2, 12, nside, nside)`` (needs healpy)."""
    import healpy as hp
    lon, lat = hp.pix2ang(nside, np.arange(12 * nside * nside), nest=True, lonlat=True)
    return np.stack([lat[perm], lon[perm]]).reshape(2, 12, nside, nside)


def coord_channels(nside: int, perm: np.ndarray) -> np.ndarray:
    """Global, seam-free coordinate channels ``(3, 12, nside, nside)`` float32:
    ``lat/90``, ``sin(lon)``, ``cos(lon)``. Replaces the regional CoordNorm (which normalised by
    a training-box half-width and wrapped longitude) now that the domain is the sphere."""
    lat, lon = face_latlon(nside, perm)
    rl = np.radians(lon)
    return np.stack([lat / 90.0, np.sin(rl), np.cos(rl)]).astype(np.float32)


def check_layout(nside: int, perm: np.ndarray) -> dict:
    """Geometric sanity: XY neighbours along x and y must be angular neighbours.

    Returns the max great-circle step (deg) between horizontally / vertically adjacent
    pixels within faces, relative to the nominal pixel size. Both should be ~1-1.5x; a
    wrong permutation gives steps of tens of degrees.
    """
    lat, lon = np.radians(face_latlon(nside, perm))
    def step(a_lat, a_lon, b_lat, b_lon):
        d = np.sin(a_lat) * np.sin(b_lat) + np.cos(a_lat) * np.cos(b_lat) * np.cos(a_lon - b_lon)
        return np.degrees(np.arccos(np.clip(d, -1, 1)))
    dx = step(lat[:, :, :-1], lon[:, :, :-1], lat[:, :, 1:], lon[:, :, 1:])
    dy = step(lat[:, :-1, :], lon[:, :-1, :], lat[:, 1:, :], lon[:, 1:, :])
    nominal = np.degrees(np.sqrt(4 * np.pi / (12 * nside * nside)))
    return {"nominal_deg": float(nominal), "max_dx_over_nominal": float(dx.max() / nominal),
            "max_dy_over_nominal": float(dy.max() / nominal),
            "mean_dx_over_nominal": float(dx.mean() / nominal)}


class FaceLayout:
    """Cached NEST<->XY permutation for one nside, with helpers."""

    def __init__(self, nside: int, perm: np.ndarray) -> None:
        self.nside = int(nside)
        self.npix = 12 * self.nside ** 2
        self.perm = np.asarray(perm, dtype=np.int64)
        if self.perm.shape != (self.npix,):
            raise ValueError(f"perm has {self.perm.shape}, expected ({self.npix},)")
        self.inv = np.empty_like(self.perm)
        self.inv[self.perm] = np.arange(self.npix)

    @classmethod
    def load(cls, nside: int, cache_dir: str | Path) -> "FaceLayout":
        p = Path(cache_dir) / f"nest2xy_{nside}.npy"
        return cls(nside, np.load(p))

    @classmethod
    def build(cls, nside: int, cache_dir: str | Path) -> "FaceLayout":
        """Compute with earth2grid, verify with healpy, cache. Run once per nside."""
        perm = nest_to_xy_perm(nside)
        rep = check_layout(nside, perm)
        if rep["max_dx_over_nominal"] > 2.5 or rep["max_dy_over_nominal"] > 2.5:
            raise RuntimeError(f"layout check failed for nside {nside}: {rep}")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(Path(cache_dir) / f"nest2xy_{nside}.npy", perm)
        return cls(nside, perm)

    def to_faces(self, x: np.ndarray):
        """``(..., npix)`` NEST -> ``(..., 12, nside, nside)`` XY. Works for numpy and torch."""
        y = x[..., self.perm]
        return y.reshape(*x.shape[:-1], 12, self.nside, self.nside)

    def from_faces(self, x):
        """``(..., 12, nside, nside)`` XY -> ``(..., npix)`` NEST."""
        y = x.reshape(*x.shape[:-3], self.npix)
        return y[..., self.inv]


if __name__ == "__main__":  # build + verify caches:  python layout.py <cache_dir> [nside ...]
    import sys
    cache = sys.argv[1]
    for n in [int(a) for a in sys.argv[2:]] or [32, 256]:
        lay = FaceLayout.build(n, cache)
        print(f"nside {n}: cached {lay.npix} pixels ->", check_layout(n, lay.perm))
