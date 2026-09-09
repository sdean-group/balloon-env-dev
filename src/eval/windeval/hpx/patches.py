"""Patches on the padded HEALPix faces (Stage 2 geometry).

A face image is ``(12, nside, nside)`` in earth2grid's HEALPIX_PAD_XY order. Padding each face
by ``pad`` pixels with ``earth2grid.healpix.pad`` fills the halo from the neighbouring faces
(the 8 three-face vertices get a fabricated corner, which only ever lies in the cropped
padding). Applying that pad once to an image of NEST pixel indices gives a lookup table
``pidx (12, nside+2pad, nside+2pad) -> NEST index``; a patch is then a gather from any NEST-
ordered array, and the coarse parent of a fine NEST pixel is ``index // (ratio**2)`` because
the nested ordering keeps children contiguous. Patch offsets are multiples of ``ratio`` so each
patch is exactly ``(P/ratio)^2`` whole coarse cells and the block-mean projection is exact.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def padded_index_map(nside: int, pad: int, cache_dir: str | Path, perm: np.ndarray | None = None) -> np.ndarray:
    """``(12, nside+2pad, nside+2pad)`` int64 NEST indices; built once (needs earth2grid), cached."""
    cache_dir = Path(cache_dir).expanduser(); p = cache_dir / f"padidx_{nside}_p{pad}.npy"
    if p.exists():
        return np.load(p)
    import torch
    from earth2grid import healpix
    if perm is None:
        from layout import FaceLayout
        perm = FaceLayout.load(nside, cache_dir).perm
    idx = torch.from_numpy(np.asarray(perm, dtype=np.float64)).reshape(1, 12, 1, nside, nside)
    with healpix.pad_backend(healpix.PaddingBackends.indexing):
        padded = healpix.pad(idx, padding=pad)[0, :, 0].numpy()
    q = np.round(padded).astype(np.int64)                     # fabricated corners: nearest source pixel
    q = np.clip(q, 0, 12 * nside * nside - 1)
    cache_dir.mkdir(parents=True, exist_ok=True); np.save(p, q)
    return q


class PatchGeometry:
    """Patch positions on the padded faces and the gathers that cut them from NEST arrays."""

    def __init__(self, nside_fine: int, nside_coarse: int, *, patch: int = 64, pad: int = 32,
                 cache_dir: str | Path = "~/data/hpx_layout") -> None:
        self.nside, self.nside_c, self.patch, self.pad = int(nside_fine), int(nside_coarse), int(patch), int(pad)
        self.ratio = self.nside // self.nside_c
        assert self.nside == self.ratio * self.nside_c and self.patch % self.ratio == 0 and self.pad % self.ratio == 0
        self.pidx = padded_index_map(self.nside, self.pad, cache_dir)        # (12, S, S)
        self.size = self.nside + 2 * self.pad
        self.offsets = np.arange(0, self.size - self.patch + 1, self.ratio)  # coarse-aligned positions

    def n_positions(self) -> int:
        return 12 * len(self.offsets) ** 2

    def random_patches(self, rng: np.random.Generator, k: int) -> list[tuple[int, int, int]]:
        f = rng.integers(0, 12, size=k); i = rng.choice(self.offsets, size=k); j = rng.choice(self.offsets, size=k)
        return [(int(a), int(b), int(c)) for a, b, c in zip(f, i, j)]

    def fine_index(self, face: int, i: int, j: int) -> np.ndarray:
        """``(P, P)`` NEST indices into the fine sphere."""
        return self.pidx[face, i:i + self.patch, j:j + self.patch]

    def coarse_index(self, face: int, i: int, j: int) -> np.ndarray:
        """``(P, P)`` NEST indices of each fine pixel's coarse parent (piecewise constant on ratio x ratio blocks)."""
        return self.fine_index(face, i, j) // (self.ratio ** 2)

    def block_mean(self, x: np.ndarray) -> np.ndarray:
        """``(..., P, P)`` -> ``(..., P/r, P/r)`` mean over each coarse cell's block."""
        r = self.ratio; s = x.shape[:-2]; n = self.patch // r
        return x.reshape(*s, n, r, n, r).mean(axis=(-3, -1))

    def lift(self, c: np.ndarray) -> np.ndarray:
        """``(..., P/r, P/r)`` -> ``(..., P, P)`` nearest lift (exact inverse of block_mean on constants)."""
        return np.repeat(np.repeat(c, self.ratio, axis=-2), self.ratio, axis=-1)


class SmoothLift:
    """Bilinear interpolation of an nside_coarse field to the nside_fine pixel centres on the sphere
    (healpy's 4-neighbour weights), as a gather: ``fine[p] = sum_k w[p, k] * coarse[idx[p, k]]``.
    Used as the residual baseline and conditioner instead of the blocky nearest lift: a residual
    that has to cancel a 1.5 m/s step at every coarse-cell edge leaves a faint 1.8-degree grid in
    the output (LOG 2026-09-09); against a smooth baseline the residual is smooth and smaller.
    Block-mean consistency is then restored by the exact nearest projection, not by the lift."""

    def __init__(self, nside_fine: int, nside_coarse: int, cache_dir: str | Path = "~/data/hpx_layout") -> None:
        cache_dir = Path(cache_dir).expanduser(); p = cache_dir / f"interp_{nside_coarse}_to_{nside_fine}.npz"
        if p.exists():
            z = np.load(p); self.idx, self.w = z["idx"], z["w"]
        else:
            import healpy as hp
            theta, phi = hp.pix2ang(nside_fine, np.arange(12 * nside_fine * nside_fine), nest=True)
            pix, wts = hp.get_interp_weights(nside_coarse, theta, phi, nest=True)      # (4, npix) each
            self.idx, self.w = pix.T.astype(np.int64).copy(), wts.T.astype(np.float32).copy()
            cache_dir.mkdir(parents=True, exist_ok=True); np.savez(p, idx=self.idx, w=self.w)

    def gather(self, c: np.ndarray, fine_index: np.ndarray) -> np.ndarray:
        """c ``(..., npix_c)``, fine_index ``(P, P)`` NEST -> ``(..., P, P)`` interpolated values."""
        idx, w = self.idx[fine_index], self.w[fine_index]                                # (P, P, 4)
        out = np.zeros(c.shape[:-1] + fine_index.shape, dtype=np.float32)
        for k in range(4):
            out += c[..., idx[..., k]] * w[..., k]
        return out
